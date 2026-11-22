"""
DPO demo: fine-tune a small instruct model on real human-preference data
and show that it learns.

    python examples/train_dpo_demo.py                       # Qwen2.5-0.5B-Instruct, 512 train pairs
    python examples/train_dpo_demo.py --train-size 2000 --epochs 2
    python examples/train_dpo_demo.py --batch-size 2        # smaller GPUs (e.g. 16 GB)

What "it learns" means
----------------------
DPO's implicit reward is  r(y) = beta * log(pi(y|x) / pi_ref(y|x)).
On HELD-OUT pairs (never trained on) we measure:
  * reward accuracy: fraction of pairs where r(chosen) > r(rejected)
  * reward margin:   mean of r(chosen) - r(rejected)

Before training the policy equals the reference, so every reward is exactly 0
(accuracy 0.00, margin 0.000) -- that's the baseline by construction.
Success = held-out accuracy clearly above 0.5 and a positive margin after training.

Memory: the policy is trained in fp32 (see DPOTrainer for why). For a 0.5B model,
policy + reference + gradients + AdamW states is ~10 GB plus activations, so a
24 GB GPU is comfortable; use --batch-size 2 or --max-length 384 on 16 GB.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

# Allow running as `python examples/train_dpo_demo.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myrlx.losses.dpo import dpo_loss  # noqa: E402
from myrlx.trainers.offline import DPOTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SWARL DPO demo")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    p.add_argument("--split", default="train")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--eval-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--output-dir", default=None, help="save the trained model + tokenizer here")
    return p.parse_args()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------- data

def format_prompt(prompt_messages: list[dict], tokenizer) -> str:
    """Render the prompt with the model's chat template (instruct models expect it)."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    # Base models without a chat template: simple fallback format.
    text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in prompt_messages)
    return text + "\nAssistant:"


def to_record(row: dict, tokenizer) -> dict:
    """Convert one HF preference row into SWARL's {prompt, chosen, rejected} format.

    Handles the common layouts:
      * conversational, implicit prompt:  chosen/rejected = [..., user msg, assistant msg]
      * conversational, explicit prompt:  prompt = [msgs], chosen/rejected = [assistant msg]
      * plain strings:                    prompt / chosen / rejected are str
    """
    chosen, rejected, prompt = row["chosen"], row["rejected"], row.get("prompt")
    if isinstance(chosen, list):
        if isinstance(prompt, list) and prompt:
            prompt_msgs = prompt
        elif isinstance(prompt, str) and prompt:
            prompt_msgs = [{"role": "user", "content": prompt}]
        else:
            prompt_msgs = chosen[:-1]
        chosen_text, rejected_text = chosen[-1]["content"], rejected[-1]["content"]
    else:
        prompt_msgs = [{"role": "user", "content": prompt}]
        chosen_text, rejected_text = chosen, rejected
    return {"prompt": format_prompt(prompt_msgs, tokenizer), "chosen": chosen_text, "rejected": rejected_text}


def fits(record: dict, tokenizer, max_length: int) -> bool:
    """Keep only pairs that fit untruncated. Long prompts would otherwise truncate away the
    whole response (known truncation bug) and silently contribute nothing to training."""
    n_prompt = len(tokenizer(record["prompt"], add_special_tokens=False)["input_ids"])
    for key in ("chosen", "rejected"):
        n_resp = len(tokenizer(record[key], add_special_tokens=False)["input_ids"]) + 1  # + EOS
        if n_prompt + n_resp > max_length:
            return False
    return record["chosen"] != record["rejected"]


def load_records(args, tokenizer) -> tuple[list[dict], list[dict]]:
    print(f"Loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed)
    needed = args.train_size + args.eval_size
    records, skipped = [], 0
    for row in ds:
        rec = to_record(row, tokenizer)
        if fits(rec, tokenizer, args.max_length):
            records.append(rec)
            if len(records) == needed:
                break
        else:
            skipped += 1
    if len(records) < needed:
        raise SystemExit(f"Only {len(records)} usable pairs found (needed {needed}). "
                         f"Try a larger --max-length or smaller sizes.")
    print(f"Using {len(records)} pairs ({skipped} skipped as too long or identical).")
    return records[: args.train_size], records[args.train_size:]


# ----------------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(trainer: DPOTrainer, records: list[dict], batch_size: int, max_length: int) -> dict:
    """Held-out reward accuracy and margin. No parameter updates."""
    loader = trainer.make_dataloader(records, batch_size=batch_size, max_length=max_length)
    correct, margin_sum, loss_sum, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        b = {k: v.to(trainer.device) for k, v in batch.items()}
        pc = trainer.policy.sum_logprobs(b["chosen_ids"], b["chosen_attn"], b["chosen_loss_mask"])
        pr = trainer.policy.sum_logprobs(b["rejected_ids"], b["rejected_attn"], b["rejected_loss_mask"])
        rc = trainer.ref.sum_logprobs(b["chosen_ids"], b["chosen_attn"], b["chosen_loss_mask"])
        rr = trainer.ref.sum_logprobs(b["rejected_ids"], b["rejected_attn"], b["rejected_loss_mask"])
        loss, chosen_r, rejected_r = dpo_loss(pc, pr, rc, rr, beta=trainer.beta)
        bs = chosen_r.shape[0]
        correct += (chosen_r > rejected_r).float().sum().item()
        margin_sum += (chosen_r - rejected_r).sum().item()
        loss_sum += loss.item() * bs
        n += bs
    return {"loss": loss_sum / n, "reward_accuracy": correct / n, "reward_margin": margin_sum / n}


def fmt(stats: dict) -> str:
    return (f"loss {stats['loss']:.4f} | acc {stats['reward_accuracy']:.3f} | "
            f"margin {stats['reward_margin']:.4f}")


# ----------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    print(f"Model {args.model} on {device} | beta={args.beta} lr={args.lr} "
          f"batch={args.batch_size} epochs={args.epochs} max_length={args.max_length}")
    trainer = DPOTrainer(args.model, beta=args.beta, lr=args.lr, device=device)
    train_records, eval_records = load_records(args, trainer.policy.tokenizer)

    before = evaluate(trainer, eval_records, args.batch_size, args.max_length)
    print(f"[eval before] {fmt(before)}   (policy == reference, so margin is 0 by construction)")

    start = time.time()
    trainer.train(train_records, epochs=args.epochs, batch_size=args.batch_size,
                  log_every=args.log_every, max_length=args.max_length)
    minutes = (time.time() - start) / 60

    after = evaluate(trainer, eval_records, args.batch_size, args.max_length)
    print(f"[eval after ] {fmt(after)}   ({minutes:.1f} min of training)")

    if after["reward_accuracy"] > 0.5 and after["reward_margin"] > 0:
        print("PASS: held-out reward accuracy > 0.5 with a positive margin -- DPO is learning.")
    else:
        print("NOT YET: no clear held-out improvement. Try more data (--train-size), "
              "more epochs, or a higher --lr.")

    if args.output_dir:
        trainer.policy.model.save_pretrained(args.output_dir)
        trainer.policy.tokenizer.save_pretrained(args.output_dir)
        print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
