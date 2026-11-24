"""
DPOTrainer: wires PolicyModel + ReferenceManager + PreferenceDataset +
dpo_loss into a real training loop. Deliberately the simplest trainer in
the library -- no generation, no rollouts -- so it validates the
model-loading/data/loss plumbing before the online (PPO/GRPO) trainer,
which reuses PolicyModel and ReferenceManager unchanged but adds a
generation step and an advantage estimator on top.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from functools import partial

from swarl.core.models import PolicyModel, ReferenceManager
from swarl.data.preference_dataset import PreferenceDataset, collate_preference_batch
from swarl.losses.dpo import dpo_loss


class DPOTrainer:
    def __init__(self, model_name: str, beta: float = 0.1, lr: float = 1e-6,
                 device: str = "cuda"):
        self.device = device
        self.policy = PolicyModel(model_name, device=device)
        self.ref = ReferenceManager(self.policy)  # auto clone + freeze
        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.policy.model.parameters(), lr=lr)

    def make_dataloader(self, records: list[dict], batch_size: int = 2):
        ds = PreferenceDataset(records, self.policy.tokenizer)
        collate = partial(collate_preference_batch, pad_token_id=self.policy.tokenizer.pad_token_id)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    def train_step(self, batch: dict) -> dict:
        chosen_ids = batch["chosen_ids"].to(self.device)
        chosen_attn = batch["chosen_attn"].to(self.device)
        chosen_mask = batch["chosen_loss_mask"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)
        rejected_attn = batch["rejected_attn"].to(self.device)
        rejected_mask = batch["rejected_loss_mask"].to(self.device)

        policy_chosen_lp = self.policy.sum_logprobs(chosen_ids, chosen_attn, chosen_mask)
        policy_rejected_lp = self.policy.sum_logprobs(rejected_ids, rejected_attn, rejected_mask)

        with torch.no_grad():
            ref_chosen_lp = self.ref.sum_logprobs(chosen_ids, chosen_attn, chosen_mask)
            ref_rejected_lp = self.ref.sum_logprobs(rejected_ids, rejected_attn, rejected_mask)

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=self.beta
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        return {
            "loss": loss.item(),
            "reward_accuracy": accuracy,
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

    def train(self, records: list[dict], epochs: int = 1, batch_size: int = 2, log_every: int = 1):
        loader = self.make_dataloader(records, batch_size=batch_size)
        step = 0
        for epoch in range(epochs):
            for batch in loader:
                stats = self.train_step(batch)
                if step % log_every == 0:
                    print(f"epoch {epoch} step {step} | "
                          f"loss {stats['loss']:.4f} | "
                          f"acc {stats['reward_accuracy']:.2f} | "
                          f"margin {stats['reward_margin']:.3f}")
                step += 1