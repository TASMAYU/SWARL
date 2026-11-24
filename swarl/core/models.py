"""
Core model wrappers.

PolicyModel wraps a HF causal LM and exposes per-token log-probabilities --
the one primitive every algorithm in this library (DPO, PPO, GRPO...)
builds on top of.

ReferenceManager auto-clones and freezes a reference copy of the policy,
used for KL control so the model doesn't drift into gibberish during RL
training. This is deliberately its own class (not baked into PolicyModel)
so you can later swap "clone in memory" for something cheaper -- e.g.
re-loading base weights from disk, or LoRA-adapter-off inference on the
same base model -- without touching trainer code.
"""
from __future__ import annotations

import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities of the *actual* next tokens.

    Shared by PolicyModel and ReferenceManager so both always compute
    log-probs the exact same way (DPO compares them directly).

    Shape: (batch, seq_len - 1). Position t predicts token t+1, so the
    last logit has nothing to predict and the first token has no
    preceding prediction -- hence the seq_len - 1.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)


def masked_sum(token_logprobs: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Sum token log-probs over the response span only (loss_mask=1 there).

    loss_mask has the same shape as input_ids (1 over response tokens,
    0 over prompt tokens) -- shift it by one position to line up with the
    shifted log-probs.
    """
    mask = loss_mask[:, 1:].to(token_logprobs.dtype)
    return (token_logprobs * mask).sum(dim=-1)


class PolicyModel:
    """Thin wrapper around a HF CausalLM."""

    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype if device == "cuda" else torch.float32,
        ).to(device)

    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-token log-probabilities of the *actual* next tokens.

        Shape: (batch, seq_len - 1). Position t predicts token t+1, so the
        last logit has nothing to predict and the first token has no
        preceding prediction -- hence the seq_len - 1.
        """
        return compute_token_logprobs(self.model, input_ids, attention_mask)

    def sum_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      loss_mask: torch.Tensor) -> torch.Tensor:
        """Sum token logprobs over the response span only (loss_mask=1 there).

        loss_mask has the same shape as input_ids (1 over response tokens,
        0 over prompt tokens) -- we shift it by one position to line up
        with the shifted logprobs above.
        """
        token_lp = self.logprobs(input_ids, attention_mask)
        return masked_sum(token_lp, loss_mask)


class ReferenceManager:
    """Owns the frozen reference copy used for KL penalties in DPO/PPO/GRPO."""

    def __init__(self, policy: PolicyModel):
        self.ref_model = copy.deepcopy(policy.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def sum_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      loss_mask: torch.Tensor) -> torch.Tensor:
        token_lp = compute_token_logprobs(self.ref_model, input_ids, attention_mask)
        return masked_sum(token_lp, loss_mask)