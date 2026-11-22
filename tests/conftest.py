"""
Shared test fixtures.

Everything here runs on CPU in seconds with no network access:
- FakeTokenizer: a deterministic character-level tokenizer, so dataset tests
  don't need to download a real tokenizer.
- tiny_model: a randomly initialised 2-layer GPT-2 (vocab 64), so model /
  trainer tests exercise real HF forward passes without downloading weights.
"""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from myrlx.core.models import PolicyModel, ReferenceManager
from myrlx.trainers.offline import DPOTrainer

VOCAB_SIZE = 64
PAD_ID = 0
EOS_ID = 1


class FakeTokenizer:
    """Character-level tokenizer: each character maps to an id in [2, VOCAB_SIZE)."""

    pad_token_id = PAD_ID
    eos_token_id = EOS_ID
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode_text(self, text: str) -> list[int]:
        return [2 + (ord(c) % (VOCAB_SIZE - 2)) for c in text]

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": self.encode_text(text)}


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
    )
    return GPT2LMHeadModel(config).eval()


def make_policy(model, tokenizer, device: str = "cpu") -> PolicyModel:
    """Build a PolicyModel around an in-memory model, bypassing the HF Hub download."""
    policy = PolicyModel.__new__(PolicyModel)
    policy.device = device
    policy.model = model.to(device)
    policy.tokenizer = tokenizer
    return policy


@pytest.fixture
def tiny_policy(tiny_model, tokenizer) -> PolicyModel:
    return make_policy(tiny_model, tokenizer)


@pytest.fixture
def tiny_trainer(tiny_policy) -> DPOTrainer:
    """A DPOTrainer wired to the tiny model (mirrors DPOTrainer.__init__ without the download)."""
    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.device = "cpu"
    trainer.policy = tiny_policy
    trainer.ref = ReferenceManager(tiny_policy)
    trainer.beta = 0.5
    trainer.optimizer = torch.optim.AdamW(tiny_policy.model.parameters(), lr=1e-3)
    return trainer
