"""End-to-end smoke tests for DPOTrainer on a tiny random model (CPU, no downloads)."""
from __future__ import annotations

import math

import pytest
import torch

RECORDS = [
    {"prompt": "Q: 2+2? A:", "chosen": " 4", "rejected": " 5"},
    {"prompt": "Q: capital of France? A:", "chosen": " Paris", "rejected": " Rome"},
]


def test_first_step_starts_at_log2(tiny_trainer):
    # Policy == reference before any update, so the very first loss must be log 2
    # with zero margin and zero accuracy (ties count as not-correct).
    batch = next(iter(tiny_trainer.make_dataloader(RECORDS, batch_size=2)))
    stats = tiny_trainer.train_step(batch)
    assert stats["loss"] == pytest.approx(math.log(2), abs=1e-4)
    assert stats["reward_margin"] == pytest.approx(0.0, abs=1e-4)


def test_overfits_a_fixed_batch(tiny_trainer):
    torch.manual_seed(0)
    batch = next(iter(tiny_trainer.make_dataloader(RECORDS, batch_size=2)))
    first = tiny_trainer.train_step(batch)
    for _ in range(30):
        last = tiny_trainer.train_step(batch)
    assert last["loss"] < first["loss"] - 0.2
    assert last["reward_accuracy"] == 1.0
    assert last["reward_margin"] > 0


def test_reference_stays_frozen_during_training(tiny_trainer):
    before = [p.detach().clone() for p in tiny_trainer.ref.ref_model.parameters()]
    tiny_trainer.train(RECORDS, epochs=2, batch_size=2, log_every=100)
    after = list(tiny_trainer.ref.ref_model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(after, before))


def test_default_dtype_is_float32():
    # Regression guard: pure-bf16 weights make AdamW updates at DPO-sized
    # learning rates round to zero, so the trainer must default to fp32.
    import inspect

    from swarl.trainers.offline import DPOTrainer

    assert inspect.signature(DPOTrainer.__init__).parameters["dtype"].default is torch.float32


def test_max_length_is_passed_through(tiny_trainer):
    # Regression guard: examples/train_dpo_demo.py passes max_length to both
    # make_dataloader and train; dropping it would crash the demo.
    loader = tiny_trainer.make_dataloader(RECORDS, batch_size=2, max_length=8)
    assert loader.dataset.max_length == 8
    tiny_trainer.train(RECORDS, epochs=1, batch_size=2, log_every=100, max_length=8)
