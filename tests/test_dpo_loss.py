"""Tests for myrlx.losses.dpo.dpo_loss against hand-computed values."""
from __future__ import annotations

import math

import pytest
import torch

from myrlx.losses.dpo import dpo_loss


def _t(*vals):
    return torch.tensor(vals, dtype=torch.float32)


def test_policy_equals_reference_gives_log2():
    # When policy == reference, every log-ratio is 0, so the loss is -log(sigmoid(0)) = log 2.
    lp = _t(-3.0, -7.5)
    loss, chosen_r, rejected_r = dpo_loss(lp, lp - 1.0, lp, lp - 1.0, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert torch.allclose(chosen_r, torch.zeros(2))
    assert torch.allclose(rejected_r, torch.zeros(2))


def test_hand_computed_value():
    # chosen log-ratio = (-1) - (-2) = 1, rejected log-ratio = (-3) - (-2) = -1
    # logits = beta * (1 - (-1)) = 0.5 * 2 = 1  ->  loss = log(1 + e^-1)
    loss, chosen_r, rejected_r = dpo_loss(_t(-1.0), _t(-3.0), _t(-2.0), _t(-2.0), beta=0.5)
    assert loss.item() == pytest.approx(math.log(1 + math.exp(-1)), abs=1e-6)
    assert chosen_r.item() == pytest.approx(0.5)
    assert rejected_r.item() == pytest.approx(-0.5)


def test_loss_is_mean_over_batch():
    a = dpo_loss(_t(-1.0), _t(-3.0), _t(-2.0), _t(-2.0), beta=0.5)[0]
    b = dpo_loss(_t(-2.0), _t(-2.0), _t(-2.0), _t(-2.0), beta=0.5)[0]
    both = dpo_loss(_t(-1.0, -2.0), _t(-3.0, -2.0), _t(-2.0, -2.0), _t(-2.0, -2.0), beta=0.5)[0]
    assert both.item() == pytest.approx((a.item() + b.item()) / 2, abs=1e-6)


def test_gradient_pushes_chosen_up_and_rejected_down():
    policy_chosen = _t(-2.0).requires_grad_()
    policy_rejected = _t(-2.0).requires_grad_()
    loss, _, _ = dpo_loss(policy_chosen, policy_rejected, _t(-2.0), _t(-2.0), beta=0.1)
    loss.backward()
    # Gradient descent subtracts the gradient: a negative grad raises chosen, a positive grad lowers rejected.
    assert policy_chosen.grad.item() < 0
    assert policy_rejected.grad.item() > 0


def test_loss_invariant_to_shared_shift():
    # The loss depends only on (policy - ref) differences: shifting both policy and ref
    # by the same constant must not change anything.
    base = dpo_loss(_t(-1.0), _t(-3.0), _t(-2.0), _t(-2.0), beta=0.3)[0]
    shifted = dpo_loss(_t(4.0), _t(2.0), _t(3.0), _t(3.0), beta=0.3)[0]
    assert base.item() == pytest.approx(shifted.item(), abs=1e-6)


def test_larger_beta_sharpens_loss():
    # With a positive margin, a larger beta makes the model "more confident" -> lower loss.
    args = (_t(-1.0), _t(-3.0), _t(-2.0), _t(-2.0))
    assert dpo_loss(*args, beta=1.0)[0] < dpo_loss(*args, beta=0.1)[0]
