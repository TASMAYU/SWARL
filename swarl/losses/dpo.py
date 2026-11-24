"""
Loss functions for preference-optimization algorithms.

dpo_loss is the DPO objective (Rafailov et al., 2023). The key identity it
relies on: under a KL-constrained reward objective, the optimal policy
satisfies r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + const, so the
reward can be read off directly from log-prob ratios of *data* -- no reward
model, no rollouts. Plugging that implicit reward into the Bradley-Terry
preference likelihood gives the loss below.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_TYPES = ("sigmoid", "robust", "hinge", "ipo")


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss.

    All four inputs are summed token log-probs over the response span,
    shape (batch,): policy and reference evaluations of the same
    chosen/rejected sequences (what PolicyModel.sum_logprobs and
    ReferenceManager.sum_logprobs produce).

    loss_type: "sigmoid" (standard DPO, default), "ipo", "hinge" or "robust".
    label_smoothing: assumed label-noise rate in [0, 0.5); used by "sigmoid"
        (conservative DPO) and "robust". 0 = trust every label.

    Returns:
        loss:           scalar, mean over the batch of the chosen loss_type
        chosen_rewards: implicit reward beta * log(pi/pi_ref) on chosen
        rejected_rewards: same, on rejected -- useful for tracking the
            reward margin and accuracy without re-computing anything.
    """
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")
    chosen_logratios = policy_chosen_lp - ref_chosen_lp      # log(pi/pi_ref), chosen
    rejected_logratios = policy_rejected_lp - ref_rejected_lp  # log(pi/pi_ref), rejected

    # h: how much more the policy prefers chosen over rejected, relative to the reference
    h =chosen_logratios - rejected_logratios
    logits = beta * h

    if loss_type == "sigmoid":
        # Standard DPO. label_smoothing > 0 gives conservative DPO (cDPO).
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    elif loss_type == "robust":
        # Robust DPO (Chowdhury et al., 2024): unbiased when a fraction
        # label_smoothing of the preference labels are flipped.
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            + F.logsigmoid(-logits) * label_smoothing
        ) / (1 - 2 * label_smoothing)
    elif loss_type == "hinge":
        # SLiC-style hinge: zero loss once beta * h >= 1.
        losses = torch.relu(1 - logits)
    elif loss_type == "ipo":
        # IPO (Azar et al., 2023): regress h toward the target 1 / (2 * beta).
        losses = (h - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}; expected one of {LOSS_TYPES}")
    loss = losses.mean()

    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    return loss, chosen_rewards, rejected_rewards
