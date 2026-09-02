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


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss.

    All four inputs are summed token log-probs over the response span,
    shape (batch,): policy and reference evaluations of the same
    chosen/rejected sequences (what PolicyModel.sum_logprobs and
    ReferenceManager.sum_logprobs produce).

    Returns:
        loss:           scalar, -log sigmoid(beta * (margin_c - margin_r))
        chosen_rewards: implicit reward beta * log(pi/pi_ref) on chosen
        rejected_rewards: same, on rejected -- useful for tracking the
            reward margin and accuracy without re-computing anything.
    """
    chosen_logratios = policy_chosen_lp - ref_chosen_lp      # log(pi/pi_ref), chosen
    rejected_logratios = policy_rejected_lp - ref_rejected_lp  # log(pi/pi_ref), rejected

    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    return loss, chosen_rewards, rejected_rewards
