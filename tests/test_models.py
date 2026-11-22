"""Tests for PolicyModel log-probs and ReferenceManager, using a tiny random GPT-2."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from myrlx.core.models import ReferenceManager


def _ids(*rows):
    return torch.tensor(rows, dtype=torch.long)


def test_logprobs_shape_and_values(tiny_policy):
    input_ids = _ids([5, 9, 12, 7, 3], [4, 4, 8, 20, 11])
    attn = torch.ones_like(input_ids)
    lp = tiny_policy.logprobs(input_ids, attn)
    assert lp.shape == (2, 4)  # seq_len - 1

    # Independent check: position t's logits predict token t+1.
    with torch.no_grad():
        logits = tiny_policy.model(input_ids=input_ids, attention_mask=attn).logits
    expected = F.log_softmax(logits[:, :-1], dim=-1).gather(2, input_ids[:, 1:, None]).squeeze(-1)
    assert torch.allclose(lp, expected, atol=1e-6)
    assert (lp <= 0).all()


def test_sum_logprobs_counts_only_response_tokens(tiny_policy):
    # prompt = first 3 tokens, response = last 3 tokens
    input_ids = _ids([5, 9, 12, 7, 3, 1])
    attn = torch.ones_like(input_ids)
    loss_mask = _ids([0, 0, 0, 1, 1, 1])

    got = tiny_policy.sum_logprobs(input_ids, attn, loss_mask)

    # Written without the mask-shift trick: for each response token at index i,
    # add log p(token_i | tokens_<i), read from the logits at position i-1.
    with torch.no_grad():
        logp = F.log_softmax(tiny_policy.model(input_ids=input_ids, attention_mask=attn).logits, dim=-1)
    expected = sum(logp[0, i - 1, input_ids[0, i]] for i in (3, 4, 5))
    assert torch.allclose(got, expected.reshape(1), atol=1e-5)


def test_right_padding_does_not_change_sum_logprobs(tiny_policy):
    # The collator right-pads. Padding + attention_mask=0 + loss_mask=0 must not change the result.
    ids = _ids([5, 9, 12, 7, 3, 1])
    mask = _ids([0, 0, 1, 1, 1, 1])
    unpadded = tiny_policy.sum_logprobs(ids, torch.ones_like(ids), mask)

    padded_ids = _ids([5, 9, 12, 7, 3, 1, 0, 0, 0])
    padded_attn = _ids([1, 1, 1, 1, 1, 1, 0, 0, 0])
    padded_mask = _ids([0, 0, 1, 1, 1, 1, 0, 0, 0])
    padded = tiny_policy.sum_logprobs(padded_ids, padded_attn, padded_mask)

    assert torch.allclose(unpadded, padded, atol=1e-5)


def test_reference_matches_policy_at_init_and_is_frozen(tiny_policy):
    ref = ReferenceManager(tiny_policy)
    assert all(not p.requires_grad for p in ref.ref_model.parameters())
    assert not ref.ref_model.training

    ids = _ids([5, 9, 12, 7, 3, 1])
    attn = torch.ones_like(ids)
    mask = _ids([0, 0, 1, 1, 1, 1])
    assert torch.allclose(
        tiny_policy.sum_logprobs(ids, attn, mask), ref.sum_logprobs(ids, attn, mask), atol=1e-6
    )


def test_reference_unchanged_after_policy_update(tiny_policy):
    ref = ReferenceManager(tiny_policy)
    ids = _ids([5, 9, 12, 7, 3, 1])
    attn = torch.ones_like(ids)
    mask = _ids([0, 0, 1, 1, 1, 1])
    ref_before = ref.sum_logprobs(ids, attn, mask).clone()

    opt = torch.optim.SGD(tiny_policy.model.parameters(), lr=0.1)
    (-tiny_policy.sum_logprobs(ids, attn, mask).sum()).backward()
    opt.step()

    assert torch.allclose(ref.sum_logprobs(ids, attn, mask), ref_before)  # reference didn't move
    assert not torch.allclose(tiny_policy.sum_logprobs(ids, attn, mask), ref_before)  # policy did
