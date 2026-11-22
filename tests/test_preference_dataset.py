"""Tests for PreferenceDataset encoding/masking and collate_preference_batch padding."""
from __future__ import annotations

import pytest
import torch

from myrlx.data.preference_dataset import PreferenceDataset, collate_preference_batch
from tests.conftest import EOS_ID, PAD_ID


def test_loss_mask_covers_response_and_eos_only(tokenizer):
    ds = PreferenceDataset([{"prompt": "abc", "chosen": "xy", "rejected": "z"}], tokenizer)
    item = ds[0]

    assert item["chosen_ids"] == tokenizer.encode_text("abc") + tokenizer.encode_text("xy") + [EOS_ID]
    assert item["chosen_mask"] == [0, 0, 0, 1, 1, 1]
    assert item["rejected_ids"] == tokenizer.encode_text("abc") + tokenizer.encode_text("z") + [EOS_ID]
    assert item["rejected_mask"] == [0, 0, 0, 1, 1]


def test_max_length_truncates_ids_and_mask_together(tokenizer):
    ds = PreferenceDataset([{"prompt": "abc", "chosen": "wxyz", "rejected": "q"}], tokenizer, max_length=5)
    item = ds[0]
    assert len(item["chosen_ids"]) == len(item["chosen_mask"]) == 5
    assert item["chosen_mask"] == [0, 0, 0, 1, 1]


@pytest.mark.xfail(
    strict=True,
    reason="Known bug (PLAN.md §9): a prompt longer than max_length truncates away the whole "
    "response, leaving an all-zero loss mask. Remove this xfail once truncation is fixed.",
)
def test_long_prompt_keeps_some_response(tokenizer):
    ds = PreferenceDataset([{"prompt": "a" * 20, "chosen": "xy", "rejected": "z"}], tokenizer, max_length=10)
    assert sum(ds[0]["chosen_mask"]) > 0


def test_collator_pads_chosen_and_rejected_separately(tokenizer):
    records = [
        {"prompt": "ab", "chosen": "long answer", "rejected": "no"},
        {"prompt": "abcd", "chosen": "ok", "rejected": "a much longer rejected answer"},
    ]
    ds = PreferenceDataset(records, tokenizer)
    batch = collate_preference_batch([ds[0], ds[1]], pad_token_id=PAD_ID)

    for side in ("chosen", "rejected"):
        ids, attn, loss = batch[f"{side}_ids"], batch[f"{side}_attn"], batch[f"{side}_loss_mask"]
        assert ids.shape == attn.shape == loss.shape
        assert ids.dtype == attn.dtype == loss.dtype == torch.long
        for row, item in enumerate((ds[0], ds[1])):
            n = len(item[f"{side}_ids"])
            assert ids[row, :n].tolist() == item[f"{side}_ids"]
            assert attn[row, :n].sum() == n and attn[row, n:].sum() == 0  # right padding
            assert (ids[row, n:] == PAD_ID).all()
            assert loss[row, :n].tolist() == item[f"{side}_mask"]
            assert loss[row, n:].sum() == 0  # padding never contributes to the loss

    # Each side is padded to its own longest sequence, not a shared length.
    assert batch["chosen_ids"].shape[1] == max(len(ds[i]["chosen_ids"]) for i in range(2))
    assert batch["rejected_ids"].shape[1] == max(len(ds[i]["rejected_ids"]) for i in range(2))
