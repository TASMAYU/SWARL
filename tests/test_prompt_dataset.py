"""Tests for PromptDataset and the left-padding collate_prompt_batch (used by online RL)."""
from __future__ import annotations

from swarl.data.prompt_dataset import PromptDataset, collate_prompt_batch
from tests.conftest import PAD_ID


def test_item_keeps_text_ids_and_meta(tokenizer):
    ds = PromptDataset([{"prompt": "2+2?", "meta": {"answer": "4"}}, {"prompt": "hi"}], tokenizer)
    assert ds[0]["prompt_ids"] == tokenizer.encode_text("2+2?")
    assert ds[0]["prompt_text"] == "2+2?"
    assert ds[0]["meta"] == {"answer": "4"}
    assert ds[1]["meta"] == {}  # meta is optional


def test_prompt_truncation(tokenizer):
    ds = PromptDataset([{"prompt": "abcdefgh"}], tokenizer, max_prompt_length=3)
    assert ds[0]["prompt_ids"] == tokenizer.encode_text("abc")


def test_collator_left_pads(tokenizer):
    ds = PromptDataset([{"prompt": "ab"}, {"prompt": "abcde", "meta": {"k": 1}}], tokenizer)
    batch = collate_prompt_batch([ds[0], ds[1]], pad_token_id=PAD_ID)

    ids, attn = batch["input_ids"], batch["attention_mask"]
    assert ids.shape == attn.shape == (2, 5)
    # Short prompt is padded on the LEFT so both prompts end in the same column.
    assert ids[0].tolist() == [PAD_ID, PAD_ID, PAD_ID] + tokenizer.encode_text("ab")
    assert attn[0].tolist() == [0, 0, 0, 1, 1]
    assert ids[1].tolist() == tokenizer.encode_text("abcde")
    assert attn[1].tolist() == [1, 1, 1, 1, 1]
    assert batch["prompt_texts"] == ["ab", "abcde"]
    assert batch["meta"] == [{}, {"k": 1}]
