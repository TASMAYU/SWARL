"""
Prompt dataset for online algorithms (PPO, GRPO, REINFORCE).

Unlike PreferenceDataset, there's no "chosen"/"rejected" here -- responses
don't exist yet at data-loading time. They get generated live during
training by the generation engine (core/generation.py), scored by a reward
function (rewards/), and only then combined into a TextRLBatch
(core/batch.py) for the loss to consume.

Expects records shaped like:
    {"prompt": str, "meta": dict}   # meta is optional, e.g. {"answer": "42"}
                                     # for verifiable-reward tasks like GSM8K

This file only prepares the *input* side. Nothing here computes rewards or
advantages -- that split is intentional so this dataset works unchanged
whether you're doing PPO, GRPO, or REINFORCE; only the loss/trainer differs.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_prompt_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt_ids = self.tokenizer(
            rec["prompt"], add_special_tokens=False
        )["input_ids"][: self.max_prompt_length]
        return {
            "prompt_ids": prompt_ids,
            "prompt_text": rec["prompt"],       # kept as raw text -- the
                                                  # generation engine (vLLM/HF)
                                                  # takes text, not just ids
            "meta": rec.get("meta", {}),         # e.g. ground-truth answer,
                                                  # test cases, for verifiers
        }


def collate_prompt_batch(batch: list[dict], pad_token_id: int):
    """Left-pads prompts -- generation needs prompts right-aligned so all
    sequences in the batch start generating from the same column."""
    seqs = [torch.tensor(b["prompt_ids"]) for b in batch]
    max_len = max(s.size(0) for s in seqs)

    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        input_ids[i, max_len - s.size(0):] = s   # left-pad
        attn_mask[i, max_len - s.size(0):] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "prompt_texts": [b["prompt_text"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }