"""
Preference dataset for offline pairwise algorithms (DPO and variants such as
IPO, SimPO, ORPO). Online methods like GRPO/DAPO use PromptDataset instead.

Expects records shaped like:
    {"prompt": str, "chosen": str, "rejected": str}

which matches the common HF preference-dataset convention (e.g.
Anthropic/hh-rlhf, trl-lib/ultrafeedback_binarized), so real datasets plug
in with zero transformation.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def _encode(self, prompt: str, response: str):
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        # 0 over the prompt, 1 over the response -- this is what
        # PolicyModel.sum_logprobs() uses to mask out prompt gradients.
        loss_mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[: self.max_length]
        return input_ids, loss_mask

    def __getitem__(self, idx):
        rec = self.records[idx]
        chosen_ids, chosen_mask = self._encode(rec["prompt"], rec["chosen"])
        rejected_ids, rejected_mask = self._encode(rec["prompt"], rec["rejected"])
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
        }


def collate_preference_batch(batch: list[dict], pad_token_id: int):
    """Pads chosen/rejected sequences separately -- they can be different lengths."""

    def pad_side(key_ids, key_mask):
        seqs = [torch.tensor(b[key_ids]) for b in batch]
        masks = [torch.tensor(b[key_mask]) for b in batch]
        max_len = max(s.size(0) for s in seqs)

        input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        loss_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)

        for i, (s, m) in enumerate(zip(seqs, masks)):
            input_ids[i, : s.size(0)] = s
            attn_mask[i, : s.size(0)] = 1
            loss_mask[i, : m.size(0)] = m
        return input_ids, attn_mask, loss_mask

    chosen_ids, chosen_attn, chosen_loss_mask = pad_side("chosen_ids", "chosen_mask")
    rejected_ids, rejected_attn, rejected_loss_mask = pad_side("rejected_ids", "rejected_mask")

    return {
        "chosen_ids": chosen_ids, "chosen_attn": chosen_attn, "chosen_loss_mask": chosen_loss_mask,
        "rejected_ids": rejected_ids, "rejected_attn": rejected_attn, "rejected_loss_mask": rejected_loss_mask,
    }