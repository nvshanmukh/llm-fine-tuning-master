"""
Tokenization + response-only label masking for supervised fine-tuning.

We deliberately do NOT depend on TRL's SFTTrainer / collators (their API has
changed repeatedly across major versions). Instead we pre-tokenize each example
here and build a ``labels`` tensor where every token up to and including the
assistant tag is set to -100, so the loss is computed only on the response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.data.prompt_template import RESPONSE_TAG, format_for_inference, format_for_training

IGNORE_INDEX = -100


def _first_sublist_index(haystack: list[int], needle: list[int]) -> int:
    """Return the index just AFTER the first occurrence of ``needle`` in ``haystack``, or -1."""
    if not needle:
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i + len(needle)
    return -1


def build_tokenized_dataset(dataset, tokenizer, max_seq_length: int):
    """
    Map a raw {instruction,input,output} dataset to {input_ids, attention_mask, labels}.

    Examples whose response would be fully truncated are dropped (they carry no
    learnable signal). The number dropped is logged.
    """
    response_ids = tokenizer(RESPONSE_TAG, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    def _encode(example: dict) -> dict:
        full = format_for_training(example)
        enc = tokenizer(
            full,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
            input_ids = input_ids[:max_seq_length - 1] + [eos_id]
        attention_mask = [1] * len(input_ids)

        split = _first_sublist_index(input_ids, response_ids)
        labels = list(input_ids)
        if split == -1:
            # response tag was truncated away -> nothing to learn from
            labels = [IGNORE_INDEX] * len(input_ids)
        else:
            for i in range(min(split, len(labels))):
                labels[i] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "_has_signal": any(x != IGNORE_INDEX for x in labels),
        }

    tokenized = dataset.map(_encode, desc="Tokenizing + masking")
    before = len(tokenized)
    tokenized = tokenized.filter(lambda x: x["_has_signal"])
    dropped = before - len(tokenized)
    if dropped:
        logger.warning(f"Dropped {dropped} examples with no unmasked response tokens")
    tokenized = tokenized.remove_columns(
        [c for c in tokenized.column_names
         if c not in ("input_ids", "attention_mask", "labels")]
    )
    return tokenized


@dataclass
class CausalCollator:
    """Dynamic padding collator for causal LM with a masked ``labels`` field."""

    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict]) -> dict:
        import torch

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        input_ids, attn, labels = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n)
            attn.append(f["attention_mask"] + [0] * n)
            labels.append(f["labels"] + [IGNORE_INDEX] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def masking_report(example: dict, tokenizer, max_seq_length: int = 512) -> dict:
    """Human-readable check that masking keeps only the response tokens (used in tests)."""
    from datasets import Dataset

    tok = build_tokenized_dataset(Dataset.from_list([example]), tokenizer, max_seq_length)[0]
    unmasked = [t for t, lab in zip(tok["input_ids"], tok["labels"]) if lab != IGNORE_INDEX]
    decoded = tokenizer.decode(unmasked)
    prompt_only = format_for_inference(example)
    return {
        "unmasked_decoded": decoded,
        "prompt_not_in_unmasked": prompt_only.strip()[:30] not in decoded,
        "output_in_unmasked": example["output"].strip()[:30] in decoded,
    }
