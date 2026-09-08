"""
Tests for training-time data handling: tokenization, response-only label
masking, config loading, and precision resolution.

The masking tests need the Qwen2.5 tokenizer (for its ChatML special tokens).
They are skipped automatically when it cannot be loaded (offline / no network).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.utils.config_utils import load_config


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
class TestTrainingConfigs:
    def test_lora_config_is_full_precision(self):
        cfg = load_config("configs/lora.yaml")
        assert cfg.model.load_in_4bit is False
        assert cfg.peft.r == 16
        assert "q_proj" in list(cfg.peft.target_modules)

    def test_model_and_dataset_revisions_are_pinned(self):
        cfg = load_config("configs/lora.yaml")  # inherits base.yaml
        assert len(str(cfg.model.revision)) == 40, "model must be pinned to a full commit SHA"
        assert len(str(cfg.data.dataset_revision)) == 40, "dataset must be pinned to a full commit SHA"

    def test_ablation_grid_present(self):
        cfg = load_config("configs/ablation.yaml")
        grid = cfg.ablation_grid
        assert list(grid.lora_rank) == [8, 16, 32]
        assert len(list(grid.learning_rate)) == 3


# ---------------------------------------------------------------------------
# Response-only label masking
# ---------------------------------------------------------------------------
class TestLabelMasking:
    EX = {
        "instruction": "What is compound interest?",
        "input": "",
        "output": "Compound interest is interest calculated on the initial principal "
                  "and also on the accumulated interest of prior periods.",
    }

    def test_only_response_tokens_are_unmasked(self, qwen_tokenizer):
        from datasets import Dataset

        from src.training.data import IGNORE_INDEX, build_tokenized_dataset

        ds = build_tokenized_dataset(Dataset.from_list([self.EX]), qwen_tokenizer, 512)
        row = ds[0]
        assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])

        unmasked = [t for t, lab in zip(row["input_ids"], row["labels"]) if lab != IGNORE_INDEX]
        assert unmasked, "no unmasked tokens"
        decoded = qwen_tokenizer.decode(unmasked)

        # The answer must be learned; the system prompt / instruction must not be.
        assert "Compound interest is interest calculated" in decoded
        assert "financial advisor assistant" not in decoded
        assert "What is compound interest?" not in decoded

    def test_masked_prefix_is_contiguous(self, qwen_tokenizer):
        from datasets import Dataset

        from src.training.data import IGNORE_INDEX, build_tokenized_dataset

        row = build_tokenized_dataset(Dataset.from_list([self.EX]), qwen_tokenizer, 512)[0]
        labels = row["labels"]
        first_unmasked = next(i for i, x in enumerate(labels) if x != IGNORE_INDEX)
        # everything before the first real label is masked...
        assert all(x == IGNORE_INDEX for x in labels[:first_unmasked])
        # ...and from there on labels mirror input_ids (teacher forcing on the answer)
        assert labels[first_unmasked:] == row["input_ids"][first_unmasked:]

    def test_sequence_respects_max_length(self, qwen_tokenizer):
        from datasets import Dataset

        from src.training.data import build_tokenized_dataset

        big = {"instruction": "Explain markets. " * 400, "input": "", "output": "Answer. " * 400}
        ds = build_tokenized_dataset(Dataset.from_list([big, self.EX]), qwen_tokenizer, 64)
        for row in ds:
            assert len(row["input_ids"]) <= 64

    def test_collator_pads_and_masks_padding(self, qwen_tokenizer):
        from datasets import Dataset

        from src.training.data import IGNORE_INDEX, CausalCollator, build_tokenized_dataset

        ex2 = dict(self.EX, output="Short answer about interest compounding over time here.")
        ds = build_tokenized_dataset(Dataset.from_list([self.EX, ex2]), qwen_tokenizer, 512)
        batch = CausalCollator(qwen_tokenizer)([ds[0], ds[1]])
        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
        # padded positions must be ignored in the loss
        pad_positions = batch["attention_mask"] == 0
        assert (batch["labels"][pad_positions] == IGNORE_INDEX).all()
