"""
Tests for the data preprocessing pipeline.

Verifies:
- Prompt template formatting correctness
- No data leakage between train/val/test splits
- Token length filtering logic
- Deduplication logic
- Stat card saving
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from datasets import Dataset

from src.data.make_dataset import (
    _hash_example,
    _normalized_key,
    _verify_no_overlap,
    count_normalized_near_duplicates,
    create_splits,
    deduplicate,
    filter_by_output_length,
    filter_empty_fields,
    save_splits,
)
from src.data.prompt_template import (
    SYSTEM_PROMPT,
    build_formatted_example,
    format_for_inference,
    format_for_training,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_examples():
    """A small sample of finance-alpaca style examples."""
    return [
        {"instruction": "What is a mutual fund?", "input": "", "output": "A mutual fund pools money from many investors."},
        {"instruction": "Explain inflation.", "input": "", "output": "Inflation is the rate of increase in prices over time."},
        {"instruction": "What is compound interest?", "input": "Assume monthly compounding.", "output": "Compound interest earns interest on both the principal and accumulated interest."},
        {"instruction": "Define bull market.", "input": "", "output": "A bull market is a period of rising stock prices."},
        {"instruction": "What is a bond?", "input": "", "output": "A bond is a fixed income instrument representing a loan from an investor to a borrower."},
    ]


@pytest.fixture
def sample_dataset(sample_examples):
    """HuggingFace Dataset from the sample examples."""
    return Dataset.from_list(sample_examples)


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------
class TestPromptTemplate:
    def test_format_for_inference_no_input(self, sample_examples):
        ex = sample_examples[0]  # no 'input'
        prompt = format_for_inference(ex)
        assert ex["instruction"] in prompt
        assert SYSTEM_PROMPT in prompt
        assert "<|im_start|>" in prompt
        assert "<|im_end|>" in prompt
        # Inference prompt must NOT contain the expected output
        assert ex["output"] not in prompt

    def test_format_for_inference_with_input(self, sample_examples):
        ex = sample_examples[2]  # has 'input'
        prompt = format_for_inference(ex)
        assert ex["instruction"] in prompt
        assert ex["input"] in prompt
        assert ex["output"] not in prompt

    def test_format_for_training_contains_output(self, sample_examples):
        ex = sample_examples[0]
        full = format_for_training(ex)
        assert ex["instruction"] in full
        assert ex["output"] in full
        # Training text must be longer than inference prompt
        assert len(full) > len(format_for_inference(ex))

    def test_build_formatted_example(self, sample_examples):
        ex_no_input = sample_examples[0]
        formatted = build_formatted_example(ex_no_input)
        assert not formatted.has_input
        assert formatted.instruction == ex_no_input["instruction"]
        assert formatted.output == ex_no_input["output"]
        assert formatted.output in formatted.full_text
        assert formatted.output not in formatted.prompt

        ex_with_input = sample_examples[2]
        formatted2 = build_formatted_example(ex_with_input)
        assert formatted2.has_input
        assert formatted2.input == ex_with_input["input"]

    def test_prompt_is_deterministic(self, sample_examples):
        """Same input should always produce the same prompt."""
        ex = sample_examples[1]
        assert format_for_inference(ex) == format_for_inference(ex)
        assert format_for_training(ex) == format_for_training(ex)


# ---------------------------------------------------------------------------
# Dataset filtering tests
# ---------------------------------------------------------------------------
class TestDatasetFiltering:
    def test_filter_empty_instruction(self, sample_dataset):
        """Examples with empty instruction should be removed."""
        with_empty = Dataset.from_list([
            {"instruction": "", "input": "", "output": "some answer"},
            {"instruction": "valid", "input": "", "output": "valid answer"},
        ])
        filtered = filter_empty_fields(with_empty)
        assert len(filtered) == 1
        assert filtered[0]["instruction"] == "valid"

    def test_filter_empty_output(self, sample_dataset):
        """Examples with empty output should be removed."""
        with_empty = Dataset.from_list([
            {"instruction": "valid question", "input": "", "output": ""},
            {"instruction": "valid question2", "input": "", "output": "valid answer"},
        ])
        filtered = filter_empty_fields(with_empty)
        assert len(filtered) == 1

    def test_filter_by_output_length_removes_short(self):
        data = Dataset.from_list([
            {"instruction": "Q1", "input": "", "output": "Short"},  # 1 word
            {"instruction": "Q2", "input": "", "output": " ".join(["word"] * 10)},  # 10 words
            {"instruction": "Q3", "input": "", "output": " ".join(["word"] * 700)},  # too long
        ])
        filtered = filter_by_output_length(data, min_words=5, max_words=600)
        assert len(filtered) == 1
        assert "word word" in filtered[0]["output"]

    def test_deduplication_removes_duplicates(self):
        data = Dataset.from_list([
            {"instruction": "What is X?", "input": "", "output": "X is Y"},
            {"instruction": "What is X?", "input": "", "output": "X is Y (duplicate)"},  # same instruction+input
            {"instruction": "What is Z?", "input": "", "output": "Z is Q"},
        ])
        deduped, n_removed = deduplicate(data)
        assert len(deduped) == 2
        assert n_removed == 1

    def test_deduplication_keeps_first(self):
        data = Dataset.from_list([
            {"instruction": "Dup?", "input": "", "output": "first"},
            {"instruction": "Dup?", "input": "", "output": "second"},
        ])
        deduped, _ = deduplicate(data)
        assert deduped[0]["output"] == "first"

    def test_hash_is_case_insensitive(self):
        h1 = _hash_example("What is X?", "")
        h2 = _hash_example("what is x?", "")
        assert h1 == h2

    def test_normalized_near_duplicate_detection(self):
        data = Dataset.from_list([
            {"instruction": "What is a bond?", "input": "", "output": "A loan to a borrower."},
            {"instruction": "What is a bond???", "input": "", "output": "A loan  to a borrower!"},
            {"instruction": "What is a stock?", "input": "", "output": "Ownership in a company."},
        ])
        # exact (instruction,input) dedup would NOT catch rows 0/1 (different punctuation)
        _, exact = deduplicate(data)
        assert exact == 0
        # normalized key ignores punctuation/whitespace -> 1 collision
        assert count_normalized_near_duplicates(data) == 1

    def test_normalized_key_ignores_punctuation(self):
        assert _normalized_key("A, B!", "", "c.") == _normalized_key("a b", "", "C")


class TestArtifacts:
    def test_save_splits_writes_provenance_files(self, tmp_path):
        ds = Dataset.from_list([
            {"instruction": f"Q{i}", "input": "", "output": f"answer number {i} here"}
            for i in range(60)
        ])
        splits = create_splits(ds, test_fraction=0.1, val_fraction=0.05, seed=42)
        save_splits(splits, tmp_path, seed=42, fractions={"test": 0.1})

        for name in ("stats.json", "splits_manifest.json", "leakage_report.json",
                     "train.json", "validation.json", "test.json"):
            assert (tmp_path / name).exists(), name

        import json
        leak = json.loads((tmp_path / "leakage_report.json").read_text())
        assert leak["leakage_detected"] is False
        manifest = json.loads((tmp_path / "splits_manifest.json").read_text())
        assert set(manifest["split_membership_sha256"]) == {"train", "validation", "test"}


# ---------------------------------------------------------------------------
# Data split tests
# ---------------------------------------------------------------------------
class TestDataSplits:
    def _make_large_dataset(self, n=200):
        return Dataset.from_list([
            {"instruction": f"Q{i}", "input": "", "output": f"A{i}" * 10}
            for i in range(n)
        ])

    def test_split_sizes_are_correct(self):
        ds = self._make_large_dataset(200)
        splits = create_splits(ds, test_fraction=0.10, val_fraction=0.05, seed=42)
        total = sum(len(v) for v in splits.values())
        assert total == 200
        # Test set should be ~10%
        assert 18 <= len(splits["test"]) <= 22

    def test_no_train_test_overlap(self):
        ds = self._make_large_dataset(200)
        splits = create_splits(ds, test_fraction=0.10, val_fraction=0.05, seed=42)
        # This should NOT raise
        _verify_no_overlap(splits["train"], splits["validation"], splits["test"])

    def test_split_is_reproducible(self):
        """Same seed must produce identical splits."""
        ds = self._make_large_dataset(100)
        s1 = create_splits(ds, seed=42)
        s2 = create_splits(ds, seed=42)
        assert list(s1["test"]["instruction"]) == list(s2["test"]["instruction"])

    def test_split_with_different_seeds_differs(self):
        ds = self._make_large_dataset(100)
        s1 = create_splits(ds, seed=42)
        s2 = create_splits(ds, seed=99)
        # With high probability, different seeds produce different test sets
        assert list(s1["test"]["instruction"]) != list(s2["test"]["instruction"])
