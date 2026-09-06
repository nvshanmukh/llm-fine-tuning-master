"""
Dataset preparation pipeline for the Finance LLM fine-tuning project.

This module:
1. Downloads the gbharti/finance-alpaca dataset from HuggingFace.
2. Performs data quality filtering (token length, duplicates, empty fields).
3. Creates reproducible train/validation/test splits.
4. Saves processed datasets to disk in JSON format.
5. Logs dataset statistics and data cards.

Design Principles:
- Test set is NEVER exposed to training/validation processes.
- All splits are created with a fixed seed for reproducibility.
- Deduplication uses instruction+input hash to prevent cross-split leakage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from transformers import AutoTokenizer

from src.data.prompt_template import format_for_training
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "gbharti/finance-alpaca"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MIN_OUTPUT_WORDS = 5
DEFAULT_MAX_OUTPUT_WORDS = 600
RANDOM_SEED = 42


def _hash_example(instruction: str, inp: str) -> str:
    """Create a deterministic hash for deduplication using instruction + input."""
    raw = f"{instruction.strip().lower()}|||{inp.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_raw_dataset(
    dataset_name: str = DATASET_NAME,
    cache_dir: str | None = None,
    hf_token: str | None = None,
) -> Dataset:
    """
    Load the raw finance-alpaca dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        cache_dir: Local directory to cache raw data.
        hf_token: HuggingFace API token (required for gated datasets).

    Returns:
        HuggingFace Dataset object (the 'train' split, as this dataset has only one split).
    """
    logger.info(f"Loading dataset: {dataset_name}")
    raw = load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    # finance-alpaca has a single 'train' split
    dataset = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
    logger.info(f"Raw dataset loaded: {len(dataset):,} examples")
    logger.info(f"Columns: {dataset.column_names}")
    return dataset


def filter_empty_fields(dataset: Dataset) -> Dataset:
    """
    Remove examples with empty instruction or output fields.

    Args:
        dataset: Input dataset.

    Returns:
        Filtered dataset.
    """
    before = len(dataset)
    dataset = dataset.filter(
        lambda x: (
            bool(x.get("instruction", "").strip())
            and bool(x.get("output", "").strip())
        )
    )
    removed = before - len(dataset)
    logger.info(f"Empty field filter: removed {removed:,} | remaining {len(dataset):,}")
    return dataset


def filter_by_output_length(
    dataset: Dataset,
    min_words: int = DEFAULT_MIN_OUTPUT_WORDS,
    max_words: int = DEFAULT_MAX_OUTPUT_WORDS,
) -> Dataset:
    """
    Filter examples based on output word count.
    Removes very short outputs (likely garbage) and very long outputs
    that would dominate context windows.

    Args:
        dataset: Input dataset.
        min_words: Minimum number of words in output.
        max_words: Maximum number of words in output.

    Returns:
        Filtered dataset.
    """
    before = len(dataset)
    dataset = dataset.filter(
        lambda x: (
            min_words <= len(x.get("output", "").split()) <= max_words
        )
    )
    removed = before - len(dataset)
    logger.info(
        f"Output length filter (words {min_words}-{max_words}): "
        f"removed {removed:,} | remaining {len(dataset):,}"
    )
    return dataset


def filter_by_token_length(
    dataset: Dataset,
    tokenizer_name: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
) -> Dataset:
    """
    Filter examples where the full formatted prompt+response exceeds
    the model's maximum sequence length.

    This prevents truncation from silently corrupting training examples.

    Args:
        dataset: Input dataset.
        tokenizer_name: HuggingFace model name for tokenization.
        max_seq_length: Maximum allowed token count for full text.

    Returns:
        Filtered dataset with token lengths added as a column.
    """
    logger.info(f"Loading tokenizer for length filtering: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )

    def _compute_token_length(example: dict) -> dict:
        full_text = format_for_training(example)
        tokens = tokenizer(full_text, truncation=False)
        return {"token_length": len(tokens["input_ids"])}

    logger.info("Computing token lengths (this may take a moment)...")
    dataset = dataset.map(_compute_token_length, desc="Computing token lengths")

    # Log distribution statistics
    lengths = dataset["token_length"]
    logger.info(
        f"Token length stats: mean={np.mean(lengths):.0f}, "
        f"median={np.median(lengths):.0f}, "
        f"p95={np.percentile(lengths, 95):.0f}, "
        f"max={max(lengths)}"
    )

    before = len(dataset)
    dataset = dataset.filter(lambda x: x["token_length"] <= max_seq_length)
    removed = before - len(dataset)
    logger.info(
        f"Token length filter (<={max_seq_length}): "
        f"removed {removed:,} | remaining {len(dataset):,}"
    )
    return dataset


def deduplicate(dataset: Dataset) -> tuple[Dataset, int]:
    """
    Remove duplicate examples using a hash of (instruction, input).
    When duplicates exist, we keep the first occurrence.

    Args:
        dataset: Input dataset.

    Returns:
        Tuple of (deduplicated dataset, number of duplicates removed).
    """
    seen_hashes: set[str] = set()
    unique_indices: list[int] = []

    for i, example in enumerate(dataset):
        h = _hash_example(example.get("instruction", ""), example.get("input", ""))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

    before = len(dataset)
    dataset = dataset.select(unique_indices)
    duplicates_removed = before - len(dataset)
    logger.info(f"Deduplication: removed {duplicates_removed:,} | remaining {len(dataset):,}")
    return dataset, duplicates_removed


def create_splits(
    dataset: Dataset,
    test_fraction: float = 0.10,
    val_fraction: float = 0.05,
    seed: int = RANDOM_SEED,
) -> DatasetDict:
    """
    Create reproducible train/validation/test splits.

    IMPORTANT: The test split is created FIRST, before any train/val split,
    ensuring there is absolutely no data leakage between test and train.

    Split methodology:
    1. Shuffle the full dataset with fixed seed.
    2. Extract test_fraction as held-out test set.
    3. From remaining data, extract val_fraction as validation set.
    4. The rest becomes the training set.

    Args:
        dataset: Fully preprocessed dataset (deduped, filtered).
        test_fraction: Fraction of data to hold out for final test evaluation.
        val_fraction: Fraction of non-test data to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits.
    """
    logger.info(
        f"Creating splits: test={test_fraction:.0%}, val={val_fraction:.0%}"
    )

    # Step 1: Split off test set first
    train_test = dataset.train_test_split(
        test_size=test_fraction,
        seed=seed,
        shuffle=True,
    )
    test_ds = train_test["test"]
    remaining = train_test["train"]

    # Step 2: Split val from remaining (val_fraction is relative to full dataset)
    effective_val_fraction = val_fraction / (1.0 - test_fraction)
    train_val = remaining.train_test_split(
        test_size=effective_val_fraction,
        seed=seed,
        shuffle=True,
    )
    train_ds = train_val["train"]
    val_ds = train_val["test"]

    # Step 3: Verify no overlap
    _verify_no_overlap(train_ds, val_ds, test_ds)

    splits = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    logger.info(
        f"Splits created: train={len(train_ds):,}, "
        f"val={len(val_ds):,}, test={len(test_ds):,}"
    )
    return splits


def _verify_no_overlap(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
) -> None:
    """
    Verify that train/val/test sets have no overlapping examples.
    Raises AssertionError if overlap is found.

    Args:
        train_ds, val_ds, test_ds: The three split datasets.
    """
    def _get_hashes(ds: Dataset) -> set:
        return {
            _hash_example(ex["instruction"], ex.get("input", ""))
            for ex in ds
        }

    train_hashes = _get_hashes(train_ds)
    val_hashes = _get_hashes(val_ds)
    test_hashes = _get_hashes(test_ds)

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes

    assert len(train_test_overlap) == 0, (
        f"DATA LEAKAGE: {len(train_test_overlap)} examples overlap between TRAIN and TEST"
    )
    assert len(train_val_overlap) == 0, (
        f"DATA LEAKAGE: {len(train_val_overlap)} examples overlap between TRAIN and VAL"
    )
    assert len(val_test_overlap) == 0, (
        f"DATA LEAKAGE: {len(val_test_overlap)} examples overlap between VAL and TEST"
    )
    logger.info("Data leakage check: PASSED - no overlap between splits")


def save_splits(
    splits: DatasetDict,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save DatasetDict splits to disk as JSON files.
    Also saves a data card (stats.json) with dataset metadata.

    Args:
        splits: DatasetDict with train/validation/test.
        output_dir: Directory to save processed data.

    Returns:
        Dict mapping split names to saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for split_name, ds in splits.items():
        out_path = output_dir / f"{split_name}.json"
        ds.to_json(str(out_path))
        saved_paths[split_name] = out_path
        logger.info(f"Saved {split_name}: {len(ds):,} examples -> {out_path}")

    # Save data card
    stats = {
        "dataset_source": DATASET_NAME,
        "license": "Apache 2.0 / CC-BY-4.0 (finance-alpaca)",
        "splits": {k: len(v) for k, v in splits.items()},
        "total_examples": sum(len(v) for v in splits.values()),
        "seed": RANDOM_SEED,
    }
    stats_path = output_dir / "stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset stats saved to: {stats_path}")

    return saved_paths


def run_pipeline(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    cache_dir: str | None = "./data/raw",
    output_dir: str = "./data/processed",
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    test_fraction: float = 0.10,
    val_fraction: float = 0.05,
    seed: int = RANDOM_SEED,
    hf_token: str | None = None,
    skip_token_filter: bool = False,
) -> DatasetDict:
    """
    Run the full data preprocessing pipeline end-to-end.

    Args:
        model_name: HuggingFace model for tokenizer.
        cache_dir: Local cache for raw HuggingFace data.
        output_dir: Where to save processed splits.
        max_seq_length: Token length cutoff for filtering.
        test_fraction: Fraction for held-out test set.
        val_fraction: Fraction for validation set.
        seed: Random seed.
        hf_token: Optional HuggingFace API token.
        skip_token_filter: If True, skip the slow token-length filter
                           (useful for quick testing).

    Returns:
        Processed DatasetDict.
    """
    set_seed(seed)
    logger.info("Starting data preprocessing pipeline...")

    # 1. Load
    dataset = load_raw_dataset(DATASET_NAME, cache_dir=cache_dir, hf_token=hf_token)

    # 2. Filter empty
    dataset = filter_empty_fields(dataset)

    # 3. Filter by output length
    dataset = filter_by_output_length(dataset)

    # 4. Deduplicate
    dataset, _ = deduplicate(dataset)

    # 5. Filter by token length (slow but important)
    if not skip_token_filter:
        dataset = filter_by_token_length(dataset, model_name, max_seq_length)

    # 6. Create splits
    splits = create_splits(dataset, test_fraction, val_fraction, seed)

    # 7. Save to disk
    save_splits(splits, output_dir)

    logger.info("Data preprocessing pipeline complete.")
    return splits
