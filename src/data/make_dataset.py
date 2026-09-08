"""
Dataset preparation pipeline for the Finance LLM fine-tuning project.

This module:
1. Downloads the gbharti/finance-alpaca dataset from HuggingFace.
2. Performs data-quality filtering (empty fields, output length, token length).
3. Detects exact and normalized (near-) duplicates and reports them.
4. Creates reproducible train/validation/test splits (TEST carved out FIRST).
5. Verifies there is no cross-split leakage.
6. Saves processed splits plus machine-readable artifacts:
       train.json / validation.json / test.json  (JSON lines)
       stats.json            -- dataset statistics card (token stats, distributions)
       splits_manifest.json  -- per-split example hashes, seed, fractions, fingerprint
       leakage_report.json   -- exact/normalized duplicate counts, cross-split overlap

Design principles:
- The test set is NEVER exposed to training / validation / tuning decisions.
- All splits are created with a fixed seed and are byte-for-byte reproducible.
- Deduplication uses an (instruction + input) hash; a second normalized hash
  catches punctuation/whitespace-only near-duplicates.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger

from src.data.prompt_template import format_for_training
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "gbharti/finance-alpaca"
# Pinned HF Hub commits for reproducibility (kept in sync with configs/base.yaml).
DEFAULT_DATASET_REVISION = "c88d3d5e7e2c7cab9f11a56f27bd5ba3ed68f075"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEFAULT_MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MIN_OUTPUT_WORDS = 5
DEFAULT_MAX_OUTPUT_WORDS = 600
RANDOM_SEED = 42

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _hash_example(instruction: str, inp: str) -> str:
    """Deterministic hash of (instruction, input) for deduplication."""
    raw = f"{instruction.strip().lower()}|||{(inp or '').strip().lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _normalized_key(instruction: str, inp: str, output: str) -> str:
    """Punctuation/whitespace-insensitive key over the full example (near-dup detection)."""
    joined = " ".join(
        _WS_RE.sub(" ", _PUNCT_RE.sub(" ", (s or "").lower())).strip()
        for s in (instruction, inp, output)
    )
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def load_raw_dataset(
    dataset_name: str = DATASET_NAME,
    cache_dir: str | None = None,
    hf_token: str | None = None,
    revision: str | None = DEFAULT_DATASET_REVISION,
) -> Dataset:
    """Load the raw finance-alpaca dataset (single 'train' split) from the Hub."""
    logger.info(f"Loading dataset: {dataset_name} @ {revision or 'main'}")
    raw = load_dataset(dataset_name, cache_dir=cache_dir, token=hf_token, revision=revision)
    dataset = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
    # Ensure the columns we rely on always exist.
    for col in ("instruction", "input", "output"):
        if col not in dataset.column_names:
            dataset = dataset.add_column(col, [""] * len(dataset))
    logger.info(f"Raw dataset loaded: {len(dataset):,} examples | columns={dataset.column_names}")
    return dataset


def filter_empty_fields(dataset: Dataset) -> Dataset:
    """Remove examples with an empty instruction or output."""
    before = len(dataset)
    dataset = dataset.filter(
        lambda x: bool((x.get("instruction") or "").strip())
        and bool((x.get("output") or "").strip())
    )
    logger.info(f"Empty-field filter: removed {before - len(dataset):,} | remaining {len(dataset):,}")
    return dataset


def filter_by_output_length(
    dataset: Dataset,
    min_words: int = DEFAULT_MIN_OUTPUT_WORDS,
    max_words: int = DEFAULT_MAX_OUTPUT_WORDS,
) -> Dataset:
    """Drop outputs that are too short (garbage) or too long (context-dominating)."""
    before = len(dataset)
    dataset = dataset.filter(
        lambda x: min_words <= len((x.get("output") or "").split()) <= max_words
    )
    logger.info(
        f"Output-length filter (words {min_words}-{max_words}): "
        f"removed {before - len(dataset):,} | remaining {len(dataset):,}"
    )
    return dataset


def _load_tokenizer(tokenizer_name: str, revision: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_name, revision=revision, trust_remote_code=True
    )


def compute_token_lengths(
    dataset: Dataset, tokenizer_name: str, revision: str | None = None
) -> Dataset:
    """Add a ``token_length`` column with the tokenized length of the full training text."""
    tokenizer = _load_tokenizer(tokenizer_name, revision)

    def _fn(example: dict) -> dict:
        return {"token_length": len(tokenizer(format_for_training(example))["input_ids"])}

    logger.info(f"Computing token lengths with tokenizer: {tokenizer_name}")
    return dataset.map(_fn, desc="Computing token lengths")


def filter_by_token_length(
    dataset: Dataset,
    tokenizer_name: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    tokenizer_revision: str | None = None,
) -> tuple[Dataset, dict]:
    """
    Remove examples whose full formatted prompt+response exceeds ``max_seq_length``.

    Returns (filtered_dataset, token_stats_dict). The stats are computed on the
    pre-filter distribution and include the truncation rate.
    """
    if "token_length" not in dataset.column_names:
        dataset = compute_token_lengths(dataset, tokenizer_name, tokenizer_revision)

    lengths = np.asarray(dataset["token_length"], dtype=int)
    before = len(dataset)
    over = int((lengths > max_seq_length).sum())
    stats = {
        "tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "max_seq_length": max_seq_length,
        "count_pre_filter": before,
        "mean": round(float(lengths.mean()), 1),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "truncation_rate_at_max_seq_length": round(over / before, 4) if before else 0.0,
        "removed_by_token_filter": over,
    }
    logger.info(
        f"Token-length stats: mean={stats['mean']} median={stats['median']} "
        f"p95={stats['p95']} max={stats['max']} | over {max_seq_length}: {over:,} "
        f"({stats['truncation_rate_at_max_seq_length']:.1%})"
    )
    dataset = dataset.filter(lambda x: x["token_length"] <= max_seq_length)
    logger.info(f"Token-length filter (<= {max_seq_length}): remaining {len(dataset):,}")
    return dataset, stats


def deduplicate(dataset: Dataset) -> tuple[Dataset, int]:
    """Remove exact (instruction, input) duplicates, keeping the first occurrence."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, ex in enumerate(dataset):
        h = _hash_example(ex.get("instruction", ""), ex.get("input", ""))
        if h not in seen:
            seen.add(h)
            keep.append(i)
    before = len(dataset)
    dataset = dataset.select(keep)
    removed = before - len(dataset)
    logger.info(f"Deduplication (exact): removed {removed:,} | remaining {len(dataset):,}")
    return dataset, removed


def count_normalized_near_duplicates(dataset: Dataset) -> int:
    """Count examples that collide on the normalized (punctuation-insensitive) key."""
    seen: set[str] = set()
    collisions = 0
    for ex in dataset:
        k = _normalized_key(ex.get("instruction", ""), ex.get("input", ""), ex.get("output", ""))
        if k in seen:
            collisions += 1
        else:
            seen.add(k)
    return collisions


def create_splits(
    dataset: Dataset,
    test_fraction: float = 0.10,
    val_fraction: float = 0.05,
    seed: int = RANDOM_SEED,
) -> DatasetDict:
    """
    Create reproducible train/validation/test splits.

    The test split is carved out FIRST (before train/val), guaranteeing zero
    leakage between test and training data. ``val_fraction`` is relative to the
    full dataset.
    """
    logger.info(f"Creating splits: test={test_fraction:.0%}, val={val_fraction:.0%}, seed={seed}")

    train_test = dataset.train_test_split(test_size=test_fraction, seed=seed, shuffle=True)
    test_ds, remaining = train_test["test"], train_test["train"]

    effective_val = val_fraction / (1.0 - test_fraction)
    train_val = remaining.train_test_split(test_size=effective_val, seed=seed, shuffle=True)
    train_ds, val_ds = train_val["train"], train_val["test"]

    _verify_no_overlap(train_ds, val_ds, test_ds)
    logger.info(
        f"Splits: train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}"
    )
    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


def _split_hashes(ds: Dataset) -> set[str]:
    return {_hash_example(ex.get("instruction", ""), ex.get("input", "")) for ex in ds}


def _verify_no_overlap(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> None:
    """Raise AssertionError if any two splits share an (instruction, input) hash."""
    train_h, val_h, test_h = _split_hashes(train_ds), _split_hashes(val_ds), _split_hashes(test_ds)
    assert not (train_h & test_h), f"DATA LEAKAGE: {len(train_h & test_h)} train/test overlap"
    assert not (train_h & val_h), f"DATA LEAKAGE: {len(train_h & val_h)} train/val overlap"
    assert not (val_h & test_h), f"DATA LEAKAGE: {len(val_h & test_h)} val/test overlap"
    logger.info("Leakage check PASSED - no overlap between splits")


def _example_stats(ds: Dataset) -> dict:
    out_words = np.asarray([len((r or "").split()) for r in ds["output"]], dtype=int)
    has_input = np.asarray([bool((r or "").strip()) for r in ds["input"]], dtype=bool)
    d = {
        "count": len(ds),
        "output_words_mean": round(float(out_words.mean()), 1) if len(ds) else 0.0,
        "output_words_median": float(np.median(out_words)) if len(ds) else 0.0,
        "output_words_p95": float(np.percentile(out_words, 95)) if len(ds) else 0.0,
        "has_input_fraction": round(float(has_input.mean()), 4) if len(ds) else 0.0,
    }
    if "token_length" in ds.column_names and len(ds):
        tl = np.asarray(ds["token_length"], dtype=int)
        d["token_length_mean"] = round(float(tl.mean()), 1)
        d["token_length_p95"] = float(np.percentile(tl, 95))
        d["token_length_max"] = int(tl.max())
    return d


def save_splits(
    splits: DatasetDict,
    output_dir: str | Path,
    *,
    extra_stats: dict | None = None,
    seed: int = RANDOM_SEED,
    fractions: dict | None = None,
) -> dict[str, Path]:
    """Save splits to JSON and write stats.json / splits_manifest.json / leakage_report.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for name, ds in splits.items():
        p = output_dir / f"{name}.json"
        ds.to_json(str(p))
        saved[name] = p
        logger.info(f"Saved {name}: {len(ds):,} -> {p}")

    def _digest(hashes: set[str]) -> str:
        joined = "".join(sorted(hashes)).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()

    manifest = {
        "seed": seed,
        "fractions": fractions or {},
        "split_sizes": {k: len(v) for k, v in splits.items()},
        # A stable fingerprint of each split's membership. Re-running the
        # pipeline with the same seed must reproduce these digests exactly.
        "split_membership_sha256": {k: _digest(_split_hashes(v)) for k, v in splits.items()},
        "example_hash_algo": "md5(instruction.strip().lower() + '|||' + input.strip().lower())",
        "sample_ids_note": (
            "test-set sample_id used by scripts/evaluate.py is the 0-based row index "
            "in test.json (stable across model variants)"
        ),
    }
    (output_dir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))

    cross = {
        "train_test_overlap": len(_split_hashes(splits["train"]) & _split_hashes(splits["test"])),
        "train_val_overlap": len(_split_hashes(splits["train"]) & _split_hashes(splits["validation"])),
        "val_test_overlap": len(_split_hashes(splits["validation"]) & _split_hashes(splits["test"])),
    }
    leakage = {
        "cross_split_overlap_by_instruction_input_hash": cross,
        "leakage_detected": any(cross.values()),
        **(extra_stats.get("duplicates", {}) if extra_stats else {}),
        "group_aware_splitting": (
            "not applicable - finance-alpaca has no document/group id; each row is "
            "an independent (instruction, input, output) triple"
        ),
    }
    (output_dir / "leakage_report.json").write_text(json.dumps(leakage, indent=2))

    stats = {
        "dataset_source": DATASET_NAME,
        "license": "Apache-2.0 (code) / CC-BY-4.0 (data) - see dataset card",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "splits": {k: _example_stats(v) for k, v in splits.items()},
        "total_examples": sum(len(v) for v in splits.values()),
        **(extra_stats or {}),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Wrote stats.json, splits_manifest.json, leakage_report.json to {output_dir}")
    return saved


def run_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = "./data/raw",
    output_dir: str = "./data/processed",
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    test_fraction: float = 0.10,
    val_fraction: float = 0.05,
    seed: int = RANDOM_SEED,
    hf_token: str | None = None,
    skip_token_filter: bool = False,
    train_fraction: float = 1.0,
    model_revision: str | None = DEFAULT_MODEL_REVISION,
    dataset_revision: str | None = DEFAULT_DATASET_REVISION,
) -> DatasetDict:
    """Run the full data preprocessing pipeline end to end."""
    set_seed(seed)
    logger.info("Starting data preprocessing pipeline...")

    dataset = load_raw_dataset(
        DATASET_NAME, cache_dir=cache_dir, hf_token=hf_token, revision=dataset_revision
    )
    raw_count = len(dataset)

    dataset = filter_empty_fields(dataset)
    dataset = filter_by_output_length(dataset)

    norm_near_dups = count_normalized_near_duplicates(dataset)
    dataset, exact_dups = deduplicate(dataset)

    token_stats: dict = {}
    if not skip_token_filter:
        dataset, token_stats = filter_by_token_length(
            dataset, model_name, max_seq_length, tokenizer_revision=model_revision
        )

    if train_fraction < 1.0:
        keep_n = int(len(dataset) * train_fraction)
        dataset = dataset.shuffle(seed=seed).select(range(keep_n))
        logger.info(f"Subsampled to train_fraction={train_fraction}: {len(dataset):,} examples")

    splits = create_splits(dataset, test_fraction, val_fraction, seed)

    extra_stats = {
        "raw_examples": raw_count,
        "examples_after_filtering": sum(len(v) for v in splits.values()),
        "dataset_revision": dataset_revision,
        "tokenizer": model_name,
        "tokenizer_revision": model_revision,
        "token_length_stats": token_stats,
        "duplicates": {
            "exact_duplicates_removed": exact_dups,
            "normalized_near_duplicates_detected": norm_near_dups,
            "normalized_near_duplicate_note": (
                "normalized key ignores case/punctuation/whitespace over "
                "instruction+input+output; these are counted but NOT removed by default"
            ),
        },
    }
    save_splits(
        splits, output_dir, extra_stats=extra_stats, seed=seed,
        fractions={"test": test_fraction, "val": val_fraction, "train_fraction": train_fraction},
    )
    logger.info("Data preprocessing pipeline complete.")
    return splits
