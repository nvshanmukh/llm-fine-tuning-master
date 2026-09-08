#!/usr/bin/env python3
"""
Data preparation entrypoint script.

Usage:
    python scripts/prepare_data.py [OPTIONS]

Options:
    --model-name    HuggingFace model name for tokenizer (default: Qwen/Qwen2.5-1.5B)
    --cache-dir     Directory to cache raw HuggingFace data (default: ./data/raw)
    --output-dir    Directory to save processed splits (default: ./data/processed)
    --max-seq-len   Maximum token length for filtering (default: 512)
    --test-frac     Test split fraction (default: 0.10)
    --val-frac      Validation split fraction (default: 0.05)
    --seed          Random seed (default: 42)
    --skip-token-filter  Skip slow token-length filtering (for quick testing)
    --hf-token      HuggingFace API token (or set HF_TOKEN env var)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the Python path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer
from loguru import logger

from src.data.make_dataset import run_pipeline
from src.utils.config_utils import load_env_overrides
from src.utils.logging_utils import setup_logger

app = typer.Typer(help="Finance LLM data preparation pipeline.")


@app.command()
def main(
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B", help="HuggingFace model name for tokenizer"),
    cache_dir: str = typer.Option("./data/raw", help="Raw data cache directory"),
    output_dir: str = typer.Option("./data/processed", help="Processed data output directory"),
    max_seq_len: int = typer.Option(512, help="Maximum sequence length for token-length filter"),
    test_frac: float = typer.Option(0.10, help="Test split fraction"),
    val_frac: float = typer.Option(0.05, help="Validation split fraction"),
    seed: int = typer.Option(42, help="Random seed"),
    train_fraction: float = typer.Option(1.0, help="Fraction of filtered data to keep (1.0 = all)"),
    skip_token_filter: bool = typer.Option(False, "--skip-token-filter", help="Skip slow token length filtering"),
    hf_token: str = typer.Option(None, help="HuggingFace API token (or use HF_TOKEN env var)"),
    log_level: str = typer.Option("INFO", help="Log level (DEBUG/INFO/WARNING/ERROR)"),
) -> None:
    """Prepare the finance-alpaca dataset for LLM fine-tuning."""
    setup_logger(level=log_level)
    load_env_overrides()

    import os
    token = hf_token or os.environ.get("HF_TOKEN")

    logger.info("=" * 60)
    logger.info("Finance LLM Dataset Preparation")
    logger.info(f"  Model:          {model_name}")
    logger.info(f"  Max seq len:    {max_seq_len}")
    logger.info(f"  Test fraction:  {test_frac:.0%}")
    logger.info(f"  Val fraction:   {val_frac:.0%}")
    logger.info(f"  Seed:           {seed}")
    logger.info(f"  Output dir:     {output_dir}")
    logger.info("=" * 60)

    splits = run_pipeline(
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        max_seq_length=max_seq_len,
        test_fraction=test_frac,
        val_fraction=val_frac,
        seed=seed,
        hf_token=token,
        skip_token_filter=skip_token_filter,
        train_fraction=train_fraction,
    )

    logger.info("Data preparation complete!")
    logger.info(f"  Train:      {len(splits['train']):,} examples")
    logger.info(f"  Validation: {len(splits['validation']):,} examples")
    logger.info(f"  Test:       {len(splits['test']):,} examples")
    logger.info(f"  Saved to:   {output_dir}")


if __name__ == "__main__":
    app()
