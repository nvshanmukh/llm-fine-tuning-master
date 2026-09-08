#!/usr/bin/env python3
"""
Training entrypoint script for LoRA fine-tuning.

Usage:
    # LoRA fine-tuning
    python scripts/train.py --config configs/lora.yaml

    # CPU / small-machine plumbing run (tiny model, few steps)
    python scripts/train.py --config configs/smoke.yaml --max-train-samples 200

    # Override specific config values
    python scripts/train.py --config configs/lora.yaml --learning-rate 1e-4 --epochs 2

    # Run ablation sweep
    python scripts/train.py --config configs/lora.yaml --ablation --ablation-config configs/ablation.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer
from loguru import logger
from omegaconf import OmegaConf

from src.utils.config_utils import load_config, load_env_overrides, merge_configs
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed

app = typer.Typer(help="Train LoRA on the finance-alpaca dataset.")


@app.command()
def main(
    config: str = typer.Option("configs/lora.yaml", help="Path to training config YAML"),
    data_dir: str = typer.Option("./data/processed", help="Path to processed data directory"),
    experiment_name: str = typer.Option(None, help="Override MLflow experiment run name"),
    model_name: str = typer.Option(None, help="Override base model (e.g. Qwen/Qwen2.5-0.5B for a smoke run)"),
    max_train_samples: int = typer.Option(None, help="Cap train/val examples (smoke runs)"),
    learning_rate: float = typer.Option(None, help="Override learning rate from config"),
    epochs: int = typer.Option(None, help="Override number of training epochs from config"),
    lora_rank: int = typer.Option(None, help="Override LoRA rank from config"),
    batch_size: int = typer.Option(None, help="Override per-device batch size from config"),
    max_steps: int = typer.Option(-1, help="Maximum training steps (-1 = full training)"),
    ablation: bool = typer.Option(False, "--ablation", help="Run ablation sweep instead of single run"),
    ablation_config: str = typer.Option("configs/ablation.yaml", help="Ablation config YAML"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Fine-tune Qwen2.5-1.5B with LoRA on finance-alpaca."""
    setup_logger(level=log_level)
    load_env_overrides()

    # Load base config
    cfg = load_config(config)

    # Apply CLI overrides
    overrides: dict[str, object] = {}
    if learning_rate is not None:
        overrides["training.learning_rate"] = learning_rate
    if epochs is not None:
        overrides["training.num_train_epochs"] = epochs
    if lora_rank is not None:
        overrides["peft.r"] = lora_rank
    if batch_size is not None:
        overrides["training.per_device_train_batch_size"] = batch_size
    if model_name is not None:
        overrides["model.model_name_or_path"] = model_name
    if max_steps > 0:
        overrides["training.max_steps"] = max_steps
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = merge_configs(cfg, override_cfg)
        logger.info(f"Applied CLI overrides: {overrides}")

    set_seed(cfg.project.get("seed", 42))

    if ablation:
        _run_ablation(cfg, ablation_config, data_dir, max_train_samples)
    else:
        _run_single(cfg, data_dir, experiment_name, max_train_samples)


def _load_data(data_dir: str, max_samples: int | None = None):
    """Load train and validation splits from processed data directory."""
    from datasets import load_dataset
    data_path = Path(data_dir)
    train_path = data_path / "train.json"
    val_path = data_path / "validation.json"

    if not train_path.exists() or not val_path.exists():
        logger.error(
            f"Processed data not found in {data_dir}. "
            f"Run `python scripts/prepare_data.py` first."
        )
        raise FileNotFoundError(f"Missing train.json or validation.json in {data_dir}")

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train")
    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(max(4, max_samples // 4), len(val_ds))))
        logger.warning(f"SMOKE RUN: capped to train={len(train_ds)}, val={len(val_ds)}")
    logger.info(f"Loaded: train={len(train_ds):,}, val={len(val_ds):,}")
    return train_ds, val_ds


def _run_single(cfg, data_dir: str, experiment_name=None, max_train_samples=None):
    """Run a single training experiment."""
    from src.training.trainer import run_training

    train_ds, val_ds = _load_data(data_dir, max_train_samples)

    logger.info("Starting training run...")
    results = run_training(
        cfg=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        experiment_name=experiment_name,
    )

    logger.info("Training complete!")
    logger.info(f"  MLflow run ID: {results['run_id']}")
    logger.info(f"  Model saved:   {results['model_path']}")
    logger.info(f"  Training time: {results['training_time_s'] / 60:.1f} min")
    logger.info(f"  Trainable params: {results['param_counts']['trainable_params'] / 1e6:.2f}M")

    # Save results summary
    results_path = Path(results["model_path"]).parent / "train_results.json"
    with results_path.open("w") as f:
        json.dump(
            {k: v for k, v in results.items() if isinstance(v, (str, int, float, dict))},
            f, indent=2,
        )
    logger.info(f"Results saved to: {results_path}")


def _run_ablation(cfg, ablation_config_path: str, data_dir: str, max_train_samples=None):
    """Run a systematic ablation study across the configured grid."""
    from itertools import product

    from omegaconf import OmegaConf

    from src.training.trainer import run_training

    ablation_cfg = load_config(ablation_config_path)
    grid = ablation_cfg.get("ablation_grid", {})

    ranks = list(grid.get("lora_rank", [cfg.peft.r]))
    lrs = list(grid.get("learning_rate", [cfg.training.learning_rate]))
    epoch_list = list(grid.get("num_train_epochs", [cfg.training.num_train_epochs]))

    experiments = list(product(ranks, lrs, epoch_list))
    logger.info(f"Running ablation: {len(experiments)} experiments")

    train_ds, val_ds = _load_data(data_dir, max_train_samples)
    all_results = []

    for i, (rank, lr, epochs) in enumerate(experiments, 1):
        exp_name = f"ablation-r{rank}-lr{lr:.0e}-e{epochs}"
        logger.info(f"[{i}/{len(experiments)}] {exp_name}")

        run_cfg = merge_configs(cfg, OmegaConf.create({
            "peft": {"r": rank},
            "training": {"learning_rate": lr, "num_train_epochs": epochs},
        }))

        try:
            results = run_training(
                cfg=run_cfg,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                experiment_name=exp_name,
            )
            all_results.append({"experiment": exp_name, "rank": rank, "lr": lr, "epochs": epochs, **results})
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            all_results.append({"experiment": exp_name, "error": str(e)})

    # Save ablation summary
    ablation_results_path = Path("experiments") / "ablation_results.json"
    ablation_results_path.parent.mkdir(exist_ok=True)
    with ablation_results_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Ablation results saved to: {ablation_results_path}")


if __name__ == "__main__":
    app()
