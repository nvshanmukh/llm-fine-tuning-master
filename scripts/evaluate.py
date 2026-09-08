#!/usr/bin/env python3
"""
Evaluation entrypoint script.

Evaluates one or more model variants on the held-out TEST set and produces
fully traceable artifacts:

    experiments/eval_results/
        <model_type>_predictions.jsonl   -- per-example prompt/reference/prediction/latency
        <model_type>_eval_report.json    -- aggregate metrics (recomputed from predictions),
                                            generation config, dataset fingerprint, seed, CIs
        error_analysis_<ft>_vs_base.json -- base vs fine-tuned comparison from saved predictions
        comparison_table.json / .md      -- side-by-side summary

Aggregate metrics in the report are ALWAYS recomputed from the saved
predictions file, so every number is reproducible with `--from-predictions`.

Usage:
    # Base model (zero-shot) on 200 held-out samples, greedy decoding
    python scripts/evaluate.py --model-type base --num-samples 200

    # Fine-tuned LoRA
    python scripts/evaluate.py --model-type lora --adapter-path ./experiments/lora/final_model

    # Full comparison: base + the LoRA adapter under ./experiments/
    python scripts/evaluate.py --compare-all --num-samples 200

    # Recompute metrics/report from an existing predictions file (no model needed)
    python scripts/evaluate.py --from-predictions experiments/eval_results/base_predictions.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer
from loguru import logger

from src.utils.config_utils import load_env_overrides
from src.utils.logging_utils import setup_logger

app = typer.Typer(help="Evaluate Finance LLM models on the held-out test set.")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B"


# ---------------------------------------------------------------------------
# Dataset loading + fingerprinting
# ---------------------------------------------------------------------------
def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _load_test_set(data_dir: str, num_samples: int) -> tuple[list[dict], dict]:
    """Load the held-out test split and return (examples, provenance)."""
    from datasets import load_dataset

    test_path = Path(data_dir) / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. Run `python scripts/prepare_data.py` first."
        )
    ds = load_dataset("json", data_files=str(test_path), split="train")
    total = len(ds)
    if 0 < num_samples < total:
        ds = ds.select(range(num_samples))
        logger.info(f"Using first {num_samples} test samples (full set: {total:,})")
    else:
        logger.info(f"Using full test set: {total:,} samples")

    examples = []
    for i, ex in enumerate(ds):
        ex = dict(ex)
        ex["sample_id"] = i  # stable positional id, identical across all models
        examples.append(ex)

    provenance = {
        "test_file": str(test_path),
        "test_file_sha256_16": _file_fingerprint(test_path),
        "test_total_examples": total,
        "eval_examples": len(examples),
    }
    return examples, provenance


# ---------------------------------------------------------------------------
# Metric computation from a predictions list
# ---------------------------------------------------------------------------
def _aggregate_from_predictions(records: list[dict], skip_bertscore: bool) -> dict:
    from src.evaluation.metrics import (
        bootstrap_mean_ci,
        compute_all_metrics,
        rouge_l_per_example,
    )

    predictions = [r["prediction"] for r in records]
    references = [r["reference"] for r in records]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]

    metrics = compute_all_metrics(predictions, references, skip_bertscore=skip_bertscore)

    rouge_l = rouge_l_per_example(predictions, references)
    metrics["rougeL_ci95"] = bootstrap_mean_ci(rouge_l)

    if latencies:
        import numpy as np

        metrics["latency_ms_median"] = round(float(np.median(latencies)), 2)
        metrics["latency_ms_p95"] = round(float(np.percentile(latencies, 95)), 2)
        metrics["latency_ms_mean"] = round(float(np.mean(latencies)), 2)
        metrics["latency_note"] = (
            "wall-clock per example, batch size 1, includes tokenisation + decode; "
            "first (warm-up) generation excluded"
        )
    metrics["num_samples"] = len(records)
    return metrics


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------
def _run_single_eval(
    model_path: str,
    adapter_path: str | None,
    model_type: str,
    data_dir: str,
    output_dir: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    skip_bertscore: bool,
    load_in_4bit: bool = False,
    seed: int = 42,
    revision: str | None = None,
) -> dict:
    """Generate predictions for one model variant, save artifacts, return the report."""
    import mlflow

    from src.data.make_dataset import DEFAULT_MODEL_REVISION
    from src.data.prompt_template import format_for_inference
    from src.inference.predict import FinanceLLMPredictor, GenerationConfig

    if revision is None and model_path == DEFAULT_BASE_MODEL:
        revision = DEFAULT_MODEL_REVISION

    logger.info(
        f"Evaluating: {model_type} | model={model_path}@{revision or 'main'} "
        f"| adapter={adapter_path} | 4bit={load_in_4bit}"
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples, provenance = _load_test_set(data_dir, num_samples)

    predictor = FinanceLLMPredictor(
        model_path=model_path,
        adapter_path=adapter_path or None,
        load_in_4bit=load_in_4bit,
        model_id=model_type,
        revision=revision,
    )

    do_sample = temperature > 0.0
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        do_sample=do_sample,
    )
    logger.info(f"Generation config: {gen_config}  (deterministic={not do_sample})")

    predictor.warmup(gen_config)

    prompts = [format_for_inference(ex) for ex in examples]
    t_start = time.time()
    inference_results = predictor.generate_batch(prompts, gen_config)
    elapsed = time.time() - t_start

    # --- Save per-example predictions (the source of truth) ---
    pred_path = out_dir / f"{model_type}_predictions.jsonl"
    records = []
    for ex, res in zip(examples, inference_results):
        rec = {
            "sample_id": ex["sample_id"],
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "reference": ex.get("output", ""),
            "prediction": res.response,
            "latency_ms": res.latency_ms,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
        }
        records.append(rec)
    with pred_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} predictions -> {pred_path}")

    # --- Recompute all aggregate metrics from the saved predictions ---
    metrics = _aggregate_from_predictions(records, skip_bertscore=skip_bertscore)
    metrics["total_inference_time_s"] = round(elapsed, 2)
    metrics["model_type"] = model_type

    eval_report = {
        "model_type": model_type,
        "model_path": model_path,
        "model_revision": revision,
        "adapter_path": adapter_path,
        "adapter_merged": getattr(predictor, "adapter_merged", None),
        "load_in_4bit": load_in_4bit,
        "seed": seed,
        "generation_config": {
            "max_new_tokens": gen_config.max_new_tokens,
            "temperature": gen_config.temperature,
            "do_sample": gen_config.do_sample,
            "top_p": gen_config.top_p,
            "repetition_penalty": gen_config.repetition_penalty,
        },
        "dataset_provenance": provenance,
        "predictions_file": str(pred_path),
        "predictions_sha256_16": _file_fingerprint(pred_path),
        "metrics": metrics,
    }
    report_path = out_dir / f"{model_type}_eval_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Evaluation report saved to: {report_path}")

    try:
        from src.utils.config_utils import resolve_mlflow_uri

        mlflow.set_tracking_uri(resolve_mlflow_uri(None))
        mlflow.set_experiment("finance-llm-evaluation")
        with mlflow.start_run(run_name=f"eval-{model_type}"):
            mlflow.log_params({
                "model_type": model_type,
                "adapter_path": adapter_path,
                "load_in_4bit": load_in_4bit,
                "num_samples": len(records),
                "max_new_tokens": max_new_tokens,
                "temperature": gen_config.temperature,
                "do_sample": gen_config.do_sample,
                "test_fingerprint": provenance["test_file_sha256_16"],
            })
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(pred_path))
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULTS -- {model_type.upper()}")
    for key in ("rouge1", "rouge2", "rougeL", "bleu4", "exact_match",
                "bertscore_f1", "latency_ms_median", "latency_ms_p95"):
        if key in metrics:
            logger.info(f"  {key:<18} {metrics[key]}")
    logger.info("=" * 60)
    return eval_report


# ---------------------------------------------------------------------------
# Comparison across variants + error analysis from saved predictions
# ---------------------------------------------------------------------------
def _load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _error_analysis_vs_base(
    base_records: list[dict],
    ft_records: list[dict],
    ft_type: str,
    out_dir: Path,
) -> dict:
    from src.evaluation.error_analysis import analyze_batch, generate_error_report
    from src.evaluation.metrics import paired_bootstrap_diff, rouge_l_per_example

    base_by_id = {r["sample_id"]: r for r in base_records}
    ft_by_id = {r["sample_id"]: r for r in ft_records}
    common = sorted(set(base_by_id) & set(ft_by_id))
    if not common:
        logger.warning(f"No overlapping sample_ids between base and {ft_type}; skipping error analysis")
        return {}

    examples = [
        {
            "instruction": base_by_id[i]["instruction"],
            "input": base_by_id[i].get("input", ""),
            "output": base_by_id[i]["reference"],
        }
        for i in common
    ]
    base_outputs = [base_by_id[i]["prediction"] for i in common]
    ft_outputs = [ft_by_id[i]["prediction"] for i in common]

    analyzed = analyze_batch(examples, base_outputs, ft_outputs)
    report = generate_error_report(
        analyzed, output_path=out_dir / f"error_analysis_{ft_type}_vs_base.json"
    )

    base_rl = rouge_l_per_example(base_outputs, [e["output"] for e in examples])
    ft_rl = rouge_l_per_example(ft_outputs, [e["output"] for e in examples])
    report["paired_rougeL_diff_ft_minus_base"] = paired_bootstrap_diff(ft_rl, base_rl)
    with (out_dir / f"error_analysis_{ft_type}_vs_base.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _write_comparison_table(reports: list[dict], out_dir: Path) -> None:
    keys = ["rouge1", "rouge2", "rougeL", "bleu4", "exact_match", "bertscore_f1",
            "latency_ms_median", "latency_ms_p95"]
    lines = ["| Model | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for r in reports:
        m = r["metrics"]
        row = [r["model_type"]] + [str(m.get(k, "—")) for k in keys]
        lines.append("| " + " | ".join(row) + " |")
    (out_dir / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "comparison_table.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    logger.info("\n" + "\n".join(lines))
    logger.info(f"Comparison table saved to: {out_dir / 'comparison_table.md'}")


def _run_comparison(
    data_dir: str,
    output_dir: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    skip_bertscore: bool,
    seed: int,
) -> None:
    out_dir = Path(output_dir)
    model_configs: list[typing.Any] = [
        {"model_type": "base", "model_path": DEFAULT_BASE_MODEL,
         "adapter_path": None, "load_in_4bit": False},
    ]
    for exp_type in ("lora",):
        adapter_path = Path(f"./experiments/{exp_type}/final_model")
        if adapter_path.exists():
            model_configs.append({
                "model_type": exp_type, "model_path": DEFAULT_BASE_MODEL,
                "adapter_path": str(adapter_path), "load_in_4bit": False,
            })
            logger.info(f"Found {exp_type} adapter: {adapter_path}")
        else:
            logger.warning(
                f"No {exp_type} adapter at {adapter_path} -- skipping. "
                f"Train it with `python scripts/train.py --config configs/{exp_type}.yaml`."
            )

    reports = []
    for mc in model_configs:
        reports.append(_run_single_eval(
            model_path=mc["model_path"], adapter_path=mc.get("adapter_path"),
            model_type=mc["model_type"], data_dir=data_dir, output_dir=output_dir,
            num_samples=num_samples, max_new_tokens=max_new_tokens, temperature=temperature,
            skip_bertscore=skip_bertscore, load_in_4bit=mc.get("load_in_4bit", False), seed=seed,
        ))

    _write_comparison_table(reports, out_dir)

    base_pred = out_dir / "base_predictions.jsonl"
    if base_pred.exists():
        base_records = _load_predictions(base_pred)
        for mc in model_configs:
            if mc["model_type"] == "base":
                continue
            ft_pred = out_dir / f"{mc['model_type']}_predictions.jsonl"
            if ft_pred.exists():
                logger.info(f"Error analysis: {mc['model_type']} vs base")
                _error_analysis_vs_base(base_records, _load_predictions(ft_pred),
                                        mc["model_type"], out_dir)


@app.command()
def main(
    model_path: str = typer.Option(DEFAULT_BASE_MODEL, help="Base model path or HF model ID"),
    revision: str = typer.Option(None, help="Pin the base model's HF commit (default: base.yaml pin)"),
    adapter_path: str = typer.Option(None, help="LoRA/QLoRA adapter path (empty = base model)"),
    model_type: str = typer.Option("base", help="Report identifier: base, lora, or qlora"),
    data_dir: str = typer.Option("./data/processed", help="Directory containing test.json"),
    output_dir: str = typer.Option("./experiments/eval_results", help="Where to save reports"),
    num_samples: int = typer.Option(200, help="Number of test samples (-1 = full test set)"),
    max_new_tokens: int = typer.Option(256, help="Max tokens to generate per sample"),
    temperature: float = typer.Option(
        0.0, help="Generation temperature. 0.0 => deterministic greedy decoding (recommended)."
    ),
    seed: int = typer.Option(42, help="Random seed (recorded in every report)"),
    skip_bertscore: bool = typer.Option(False, "--skip-bertscore", help="Skip BERTScore (slow)"),
    load_in_4bit: bool = typer.Option(False, "--4bit", help="Load base model in 4-bit"),
    compare_all: bool = typer.Option(False, "--compare-all", help="Evaluate base + all adapters"),
    from_predictions: str = typer.Option(
        None, help="Recompute a report from an existing *_predictions.jsonl (no model load)"
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Evaluate the Finance LLM on the held-out test set."""
    setup_logger(level=log_level)
    load_env_overrides()

    from src.utils.seed import set_seed

    set_seed(seed)

    if from_predictions:
        path = Path(from_predictions)
        records = _load_predictions(path)
        metrics = _aggregate_from_predictions(records, skip_bertscore=skip_bertscore)
        report = {
            "recomputed_from": str(path),
            "predictions_sha256_16": _file_fingerprint(path),
            "metrics": metrics,
        }
        out = path.with_name(path.stem.replace("_predictions", "") + "_recomputed_report.json")
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Recomputed report -> {out}")
        logger.info(json.dumps(metrics, indent=2, default=str))
        return

    if compare_all:
        _run_comparison(data_dir, output_dir, num_samples, max_new_tokens,
                        temperature, skip_bertscore, seed)
    else:
        _run_single_eval(
            model_path=model_path, adapter_path=adapter_path, model_type=model_type,
            data_dir=data_dir, output_dir=output_dir, num_samples=num_samples,
            max_new_tokens=max_new_tokens, temperature=temperature,
            skip_bertscore=skip_bertscore, load_in_4bit=load_in_4bit, seed=seed,
            revision=revision,
        )


if __name__ == "__main__":
    app()
