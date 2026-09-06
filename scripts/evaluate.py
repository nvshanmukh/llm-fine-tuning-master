#!/usr/bin/env python3
"""
Evaluation entrypoint script.

Evaluates one or more model variants on the held-out TEST set and
produces a structured JSON report with all metrics.

Usage:
    # Evaluate base model (zero-shot)
    python scripts/evaluate.py --model-type base

    # Evaluate fine-tuned LoRA model
    python scripts/evaluate.py --model-type lora \\
        --adapter-path ./experiments/lora/final_model

    # Evaluate QLoRA model
    python scripts/evaluate.py --model-type qlora \\
        --adapter-path ./experiments/qlora/final_model --4bit

    # Run full comparison (base + all discovered fine-tuned models)
    python scripts/evaluate.py --compare-all
"""
from __future__ import annotations

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


@app.command()
def main(
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-1.5B",
        help="Base model path or HuggingFace model ID",
    ),
    adapter_path: str = typer.Option(
        None,
        help="LoRA/QLoRA adapter path (leave empty for base model evaluation)",
    ),
    model_type: str = typer.Option(
        "base",
        help="Model identifier used in reports: base, lora, or qlora",
    ),
    data_dir: str = typer.Option(
        "./data/processed",
        help="Directory containing test.json",
    ),
    output_dir: str = typer.Option(
        "./experiments/eval_results",
        help="Where to save evaluation reports",
    ),
    num_samples: int = typer.Option(
        200,
        help="Number of test samples to evaluate (use -1 for full test set)",
    ),
    max_new_tokens: int = typer.Option(256, help="Max tokens to generate per sample"),
    temperature: float = typer.Option(0.1, help="Generation temperature"),
    skip_bertscore: bool = typer.Option(
        False,
        "--skip-bertscore",
        help="Skip BERTScore computation (slow; requires downloading BERT model)",
    ),
    load_in_4bit: bool = typer.Option(
        False,
        "--4bit",
        help="Load model in 4-bit quantization (for QLoRA inference)",
    ),
    compare_all: bool = typer.Option(
        False,
        "--compare-all",
        help="Run evaluation for all available model variants and print comparison table",
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Evaluate the Finance LLM on the held-out test set."""
    setup_logger(level=log_level)
    load_env_overrides()

    if compare_all:
        _run_comparison(
            data_dir=data_dir,
            output_dir=output_dir,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            skip_bertscore=skip_bertscore,
        )
    else:
        _run_single_eval(
            model_path=model_path,
            adapter_path=adapter_path,
            model_type=model_type,
            data_dir=data_dir,
            output_dir=output_dir,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            skip_bertscore=skip_bertscore,
            load_in_4bit=load_in_4bit,
        )


def _load_test_set(data_dir: str, num_samples: int):
    """Load the held-out test split from disk."""
    from datasets import load_dataset

    test_path = Path(data_dir) / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. "
            "Run `python scripts/prepare_data.py` first."
        )
    ds = load_dataset("json", data_files=str(test_path), split="train")
    total = len(ds)
    if num_samples > 0 and num_samples < total:
        ds = ds.select(range(num_samples))
        logger.info(f"Using {num_samples} test samples (full set: {total:,})")
    else:
        logger.info(f"Using full test set: {total:,} samples")
    return ds


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
) -> dict:
    """Run evaluation for a single model variant and return the results dict."""
    import mlflow

    from src.data.prompt_template import format_for_inference
    from src.evaluation.error_analysis import analyze_batch, generate_error_report
    from src.evaluation.metrics import compute_all_metrics
    from src.inference.predict import FinanceLLMPredictor, GenerationConfig

    logger.info(
        f"Evaluating: {model_type} | model={model_path} | "
        f"adapter={adapter_path} | 4bit={load_in_4bit}"
    )

    # 1. Load test data
    test_ds = _load_test_set(data_dir, num_samples)

    # 2. Load model
    predictor = FinanceLLMPredictor(
        model_path=model_path,
        adapter_path=adapter_path or None,
        load_in_4bit=load_in_4bit,
        model_id=model_type,
    )

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
    )

    # 3. Generate predictions
    logger.info("Generating predictions on test set...")
    examples = list(test_ds)
    prompts = [format_for_inference(ex) for ex in examples]
    references = [ex["output"] for ex in examples]

    t_start = time.time()
    inference_results = predictor.generate_batch(prompts, gen_config)
    elapsed = time.time() - t_start

    predictions = [r.response for r in inference_results]
    latencies = [r.latency_ms for r in inference_results]

    # 4. Compute automatic metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_all_metrics(
        predictions, references, skip_bertscore=skip_bertscore
    )
    metrics["mean_latency_ms"] = round(sum(latencies) / len(latencies), 2)
    metrics["total_inference_time_s"] = round(elapsed, 2)
    metrics["num_samples"] = len(examples)
    metrics["model_type"] = model_type

    # 5. Error analysis
    logger.info("Running error analysis...")
    analyzed = analyze_batch(
        examples=examples,
        base_outputs=predictions,       # same list; cross-model comparison in --compare-all
        finetuned_outputs=predictions,
    )
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    error_report = generate_error_report(
        analyzed,
        output_path=output_dir_path / f"{model_type}_error_analysis.json",
    )

    # 6. Save full report
    eval_report = {
        "model_type": model_type,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "num_samples": len(examples),
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "metrics": metrics,
        "error_analysis_summary": error_report["summary"],
        "error_distribution": error_report["error_distribution"],
    }

    report_path = output_dir_path / f"{model_type}_eval_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Evaluation report saved to: {report_path}")

    # 7. Log to MLflow
    try:
        mlflow.set_experiment("finance-llm-evaluation")
        with mlflow.start_run(run_name=f"eval-{model_type}"):
            mlflow.log_params({
                "model_type": model_type,
                "num_samples": len(examples),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            })
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, float)}
            )
            mlflow.log_artifact(str(report_path))
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    # 8. Print summary
    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULTS -- {model_type.upper()}")
    logger.info(f"  ROUGE-1:       {metrics.get('rouge1', 'N/A')}")
    logger.info(f"  ROUGE-2:       {metrics.get('rouge2', 'N/A')}")
    logger.info(f"  ROUGE-L:       {metrics.get('rougeL', 'N/A')}")
    logger.info(f"  BLEU-4:        {metrics.get('bleu4', 'N/A')}")
    logger.info(f"  Exact Match:   {metrics.get('exact_match', 'N/A')}")
    logger.info(f"  Mean latency:  {metrics.get('mean_latency_ms', 'N/A')} ms")
    logger.info("=" * 60)

    return eval_report


def _run_comparison(
    data_dir: str,
    output_dir: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    skip_bertscore: bool,
) -> None:
    """
    Run evaluation across base, LoRA, and QLoRA models and print a comparison table.
    Automatically discovers fine-tuned adapters in ./experiments/.
    """
    model_configs: list[typing.Any] = [
        {
            "model_type": "base",
            "model_path": "Qwen/Qwen2.5-1.5B",
            "adapter_path": None,
            "load_in_4bit": False,
        },
    ]

    # Auto-discover fine-tuned adapters
    for exp_type, use_4bit in [("lora", False), ("qlora", True)]:
        adapter_path = Path(f"./experiments/{exp_type}/final_model")
        if adapter_path.exists():
            model_configs.append({
                "model_type": exp_type,
                "model_path": "Qwen/Qwen2.5-1.5B",
                "adapter_path": str(adapter_path),
                "load_in_4bit": use_4bit,
            })
            logger.info(f"Found {exp_type} adapter: {adapter_path}")
        else:
            logger.warning(
                f"No {exp_type} adapter at {adapter_path} -- skipping. "
                f"Run `python scripts/train.py --config configs/{exp_type}.yaml` first."
            )

    all_reports = []
    for mc in model_configs:
        report = _run_single_eval(
            model_path=mc["model_path"],
            adapter_path=mc.get("adapter_path"),
            model_type=mc["model_type"],
            data_dir=data_dir,
            output_dir=output_dir,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            skip_bertscore=skip_bertscore,
            load_in_4bit=mc.get("load_in_4bit", False),
        )
        all_reports.append(report)

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE")
    logger.info(
        f"{'Model':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} "
        f"{'BLEU-4':>8} {'EM':>6} {'Latency(ms)':>12}"
    )
    logger.info("-" * 70)
    for r in all_reports:
        m = r["metrics"]
        logger.info(
            f"{r['model_type']:<12} "
            f"{m.get('rouge1', 0):>8.4f} "
            f"{m.get('rouge2', 0):>8.4f} "
            f"{m.get('rougeL', 0):>8.4f} "
            f"{m.get('bleu4', 0):>8.4f} "
            f"{m.get('exact_match', 0):>6.3f} "
            f"{m.get('mean_latency_ms', 0):>12.1f}"
        )
    logger.info("=" * 80)

    # Save comparison JSON
    comp_path = Path(output_dir) / "comparison_table.json"
    with comp_path.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    logger.info(f"Full comparison saved to: {comp_path}")


if __name__ == "__main__":
    app()
