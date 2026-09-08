#!/usr/bin/env python3
"""
Post-training inference-quantization comparison.

Runs the SAME held-out test examples through the selected model in two (or more)
precisions and reports, for each: evaluation quality (ROUGE/BLEU/EM), serialized
footprint, and latency (median / p95). Uses greedy decoding so quality
differences are attributable to quantization, not sampling noise.

Modes:
  fp             full precision (torch_dtype auto)
  dynamic_int8   torch CPU dynamic int8 on nn.Linear (no CUDA needed)
  int4 / int8    bitsandbytes 4-/8-bit  (CUDA + bitsandbytes only)

Usage:
  python scripts/quantize_compare.py --model-path Qwen/Qwen2.5-1.5B \
      --adapter-path experiments/lora/final_model --modes fp,dynamic_int8 --num-samples 100
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

from src.utils.logging_utils import setup_logger

app = typer.Typer(help="Compare inference quantization modes on the test set.")


def _predictor_for_mode(mode: str, model_path: str, adapter_path: str | None):
    from src.inference.predict import FinanceLLMPredictor

    kw: dict = dict(model_path=model_path, adapter_path=adapter_path or None,
                    model_id=f"{mode}", device_map=None)
    if mode == "fp":
        kw["torch_dtype_str"] = "float32"  # clean fp32 -> int8 baseline
    elif mode == "dynamic_int8":
        kw["dynamic_int8"] = True
    elif mode == "int8":
        kw["load_in_8bit"] = True
    elif mode == "int4":
        kw["load_in_4bit"] = True
    else:
        raise typer.BadParameter(f"unknown mode {mode!r}")
    return FinanceLLMPredictor(**kw)


@app.command()
def main(
    model_path: str = typer.Option("Qwen/Qwen2.5-1.5B"),
    adapter_path: str = typer.Option(None),
    data_dir: str = typer.Option("./data/processed"),
    output: str = typer.Option("./experiments/eval_results/quantization_comparison.json"),
    modes: str = typer.Option("fp,dynamic_int8", help="Comma-separated list"),
    num_samples: int = typer.Option(100),
    max_new_tokens: int = typer.Option(128),
    seed: int = typer.Option(42),
) -> None:
    setup_logger()
    from scripts.evaluate import _load_test_set
    from src.data.prompt_template import format_for_inference
    from src.evaluation.metrics import bootstrap_mean_ci, compute_all_metrics, rouge_l_per_example
    from src.inference.predict import GenerationConfig
    from src.utils.seed import set_seed

    set_seed(seed)
    examples, provenance = _load_test_set(data_dir, num_samples)
    prompts = [format_for_inference(e) for e in examples]
    references = [e.get("output", "") for e in examples]
    gen = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)

    results: dict[str, dict] = {}
    for mode in [m.strip() for m in modes.split(",") if m.strip()]:
        logger.info(f"=== mode: {mode} ===")
        try:
            predictor = _predictor_for_mode(mode, model_path, adapter_path)
        except Exception as e:
            results[mode] = {"status": "unavailable", "error": f"{type(e).__name__}: {e}"}
            logger.warning(f"mode {mode} unavailable: {e}")
            continue

        predictor.warmup(gen)
        infer = predictor.generate_batch(prompts, gen, show_progress=True)
        preds = [r.response for r in infer]
        lat = sorted(r.latency_ms for r in infer)
        import numpy as np

        metrics = compute_all_metrics(preds, references, skip_bertscore=True)
        metrics["rougeL_ci95"] = bootstrap_mean_ci(rouge_l_per_example(preds, references))
        results[mode] = {
            "status": "ok",
            "model_size_mb": predictor.model_size_mb,
            "latency_ms_median": round(float(np.median(lat)), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "metrics": metrics,
        }
        del predictor

    # Deltas vs the first successful mode
    ok_modes = [m for m, v in results.items() if v.get("status") == "ok"]
    report = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "seed": seed,
        "num_samples": len(examples),
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": False},
        "dataset_provenance": provenance,
        "modes": results,
        "note": (
            "greedy decoding; latency is CPU/GPU wall-clock per example, batch 1, "
            "warm-up excluded; model_size_mb is the serialized state_dict size"
        ),
    }
    if len(ok_modes) >= 2:
        ref_mode = ok_modes[0]
        ref = results[ref_mode]
        report["deltas_vs_" + ref_mode] = {
            m: {
                "rougeL_delta": round(results[m]["metrics"]["rougeL"] - ref["metrics"]["rougeL"], 4),
                "size_ratio": round(results[m]["model_size_mb"] / ref["model_size_mb"], 3)
                if ref["model_size_mb"] else None,
                "latency_ratio": round(
                    results[m]["latency_ms_median"] / ref["latency_ms_median"], 3
                ) if ref["latency_ms_median"] else None,
            }
            for m in ok_modes[1:]
        }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {out}")
    logger.info(json.dumps(report.get("deltas_vs_" + (ok_modes[0] if ok_modes else ""), {}), indent=2))


if __name__ == "__main__":
    app()
