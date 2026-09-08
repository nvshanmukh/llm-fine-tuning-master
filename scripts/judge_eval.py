#!/usr/bin/env python3
"""
LLM-as-a-Judge evaluation (blinded pairwise: fine-tuned vs base).

Reads two prediction files produced by ``scripts/evaluate.py`` and asks an
external judge LLM which answer is better, for each shared test example. Every
pair is judged twice (A/B order swapped) to measure and cancel position bias.

The judge is an OpenAI-compatible chat endpoint configured via environment:

    JUDGE_API_BASE   e.g. https://api.openai.com/v1   (or a local server)
    JUDGE_API_KEY    API key for that endpoint
    JUDGE_MODEL      e.g. gpt-4o-mini                  (default: gpt-4o-mini)

If credentials are missing the script writes a report with status
``blocked_missing_credentials`` and the exact command to re-run, then exits 0.
The candidate models are NEVER used to judge themselves.

Usage:
    python scripts/judge_eval.py \\
        --base experiments/eval_results/base_predictions.jsonl \\
        --finetuned experiments/eval_results/lora_predictions.jsonl \\
        --num-samples 100
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer
from loguru import logger

from src.evaluation.judge import judge_pairwise
from src.utils.logging_utils import setup_logger

app = typer.Typer(help="Blinded pairwise LLM-as-a-judge evaluation.")


def _load(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["sample_id"]: json.loads(line) for line in f if line.strip()}


@app.command()
def main(
    base: str = typer.Option(..., help="Path to base_predictions.jsonl"),
    finetuned: str = typer.Option(..., help="Path to <ft>_predictions.jsonl"),
    output: str = typer.Option("experiments/eval_results/judge_pairwise.json"),
    num_samples: int = typer.Option(100, help="Max shared examples to judge"),
    seed: int = typer.Option(42),
) -> None:
    setup_logger()
    base_p, ft_p = Path(base), Path(finetuned)
    base_rec, ft_rec = _load(base_p), _load(ft_p)
    common = sorted(set(base_rec) & set(ft_rec))
    random.Random(seed).shuffle(common)
    common = common[:num_samples]

    api_base = os.environ.get("JUDGE_API_BASE")
    api_key = os.environ.get("JUDGE_API_KEY")
    model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not api_base or not api_key:
        report = {
            "status": "blocked_missing_credentials",
            "reason": "JUDGE_API_BASE and JUDGE_API_KEY are not set.",
            "judge_model": model,
            "shared_examples_available": len(set(base_rec) & set(ft_rec)),
            "rerun_command": (
                "JUDGE_API_BASE=https://api.openai.com/v1 JUDGE_API_KEY=sk-... "
                f"JUDGE_MODEL={model} python scripts/judge_eval.py "
                f"--base {base} --finetuned {finetuned} --num-samples {num_samples}"
            ),
            "limitations": [
                "verbosity bias: judges tend to prefer longer answers",
                "position bias: mitigated here by judging each pair in both orders",
                "not ground truth: calibrate against human annotation before trusting",
                "the candidate models are not used to judge themselves",
            ],
        }
        out_path.write_text(json.dumps(report, indent=2))
        logger.warning(f"Judge credentials missing -- wrote pending report to {out_path}")
        return

    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    def call_llm(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=400,
        )
        return resp.choices[0].message.content or ""

    ft_type = ft_p.stem.replace("_predictions", "")
    per_example = []
    tally = {"finetuned": 0, "base": 0, "tie": 0}
    for sid in common:
        ex_b, ex_f = base_rec[sid], ft_rec[sid]
        votes = []
        for swap in (False, True):
            # answer_a = fine-tuned, answer_b = base (identity hidden from judge)
            r = judge_pairwise(
                ex_b["instruction"], ex_b.get("input", ""), ex_b["reference"],
                ex_f["prediction"], ex_b["prediction"], call_llm, swap=swap,
            )
            votes.append("finetuned" if r["winner"] == "a" else "base" if r["winner"] == "b" else "tie")
        verdict = votes[0] if votes[0] == votes[1] else "tie"  # disagree across orders => tie
        tally[verdict] += 1
        per_example.append({"sample_id": sid, "votes": votes, "verdict": verdict})

    n = len(common)
    report = {
        "status": "completed",
        "judge_model": model,
        "comparison": f"{ft_type} (fine-tuned) vs base",
        "n_examples": n,
        "win_rate_finetuned": round(tally["finetuned"] / n, 4) if n else 0.0,
        "win_rate_base": round(tally["base"] / n, 4) if n else 0.0,
        "tie_rate": round(tally["tie"] / n, 4) if n else 0.0,
        "tally": tally,
        "position_bias_note": "each pair judged in both A/B orders; disagreement counted as tie",
        "per_example": per_example,
        "limitations": [
            "verbosity bias", "not ground truth without human calibration",
            "candidate models are not used to judge themselves",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Judge report -> {out_path}: {report['win_rate_finetuned']=} {report['win_rate_base']=}")


if __name__ == "__main__":
    app()
