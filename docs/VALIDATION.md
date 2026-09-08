# Validation Performed

Every check below is tagged with a **level**:

- **L1 — code-path test**: synthetic fixture or tiny substitute model. Proves plumbing only.
- **L2 — target-family smoke**: real model/tokenizer at tiny scale (few steps / few samples).
  Proves integration, *not* model quality.
- **L3 — full experiment**: intended model + full split + real config. Only L3 may back
  benchmark or resume claims.

L1/L2 results here must **never** be presented as L3 results.

---

## Executed

| # | What | Level | Command | Result |
|---|---|---|---|---|
| V1 | Unit + integration test suite | L1 | `pytest -q` | **105 passed** — `test_api.py` 24, `test_api_integration.py` 3 (real tiny model), `test_data.py` 20, `test_evaluation.py` 40, `test_inference.py` 21, `test_training.py` 7 |
| V2 | No-model pipeline smoke | L1 | `python scripts/inference.py --smoke-test` | PASS (prompt template, config merge, metrics, error heuristics, dataclasses) |
| V3 | Real inference path | L1 | `python scripts/inference.py --real-smoke-test` | PASS — `sshleifer/tiny-gpt2` loaded via `FinanceLLMPredictor`, 92→16 tokens in ~77 ms |
| V4 | **Full data preparation** on the real dataset + real Qwen2.5-1.5B tokenizer | **L3 (data only)** | `python scripts/prepare_data.py --model-name Qwen/Qwen2.5-1.5B --max-seq-len 512 --seed 42` | Completed. 68,912 raw → 52,180 kept → train 44,353 / val 2,609 / test 5,218. Leakage overlaps all 0. Artifacts in `data/processed/`. |
| V5 | Label-masking correctness | L2 | `pytest tests/test_training.py -k LabelMasking` | PASS with the real Qwen2.5 tokenizer — only response tokens unmasked, padding = -100 |
| V6 | LoRA training plumbing | L2 | `python scripts/train.py --config configs/smoke.yaml --max-train-samples 24` | PROVENANCE RUN 2 — real optimizer steps on `Qwen/Qwen2.5-0.5B` (CPU), 0.22% trainable, adapter saved (15 MB) + reload/merge verified, MLflow run `7c58dbfac89b...` in `sqlite:///mlflow.db` |
| V7 | Evaluation + error-analysis pipeline | L2 | `evaluate.py --model-type base` then `--model-type lora --adapter-path …` + `_error_analysis_vs_base` | PROVENANCE RUN 3 — per-example dump, fixed `sample_id`s, greedy decoding, bootstrap CI, paired diff, `error_analysis_lora_vs_base.json`, `--from-predictions` recompute matches |
| V8 | Config composition (`defaults:` merge) | L1 | `test_training.py::TestTrainingConfigs` + `inference.py --smoke-test` | PASS |
| V9 | Ruff | L1 | `ruff check .` | **All checks passed** |
| V10 | mypy | L1 | `mypy src scripts` | **Success: no issues found in 29 source files** |
| V11 | Editable install + entry points | L1 | `pip install -e .` | EXIT 0; `prepare-data`/`train-model`/`evaluate-model`/`finance-infer`/`judge-eval` resolve |
| V12 | Data pipeline reproducibility | L2 | `prepare_data.py --seed 42` run twice | identical split sizes (44,353 / 2,609 / 5,218) and `split_membership_sha256` digests |

## NOT executed (blocked) — with the exact blocker

| What | Blocker | Smallest next step |
|---|---|---|
| Base (1.5B) evaluation on the test set | ~4.8 GB free RAM; 1.5B fp32 ≈ 6 GB → OOM. No GPU. | Run `python scripts/evaluate.py --model-type base --num-samples 200` on a machine with ≥16 GB RAM or any CUDA GPU |
| Full LoRA training (1.5B, 3 epochs) | No GPU. CPU ETA is days. | `python scripts/train.py --config configs/lora.yaml` on a ≥12 GB-VRAM GPU (~1–2 h on a T4) |
| QLoRA training + 4-bit inference + quantization comparison | `bitsandbytes` 4-bit needs CUDA; unavailable and no cp314 wheel. | `python scripts/train.py --config configs/qlora.yaml` on a CUDA host (≥6 GB VRAM) |
| Ablation sweep | Same as LoRA (18 runs). | `python scripts/train.py --config configs/lora.yaml --ablation` |
| LLM-as-judge run | `JUDGE_API_BASE` / `JUDGE_API_KEY` not set. | `JUDGE_API_BASE=… JUDGE_API_KEY=… python scripts/judge_eval.py --base experiments/eval_results/base_predictions.jsonl --finetuned experiments/eval_results/lora_predictions.jsonl` |
| Docker build / run / endpoint calls | Docker daemon not running on this host. | `docker build -t finance-llm-api . && docker compose up` then `curl localhost:8000/health` |

## Lint / type

- `ruff check .` → **All checks passed** (rulesets E/F/I/UP/B/C4/BLE/RUF/SIM).
- `mypy src scripts` → **Success: no issues found in 29 source files**
  (`follow_imports = silent` — a full recursive mypy over the ML deps' stubs
  hangs on Python 3.14, so imports into `torch`/`transformers`/`datasets` are
  not followed).
- `pytest -q` → **105 passed**.

## Environment migration note

The repo originally pinned torch 2.3 / transformers 4.44 / trl 0.9.6 /
bitsandbytes 0.43 (a mid-2024 stack for Python 3.10). None of those have wheels
for the only interpreter available here (3.14). The code was migrated to the
current stack (torch 2.14, transformers 5.16, peft 0.20, no trl) and the
training path was rewritten to use the plain HF `Trainer` + an explicit
response-masking collator (`src/training/data.py`) instead of TRL's
ever-changing `SFTTrainer`. `bitsandbytes` remains GPU/CUDA-only.
