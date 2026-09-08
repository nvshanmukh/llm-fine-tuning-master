# Experiment Provenance

Every executed run, with enough detail to trace or reproduce it. Runs here are
**L1/L2 plumbing** on this CPU-only host — they are NOT benchmark results.

---

## RUN 1 — Data preparation (L3, data only)

| Field | Value |
|---|---|
| Command | `python scripts/prepare_data.py --model-name Qwen/Qwen2.5-1.5B --max-seq-len 512 --seed 42` |
| Dataset | `gbharti/finance-alpaca`, HF default revision, single `train` split, 68,912 rows |
| Tokenizer | `Qwen/Qwen2.5-1.5B` (default revision) |
| Seed | 42 |
| Code state | this branch (see `git rev-parse HEAD`) |
| Host | Windows 11, Python 3.14.4, CPU only |
| Outputs | `data/processed/{train,validation,test}.json` (git-ignored), `stats.json`, `splits_manifest.json`, `leakage_report.json` (committed) |
| Result | 52,180 examples kept → train 44,353 / val 2,609 / test 5,218; cross-split leakage 0/0/0; truncation rate @512 = 1.22% |
| Reproducible | yes — `splits_manifest.json.split_membership_sha256` is stable for seed 42 |

---

## RUN 2 — LoRA training plumbing (L2)

| Field | Value |
|---|---|
| Command | `python scripts/train.py --config configs/smoke.yaml --max-train-samples 24` |
| Purpose | prove the training path: PEFT target resolution, real optimizer steps, response-only masking, adapter save, MLflow logging |
| Base model | **`Qwen/Qwen2.5-0.5B`** (a sibling of the 1.5B target — the target OOMs at 4.8 GB free RAM) |
| Dataset subset | first 24 train / 6 val examples from `data/processed/` |
| Config | LoRA r=8, alpha=16, targets `q_proj,k_proj,v_proj,o_proj`; `max_steps=3`, batch 1, lr 2e-4, fp32, no grad checkpointing, seq 256 |
| Seed | 42 |
| Host | Windows 11, Python 3.14.4, CPU (`torch 2.14.0+cpu`) |
| MLflow | `sqlite:///mlflow.db`, experiment `finance-llm-smoke`, run_id `7c58dbfac89b4fecb29e7720f21ba5b9` |
| Trainable params | 1,081,344 / 495,114,112 (**0.2184%**) — MEASURED |
| Train loss (3 steps) | 2.374 → 2.265 → 2.864 (mean `train_loss` 2.501) — MEASURED, plumbing-scale only |
| Eval loss | 1.869 (6 examples) — MEASURED, plumbing-scale only |
| Wall time | 49.5 s (CPU) |
| Adapter | `experiments/smoke/smoke-lora/final_model/` — `adapter_config.json` + `adapter_model.safetensors` (4.35 MB), **15.06 MB** total dir |
| Adapter reload | verified by RUN 3 (`FinanceLLMPredictor` loads and merges it) |
| Status | `completed` |

> These loss numbers say nothing about model quality: 3 optimizer steps on 24
> examples with a 0.5B model. They only prove gradients flow and the loop runs.

---

## RUN 3 — Evaluation plumbing (L2)

| Field | Value |
|---|---|
| Commands | `python scripts/evaluate.py --model-type base_smoke  --model-path Qwen/Qwen2.5-0.5B --num-samples 5 --skip-bertscore --output-dir experiments/eval_results_smoke`<br>`python scripts/evaluate.py --model-type lora_smoke --model-path Qwen/Qwen2.5-0.5B --adapter-path experiments/smoke/smoke-lora/final_model --num-samples 5 --skip-bertscore --output-dir experiments/eval_results_smoke` |
| Purpose | prove: adapter reload, deterministic (greedy) generation on fixed `sample_id`s, per-example prediction dump, aggregate recompute, bootstrap CI, MLflow eval logging |
| Base model | `Qwen/Qwen2.5-0.5B` |
| Test examples | `test.json` rows 0–4 (identical for both models) |
| Decoding | greedy (`temperature=0.0`, `do_sample=False`), `max_new_tokens=256` |
| Outputs | `<type>_predictions.jsonl`, `<type>_eval_report.json` (with `dataset_provenance`, `predictions_sha256_16`, generation config, `rougeL_ci95`), `error_analysis_lora_vs_base.json` |
| Test fingerprint | `test.json` sha256[:16] = `a7cc705c67b60c50` (recorded in every report) |
| Result (n=3, max_new_tokens=40, greedy) | base ROUGE-L 0.2469 (CI 0.219–0.283) · lora ROUGE-L 0.2549 (CI 0.213–0.283); paired ROUGE-L Δ(lora−base) = +0.008, 95% CI [−0.006, +0.030] → **not significant** |
| Recompute check | `evaluate.py --from-predictions base_predictions.jsonl` reproduces ROUGE-1 0.3186 exactly |
| Note | **3 examples on a 0.5B model — the ROUGE/BLEU numbers are meaningless.** Only the pipeline is validated: adapter reload+merge, fixed `sample_id`s, per-example dump, aggregate recompute, bootstrap CI, paired diff, error-analysis-vs-base, MLflow logging. |

---

## Pending (blocked — see docs/VALIDATION.md)

`base` / `lora` / `qlora` evaluation of **Qwen/Qwen2.5-1.5B**, the ablation
sweep, the quantization comparison, the LLM-judge run, and the Docker build.
Commands are in `README.md` → "Reproducing the full experiments".
