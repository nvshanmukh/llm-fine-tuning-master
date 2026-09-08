# Finance LLM Fine-Tuning & Evaluation Platform

[![CI](https://github.com/nvshanmukh/llm-fine-tuning-master/actions/workflows/ci.yml/badge.svg)](https://github.com/nvshanmukh/llm-fine-tuning-master/actions/workflows/ci.yml)

Domain-specific instruction-following LLM for **personal-finance Q&A**, built as
an end-to-end ML-engineering project: dataset → validation → isolated splits →
baseline → LoRA fine-tuning → experiment tracking → controlled evaluation →
error analysis → inference quantization → CLI → FastAPI → Docker.

> **Status: `IMPLEMENTATION COMPLETE — EXPERIMENTS NOT YET RUN`.**
> The whole pipeline is implemented, tested (105 tests, CI green) and verified at
> data / plumbing scale. Training and evaluation of the **1.5B target model** are
> a deliberate next step — they need a GPU or a high-RAM box, which the dev
> machine here doesn't have. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for
> exactly what ran and [`docs/REQUIREMENTS_MATRIX.md`](docs/REQUIREMENTS_MATRIX.md)
> for per-requirement evidence. **No performance numbers are claimed until measured.**
>
> **To continue:** run the block under
> [Reproducing the experiments](#reproducing-the-experiments) on a GPU / Colab /
> Kaggle, then paste the numbers from `experiments/eval_results/comparison_table.md`
> into the Results table and the Resume-bullets section.

---

## Why fine-tuning (not just prompting or RAG)

| Approach | Limitation for a finance-advisor assistant |
|---|---|
| Prompting only | base models hedge, ramble, and drift from a consistent answer format |
| RAG | needs a curated KB, adds latency, does not change *response style / instruction-following* |
| Fine-tuning (this project) | directly adapts answer style, length and domain phrasing to curated examples |

RAG and fine-tuning are complementary. This project isolates the **fine-tuning**
contribution and measures whether it actually helps (an honest negative result
is an acceptable outcome).

---

## Architecture (as actually implemented)

```
prepare_data.py ──> data/processed/{train,validation,test}.json
   │                 + stats.json, splits_manifest.json, leakage_report.json
   │  (HF finance-alpaca ─ empty/length/token filters ─ exact+normalized dedup
   │   ─ TEST split carved out FIRST ─ leakage assertion)
   ▼
train.py ──> src/training/trainer.py
   │   HF Trainer + PEFT LoRA, response-only label masking
   │   (src/training/data.py), MLflow tracking (sqlite:///mlflow.db)
   │   ──> experiments/<name>/final_model/  (PEFT adapter)
   ▼
evaluate.py
   │   FinanceLLMPredictor generates on fixed test sample_ids (greedy)
   │   ──> experiments/eval_results/<model>_predictions.jsonl   (source of truth)
   │       <model>_eval_report.json   (ROUGE/BLEU/BERTScore/EM + bootstrap CI + latency p50/p95)
   │       error_analysis_<ft>_vs_base.json   (base vs fine-tuned, from saved predictions)
   │       comparison_table.md/json
   ▼
judge_eval.py (optional)   ── blinded pairwise LLM-as-judge (A/B swapped), OpenAI-compatible
quantize_compare.py        ── fp vs torch dynamic-int8 (CPU); int8/int4 optional (bitsandbytes)
   ▼
src/api/main.py  ── FastAPI: GET /health, POST /generate, POST /evaluate
Dockerfile        ── CPU inference image (non-root, HF cache volume, healthcheck = readiness)
```

---

## Dataset — `gbharti/finance-alpaca`

| Attribute | Value (MEASURED unless noted) |
|---|---|
| Source | [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca) |
| License | Apache-2.0 (code) / CC-BY-4.0 (data) — see dataset card |
| Format | Alpaca `{instruction, input, output}` |
| Raw examples | **68,912** |
| After empty-field filter | 68,911 (−1) |
| After output length 5–600 words | 62,851 (−6,060) |
| After exact `(instruction,input)` dedup | 52,825 (−10,026) |
| After token-length ≤ 512 filter | **52,180** (−645; truncation rate at 512 = **1.22 %**) |
| Normalized near-duplicates | 0 (counted, not removed) |
| Splits (seed 42) | train **44,353** · validation **2,609** · test **5,218** |
| Cross-split leakage | 0 / 0 / 0 |
| Token length (full training text) | mean 152.8 · median 131 · p95 319 · p99 543 · max 1013 |

Regenerate everything (deterministic, ~2 min, CPU):

```bash
python scripts/prepare_data.py --seed 42
```

Provenance artifacts (`data/processed/stats.json`, `splits_manifest.json`,
`leakage_report.json`) are committed; the large split JSONs are git-ignored and
regenerable.

> **Data-quality note:** the 10,026 removed "duplicates" share an
> `(instruction, input)` with another row but often have a *different* `output`
> (same question, different answer). The pipeline keeps the first occurrence.

---

## Model — `Qwen/Qwen2.5-1.5B`

- 1.5B params, Apache-2.0, first-class `transformers`/`peft` support.
- ChatML special tokens (`<|im_start|>` / `<|im_end|>`) are in the tokenizer.
- **Pinned** to HF commit `8faed761…` (`configs/base.yaml` `model.revision`); the
  dataset is pinned to `c88d3d5e…` (`data.dataset_revision`). Every model /
  tokenizer / dataset load threads the revision through and records it in
  `stats.json` and the eval reports.

Prompt template (`src/data/prompt_template.py`) — Alpaca instruction wrapped in
Qwen ChatML; `RESPONSE_TAG = "<|im_start|>assistant\n"` marks where the loss
starts (everything before it is masked to `-100`).

### LoRA config (`configs/lora.yaml`, inherits `configs/base.yaml`)

`r=16, alpha=32, dropout=0.05`, targets
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`; 3 epochs, effective
batch 16, lr 2e-4, cosine schedule. `configs/` composes with a lightweight
`defaults:` merge (`base.yaml` first, then overrides). Training uses the plain HF
`Trainer` + an explicit response-only-masking collator (`src/training/data.py`).

`configs/smoke.yaml` is a CPU-friendly plumbing config (tiny model, a few steps).

---

## Results

**PENDING — not measured on this host.** The base/LoRA/quantized table is
populated only by real runs; see the reproduction commands below.

| Model | Trainable params | Train time | Peak mem | ROUGE-L | BLEU-4 | BERTScore F1 | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| Base (zero-shot) | — | — | PENDING | PENDING | PENDING | PENDING | PENDING |
| LoRA (r=16) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| LoRA + int8 inference | — | — | PENDING | PENDING | PENDING | PENDING | PENDING |

The comparison table is generated from
`experiments/eval_results/comparison_table.md` — do not hand-edit it.

### What *has* run (see `docs/VALIDATION.md` + `docs/PROVENANCE.md`)

| Check | Level | Outcome |
|---|---|---|
| `pytest` | L1 | 105 passed |
| `ruff check .` · `mypy src scripts` | L1 | clean |
| Full data preparation (real dataset + Qwen2.5-1.5B tokenizer) | L3 (data) | numbers in the Dataset table |
| Response-only label masking | L2 | verified with the real Qwen tokenizer |
| Real inference path (`FinanceLLMPredictor`) | L1 | PASS (`sshleifer/tiny-gpt2`) |
| LoRA training loop | L2 | `Qwen/Qwen2.5-0.5B`, CPU — real optimizer steps, 0.22% trainable, adapter saved + reload/merge verified, MLflow logged |
| Evaluation + error-analysis pipeline | L2 | base vs LoRA on fixed test `sample_id`s, per-example dump, bootstrap CI, paired diff, `--from-predictions` recompute matches |
| Inference quantization (fp32 vs CPU dynamic-int8) | L2 | size ×0.53, latency ×0.66, ROUGE-L −0.197 on 0.5B/n=3 (honest negative result) |

---

## Evaluation protocol

- Base and LoRA are scored on the **same** test `sample_id`s (0-based row index
  in `test.json`), same prompt template, same decoding (**greedy by default**,
  `temperature=0.0`).
- `experiments/eval_results/<model>_predictions.jsonl` is the source of truth;
  every aggregate in the report is recomputed from it
  (`python scripts/evaluate.py --from-predictions <file>`).
- ROUGE-L reports a 95 % bootstrap CI; fine-tuned-vs-base reports a **paired**
  bootstrap difference (`paired_bootstrap_diff`).
- Latency: one warm-up generation is discarded, then per-example wall-clock
  (batch 1, includes tokenize + decode); report shows median and p95 plus a
  scope note.
- Metrics: ROUGE-1/2/L, BLEU-4 (smoothed), BERTScore-F1 (`distilbert-base-uncased`),
  exact match, length stats.

### LLM-as-a-judge (optional)

`scripts/judge_eval.py` runs a **blinded pairwise** comparison (fine-tuned vs
base), judging each pair in both A/B orders to cancel position bias. The
candidate models are never used to judge themselves. Without `JUDGE_API_BASE` /
`JUDGE_API_KEY` it writes a `blocked_missing_credentials` report containing the
exact rerun command. Known biases (verbosity, not-ground-truth) are recorded in
the report.

### Error analysis

`error_analysis_<ft>_vs_base.json` is built from the saved predictions and shows
**both** improvements and regressions (top-5 each) plus a persistent-failure
list. The automatic categorizer is heuristic and surface-level
(`incomplete_answer`, `irrelevant_output`, `domain_knowledge_gap`,
`formatting_failure`, `instruction_failure`, `hallucination`-by-repetition,
`acceptable`); `incorrect_reasoning` is deliberately left to human / LLM review.

### Inference quantization

`scripts/quantize_compare.py` runs the **same** test examples (greedy) through
the model in several precisions and reports quality / serialized size / latency:

| mode | needs | status |
|---|---|---|
| `fp` | — | baseline |
| `dynamic_int8` | CPU only (torch) | **measured** — see below |
| `int8` / `int4` | `pip install bitsandbytes` + CUDA | optional; recorded as `unavailable` if absent, never silently substituted |

**Measured (L2, `Qwen/Qwen2.5-0.5B`, n=3, CPU — plumbing scale, not a benchmark):**
fp32 → torch `dynamic_int8`: size ×0.53, latency ×0.66, **ROUGE-L −0.197
(quality regressed badly)**. Naive dynamic int8 without calibration is a poor
tradeoff here; NF4/GPTQ typically degrade far less. Re-run with n≥200 before
concluding anything (`docs/PROVENANCE.md` RUN 4).

---

## CLI

```bash
# no-model pipeline check
python scripts/inference.py --smoke-test

# real model-loading check (tiny model, plumbing only)
python scripts/inference.py --real-smoke-test

# a single question against the base model
python scripts/inference.py -i "What is the difference between a mutual fund and an ETF?"

# with the fine-tuned adapter
python scripts/inference.py -i "Explain compound interest" \
  --adapter-path experiments/lora/final_model

# interactive loop
python scripts/inference.py --interactive
```

## FastAPI

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | 200 + model info **only when a model is loaded**; 503 otherwise (readiness) |
| `POST /generate` | `{instruction, input_context?, max_new_tokens?, temperature?, top_p?, do_sample?}` → response + latency + token counts |
| `POST /evaluate` | `{reference, candidate, ...}` → ROUGE/BLEU/EM (no model needed) |

`API_SKIP_MODEL_LOAD=true` starts the server without a model (probes/tests).
Request bodies are never written to the logs.

## Docker (CPU inference)

```bash
docker build -t finance-llm-api .
docker run -p 8000:8000 \
  -e INFERENCE_MODEL_PATH=Qwen/Qwen2.5-1.5B \
  -e INFERENCE_ADAPTER_PATH=/models/lora/final_model \
  -v $(pwd)/experiments:/models:ro \
  -v hf_cache:/app/.cache/huggingface \
  finance-llm-api
# or: docker compose up --build
```

Non-root user, `HF_HOME=/app/.cache/huggingface` (matches the compose volume),
healthcheck fails whenever `/health` is not 200. **Not built here** (Docker
daemon unavailable) — static review only.

---

## Reproducing the experiments

```bash
pip install -r requirements.txt
cp .env.example .env                                  # set HF_TOKEN if needed

python scripts/prepare_data.py --seed 42              # ~2 min CPU

# LoRA on the 1.5B target -- needs a GPU (~12 GB VRAM) or a high-RAM machine.
python scripts/train.py --config configs/lora.yaml

# LoRA plumbing run that fits on a CPU laptop (tiny model, few steps):
python scripts/train.py --config configs/smoke.yaml --max-train-samples 200

python scripts/evaluate.py --compare-all --num-samples 200   # base + lora
python scripts/train.py --config configs/lora.yaml --ablation

# optional
JUDGE_API_BASE=... JUDGE_API_KEY=... python scripts/judge_eval.py \
  --base experiments/eval_results/base_predictions.jsonl \
  --finetuned experiments/eval_results/lora_predictions.jsonl
python scripts/quantize_compare.py --model-path Qwen/Qwen2.5-1.5B \
  --adapter-path experiments/lora/final_model --modes fp,dynamic_int8

mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q
```

### Hardware

| Task | Needs | Notes |
|---|---|---|
| Data prep | CPU only | ~2 min |
| Smoke LoRA (`configs/smoke.yaml`, Qwen2.5-0.5B) | CPU, ~2 GB RAM | plumbing only |
| Base inference (1.5B) | ~6 GB RAM or 4 GB VRAM | fp32 OOMs at 4.8 GB free |
| LoRA fine-tuning (1.5B) | ~12 GB VRAM, or a high-RAM CPU box (slow) | bf16/fp16 on GPU |
| Inference API (int8) | `pip install bitsandbytes` + CUDA | optional |

---

## Limitations

- **The 1.5B experiments have not been run here** — see status banner. All
  quality numbers are PENDING.
- finance-alpaca outputs are GPT-4-generated; high coverage, not guaranteed
  expert-grade. US-centric skew.
- Dedup is exact + normalized only; paraphrase near-duplicates may remain.
- ROUGE/BLEU measure lexical overlap, not factual correctness; BERTScore is
  semantic but imperfect for finance reasoning.
- LLM-as-judge is biased (verbosity, position) and is not ground truth.
- Error categorization is heuristic; `incorrect_reasoning` needs human review.
- The Docker image ships no auth — put it behind a gateway for real use.

## Future work

- Run the LoRA + ablation matrix on a GPU (or high-RAM box) and fill the results
  table from artifacts.
- Human-calibrate the LLM judge; add MinHash paraphrase dedup.
- GGUF export for CPU deployment; SSE streaming in the API.

## Resume bullets

Not written yet — **on purpose**. Every number a bullet would need (ROUGE-L
delta, trainable-parameter %, training time, latency, memory saving) is `PENDING`
until the L3 runs happen. Fill them from `experiments/eval_results/comparison_table.md`
and the MLflow run once those experiments complete; do not estimate them.

## License

MIT (code). Dataset: Apache-2.0 / CC-BY-4.0. Model: Apache-2.0 (Qwen2.5-1.5B).
