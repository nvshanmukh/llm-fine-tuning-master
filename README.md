# Finance LLM Fine-Tuning & Evaluation Platform

> **Domain-Specific Instruction-Following LLM** via QLoRA on `Qwen/Qwen2.5-1.5B`  
> A production-quality ML engineering portfolio project demonstrating the complete LLM fine-tuning lifecycle.

---

## Overview

This project fine-tunes `Qwen/Qwen2.5-1.5B` — a state-of-the-art 1.5B parameter open-source model — on the `gbharti/finance-alpaca` financial Q&A dataset using parameter-efficient fine-tuning (PEFT). The system demonstrates the complete pipeline from raw data to a deployed inference API.

**What this project demonstrates:**
- Data preprocessing with rigorous leakage prevention
- Baseline measurement *before* claiming improvement
- Controlled LoRA vs QLoRA experiments with tracked hyperparameters
- Quantitative evaluation (ROUGE, BLEU, BERTScore) + qualitative error analysis
- LLM-as-a-Judge evaluation with documented limitations
- Ablation study: does LoRA rank materially affect performance?
- Post-training quantization comparison
- Production FastAPI + Docker inference service

---

## Why Fine-Tuning? (Not just RAG or Prompting)

For a **financial advisor assistant**, vanilla prompting or RAG has specific limitations:

| Approach | Limitation |
|---|---|
| **Prompting only** | Base LLMs lack deep domain calibration; responses are often verbose, generic, or hedge excessively |
| **RAG** | Requires a curated knowledge base; latency increases; doesn't improve *reasoning style* or instruction-following format |
| **Fine-tuning** | Directly adapts response style, tone, length, and domain-specific reasoning to match curated expert examples |

RAG and fine-tuning are **complementary**, not mutually exclusive. This project isolates the fine-tuning contribution specifically.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                                  │
│  gbharti/finance-alpaca                                             │
│  ↓ filter_empty_fields                                              │
│  ↓ filter_by_output_length (5–600 words)                           │
│  ↓ filter_by_token_length (<= 512 tokens)                           │
│  ↓ deduplicate (MD5 hash of instruction+input)                      │
│  ↓ create_splits [TEST carved out FIRST → then val/train]           │
│  ┌────────────┬──────────────┬───────────┐                          │
│  │  train.json│ validation.  │  test.json│  ← isolated, never       │
│  │  (~85%)    │   json (~5%) │  (10%)    │    seen during training   │
│  └────────────┴──────────────┴───────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      BASELINE EVALUATION                            │
│  Qwen/Qwen2.5-1.5B (base) → generate on test.json                 │
│  → ROUGE-1/2/L, BLEU-4, BERTScore, latency, GPU memory            │
│  → ESTABLISHES BASELINE (no claims of improvement without this)    │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FINE-TUNING EXPERIMENTS                        │
│                                                                     │
│  Experiment A: Base model (zero-shot)                               │
│  Experiment B: LoRA (r=16, bfloat16, 3 epochs)                     │
│  Experiment C: QLoRA (r=16, 4-bit NF4, 3 epochs)                   │
│                                                                     │
│  Ablation: r={8,16,32} × lr={1e-4,2e-4,5e-4} × epochs={2,3}       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MLflow Tracking: config, loss curves, metrics, GPU memory   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION & ANALYSIS                          │
│  Automatic: ROUGE-1/2/L, BLEU-4, BERTScore F1, Exact Match        │
│  LLM-as-Judge: Correctness, Relevance, Completeness (1–5 scale)   │
│  Error Analysis: hallucination / incomplete / irrelevant / etc.    │
│  Quantization: Compare full-precision vs int8 inference            │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT                                     │
│  FastAPI: POST /generate | POST /evaluate | GET /health            │
│  Docker: Multi-stage image (inference only, no training deps)      │
│  docker-compose: Single-command deployment with HF cache volume    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Attribute | Value |
|---|---|
| **Source** | [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca) |
| **License** | Apache 2.0 / CC-BY-4.0 |
| **Domain** | Personal finance, investing, economics, accounting |
| **Raw size** | ~68,900 examples |
| **After filtering** | ~53,000–58,000 examples |
| **Format** | Alpaca: `{instruction, input, output}` |
| **Max seq length** | 512 tokens (Qwen2.5-1.5B tokenizer) |

**Filtering steps:**
1. Remove empty `instruction` or `output`
2. Remove outputs with <5 or >600 words
3. Remove examples where full prompt+response exceeds 512 tokens
4. Deduplicate on `MD5(instruction.lower() + input.lower())`
5. Test split **first** (prevents leakage), then val/train

---

## Methodology

### Model: Qwen/Qwen2.5-1.5B

Selected because:
- **1.5B parameters**: fits in <3GB VRAM (bf16), <1GB VRAM (4-bit QLoRA)
- **State-of-the-art**: outperforms similar-size models on instruction-following benchmarks (2024)
- **HuggingFace native**: full transformers + PEFT support
- **Apache 2.0 license**: commercially usable
- **ChatML tokens**: `<|im_start|>` / `<|im_end|>` natively supported

### Prompt Template (Alpaca + Qwen ChatML)

```
<|im_start|>system
You are a knowledgeable financial advisor assistant. Answer questions accurately,
concisely, and in plain English. Do not make up numbers or facts. If you are unsure, say so.
<|im_end|>
<|im_start|>user
### Instruction:
{instruction}

### Input:
{input}
<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

### LoRA Configuration

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Trainable parameters: ~5.5M / 1,543M total (**0.36% of parameters**).

### QLoRA Configuration

Same as LoRA but base model loaded in **4-bit NF4** via `bitsandbytes`:
```yaml
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true
```

---

## Experiments

### Hyperparameter Configuration

All experiments are fully configuration-driven (see `configs/`):

```yaml
# configs/qlora.yaml (excerpt)
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4   # effective batch = 16
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true
  max_seq_length: 512
```

### Ablation Study

The ablation explores the question:
> **"Does increasing LoRA rank materially improve task performance relative to the additional trainable parameters?"**

Grid searched:
- LoRA rank `r` ∈ {8, 16, 32}
- Learning rate ∈ {1e-4, 2e-4, 5e-4}  
- Epochs ∈ {2, 3}

---

## Results

> **Note:** The results table below will be populated after training runs complete on a cloud GPU. Placeholder values are marked `[PENDING]`. The code is fully functional and has been tested end-to-end with a smoke test on CPU.

| Model | Trainable Params | Training Time | GPU Memory | ROUGE-L | BLEU-4 | BERTScore F1 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base (zero-shot) | — | — | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| LoRA (r=16) | ~5.5M (0.36%) | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| QLoRA (r=16) | ~5.5M (0.36%) | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

**Update this table with real values after running on GPU. Never fabricate numbers.**

---

## Error Analysis

Error categories:
- `hallucination` — factual claims not grounded in the reference
- `incomplete_answer` — response cut off or too short (<10 words)
- `incorrect_reasoning` — logical errors in explanation
- `instruction_failure` — does not follow instruction format
- `irrelevant_output` — off-topic response
- `formatting_failure` — structural issues
- `domain_knowledge_gap` — missing specific financial terminology
- `acceptable` — reasonable response

Error analysis results are saved to `experiments/eval_results/{model}_error_analysis.json`.

---

## LLM-as-a-Judge Evaluation

The project implements an optional LLM-as-a-Judge evaluation layer (see `src/evaluation/judge.py`).

**Dimensions evaluated (1–5 scale):**
- Correctness
- Relevance
- Completeness
- Instruction Following
- Hallucination Score (5 = no hallucination)

**Documented Limitations:**
1. LLM judges exhibit **verbosity bias** — longer answers are often scored higher regardless of quality.
2. LLM judge scores are **not ground truth** — treat as a supplementary signal.
3. Without human annotation as calibration, scores are relative to the judge LLM's own biases.
4. GPT-4 judges may not accurately evaluate specialized finance terminology.
5. **Always combine** automatic metrics + LLM judge + qualitative human review.

---

## Quantization

Post-training quantization is implemented in `src/inference/quantize.py`.

**Comparison:**
- **Full precision (bfloat16)**: Maximum quality, ~3GB VRAM for 1.5B model
- **4-bit NF4 (QLoRA inference)**: ~1GB VRAM, typically <3% quality degradation on ROUGE
- **int8 (bitsandbytes)**: ~1.5GB VRAM, intermediate tradeoff

Model size comparison is measured with `compare_model_sizes()` using actual file sizes.

---

## Deployment

### FastAPI Service

```bash
# Start the inference API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up --build
```

**Endpoints:**

```
GET  /health    → Model status, device info, model size
POST /generate  → Generate financial answer
POST /evaluate  → Compute ROUGE/BLEU/EM for a prediction/reference pair
```

**Example request:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "What is the difference between a mutual fund and an ETF?",
    "max_new_tokens": 256,
    "temperature": 0.1
  }'
```

**Example response:**

```json
{
  "response": "A mutual fund is actively managed and priced once per day...",
  "model": "finetuned-qlora",
  "latency_ms": 1842,
  "input_tokens": 87,
  "output_tokens": 112
}
```

### Docker

```bash
# Build inference image
docker build -t finance-llm-api .

# Run with local fine-tuned adapter
docker run -p 8000:8000 \
  -e INFERENCE_MODEL_PATH=Qwen/Qwen2.5-1.5B \
  -e INFERENCE_ADAPTER_PATH=/models/qlora \
  -e LOAD_IN_4BIT=true \
  -v ./experiments:/models:ro \
  finance-llm-api

# Build training image (GPU)
docker build -f Dockerfile.train -t finance-llm-train .
```

---

## Reproducing Experiments

```bash
# 1. Clone and set up
git clone <repo>
cd llm-finetuning-platform
pip install -r requirements.txt  # or requirements-dev.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set HF_TOKEN if needed

# 3. Prepare data
python scripts/prepare_data.py \
  --model-name Qwen/Qwen2.5-1.5B \
  --max-seq-len 512 \
  --seed 42

# 4. Validate pipeline (no GPU needed)
python scripts/inference.py --smoke-test

# 5. Train LoRA (requires GPU with ≥6GB VRAM)
python scripts/train.py --config configs/lora.yaml

# 6. Train QLoRA (requires GPU with ≥4GB VRAM)
python scripts/train.py --config configs/qlora.yaml

# 7. Evaluate all models
python scripts/evaluate.py --compare-all --num-samples 200

# 8. Run ablation study
python scripts/train.py --config configs/lora.yaml --ablation

# 9. View MLflow UI
mlflow ui --backend-store-uri ./mlruns

# 10. Run tests
pytest tests/ -v
```

---

## Hardware Requirements

| Task | Min VRAM | Recommended |
|---|---|---|
| Data preparation | CPU only | CPU only |
| Base model inference | 4 GB | 6 GB |
| LoRA fine-tuning | 6 GB | 16 GB |
| QLoRA fine-tuning | 4 GB | 8 GB |
| Inference API (4-bit) | 2 GB | 4 GB |

**Free GPU options for students:**
- Google Colab T4 (16 GB VRAM) — free tier
- Kaggle P100 (16 GB VRAM) — free tier
- Google Colab A100 (40 GB) — Colab Pro

---

## Project Structure

```
llm-finetuning-platform/
├── data/
│   ├── raw/                    # HF cache (git-ignored)
│   ├── processed/              # Filtered, split datasets (JSON)
│   │   ├── train.json
│   │   ├── validation.json
│   │   ├── test.json           # ISOLATED — never used in training
│   │   └── stats.json          # Dataset statistics card
│   └── README.md               # Dataset documentation
│
├── configs/
│   ├── base.yaml               # Shared defaults
│   ├── lora.yaml               # Full-precision LoRA config
│   ├── qlora.yaml              # 4-bit QLoRA config
│   └── ablation.yaml           # Ablation sweep grid
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py     # Data pipeline (download, filter, split)
│   │   └── prompt_template.py  # Alpaca + Qwen ChatML prompt formatting
│   ├── training/
│   │   ├── trainer.py          # LoRA/QLoRA training logic + MLflow
│   │   └── callbacks.py        # MLflowDetailedCallback
│   ├── evaluation/
│   │   ├── metrics.py          # ROUGE, BLEU, BERTScore, EM
│   │   ├── judge.py            # LLM-as-a-Judge evaluation
│   │   └── error_analysis.py   # Failure categorization
│   ├── inference/
│   │   ├── predict.py          # FinanceLLMPredictor (base + adapter)
│   │   └── quantize.py         # Post-training quantization
│   ├── api/
│   │   ├── main.py             # FastAPI app (lifespan, endpoints)
│   │   └── schemas.py          # Pydantic request/response models
│   └── utils/
│       ├── config_utils.py     # OmegaConf YAML loading
│       ├── logging_utils.py    # Loguru setup
│       ├── seed.py             # Reproducibility seeds
│       └── hardware.py         # GPU/CPU introspection
│
├── scripts/
│   ├── prepare_data.py         # Data prep CLI
│   ├── train.py                # Training CLI (LoRA/QLoRA/ablation)
│   ├── evaluate.py             # Evaluation CLI (single/compare-all)
│   └── inference.py            # Interactive inference CLI + smoke-test
│
├── experiments/                # Saved model checkpoints and eval reports
│   └── eval_results/
│
├── tests/
│   ├── test_data.py            # Data pipeline tests (splits, leakage)
│   ├── test_evaluation.py      # Metric and error analysis tests
│   ├── test_api.py             # FastAPI endpoint tests (mocked model)
│   └── test_inference.py       # Inference pipeline smoke tests
│
├── Dockerfile                  # Multi-stage inference image
├── Dockerfile.train            # GPU training image
├── docker-compose.yml          # Inference service orchestration
├── .dockerignore
├── requirements.txt            # Full training dependencies
├── requirements-inference.txt  # Lightweight inference-only dependencies
├── requirements-dev.txt        # Development extras
├── pyproject.toml              # Build config + tool settings
├── .env.example                # Environment variable template
├── .gitignore
└── README.md                   # This file
```

---

## Limitations

**Dataset:**
- Finance-alpaca is synthetic (generated by GPT-4). It has high coverage but outputs may not reflect real expert financial advice.
- Deduplication removes exact matches only — near-duplicate paraphrases may remain.

**Compute:**
- Without a GPU, training is not feasible locally. QLoRA makes this significantly more accessible (4GB VRAM minimum).

**Evaluation:**
- ROUGE/BLEU measure lexical overlap, not factual correctness. A fluent but wrong answer can score well.
- BERTScore is better at semantic similarity but still imperfect for finance-specific reasoning.
- LLM-as-a-judge is biased (verbosity bias, position bias) and should not be treated as ground truth.

**Hallucination:**
- No hallucination detection tool is perfect. The error analysis is heuristic-based for the automatic pass and requires human review for validation.

**Deployment:**
- The Docker image has no authentication. For production use, add an API key middleware or deploy behind a reverse proxy with authentication.

---

## Future Work

- **DPO/ORPO alignment**: Apply preference optimization after SFT to reduce hallucination
- **RAG integration**: Combine fine-tuned model with a finance document retriever
- **Larger model**: Run LoRA on Qwen2.5-7B with the same pipeline
- **Human evaluation**: Calibrate LLM judge scores against domain expert annotations
- **Streaming inference**: Add server-sent events (SSE) to the FastAPI service for streaming responses
- **GGUF quantization**: Export via llama.cpp for CPU-only deployment

---

## Resume Bullets (fill in with actual measured values)

> These are target bullet structures. Replace `[X]` with real numbers from your experiments.

- "Fine-tuned `Qwen/Qwen2.5-1.5B` using QLoRA on `gbharti/finance-alpaca` (~55K examples), achieving **X% improvement in ROUGE-L** over the pretrained baseline with only **0.36% trainable parameters** (5.5M/1.5B)."
- "Conducted **N controlled LoRA/QLoRA experiments** across rank, learning rate, and epoch configurations using MLflow, analyzing performance, GPU memory, and inference latency tradeoffs."
- "Deployed the fine-tuned model as a **FastAPI + Docker inference service**, reducing model memory footprint by ~60% via 4-bit NF4 quantization with <X% ROUGE degradation."

---

## License

MIT License. Dataset: Apache 2.0 / CC-BY-4.0 (finance-alpaca). Model: Apache 2.0 (Qwen2.5).
