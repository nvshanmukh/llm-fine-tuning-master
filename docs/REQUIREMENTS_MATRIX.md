# Requirement → Evidence Matrix

Status vocabulary: `VERIFIED` (executed with evidence), `IMPLEMENTED_UNVERIFIED`
(code exists, runtime check not done), `PARTIAL`, `MISSING`, `BLOCKED`, `N/A`.

Value vocabulary: `MEASURED`, `DERIVED`, `ESTIMATED`, `PENDING`, `BLOCKED`.

## Environment this was verified in

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.14.4 (only interpreter available) |
| GPU / CUDA | none — CPU only, no `nvidia-smi`, `torch.cuda.is_available() == False` |
| RAM | 16 GB total, ~4.8 GB free at start |
| Docker | installed (29.7.2) but **daemon not running** |
| Key libs (venv) | torch 2.14.0+cpu, transformers 5.16.1, datasets 5.0.1, peft 0.20.0, accelerate 1.14.0, mlflow 3.16.0, trl 1.12 (unused) |

### Scope note

**QLoRA / 4-bit *training* was removed from scope** by decision (no GPU on the
target hardware): `configs/qlora.yaml` and `Dockerfile.train` were deleted and
`bitsandbytes` dropped from the requirements. The project now targets **LoRA**
fine-tuning. Optional int8/int4 *inference* quantization stays in
`scripts/quantize_compare.py` — install `bitsandbytes` on a CUDA host to use it.

### Hard external constraints

1. **No GPU** → LoRA training of the 1.5B target model is not feasible locally.
2. **Python 3.14** → the originally pinned stack (torch 2.3, transformers 4.44,
   trl 0.9.6) has no wheels; the code was migrated to the modern stack that does
   install.
3. **~4.8 GB free RAM** → loading the 1.5B target model in fp32 (~6 GB) OOMs;
   target-model *inference* is also blocked locally. A 0.5B sibling is used for
   the training-plumbing smoke.
4. **Docker daemon down** → image build/run not executed.

---

## Section-by-section

| # | Requirement | Status | Implementation | Acceptance check | Evidence | Remaining action |
|---|---|---|---|---|---|---|
| 1 | Lifecycle wired end-to-end (data→…→deploy) | PARTIAL | all `src/` + `scripts/` + `Dockerfile*` | each stage runs | data/eval/inference/API/tests run (below); training only at plumbing scale | run full LoRA on GPU |
| 6 | Real instruction-following domain | VERIFIED | finance Q&A (`gbharti/finance-alpaca`) | dataset is instruction/input/output finance | dataset downloaded, 68,912 rows, Alpaca schema | — |
| 6 | Fine-tuning defensible vs prompt/RAG | VERIFIED (doc) | README "Why fine-tuning" | argument present, honest | README | — |
| 6 | Dataset source + license documented | VERIFIED | `data/README.md`, `stats.json` | name/version/license recorded | `stats.json.dataset_source`, license string | confirm CC-BY-4.0 on HF card |
| 6 | Real example counts computed | VERIFIED / MEASURED | `make_dataset.py` | counts from code not copied | `data/processed/stats.json`: raw 68,912 → filtered 52,180 | — |
| 6 | Deterministic reproducible splits | VERIFIED / MEASURED | `create_splits`, seed=42 | rerun → identical membership digest | `splits_manifest.json.split_membership_sha256`; `test_data.py::test_split_is_reproducible` | — |
| 6 | Test set isolated from tuning/model selection | VERIFIED | test carved out first; eval uses fixed `sample_id` | code path review + tests | `create_splits` order; `evaluate.py` uses positional `sample_id` | keep discipline in full runs |
| 6 | Exact duplicate detection | VERIFIED / MEASURED | `deduplicate` | count reported | 10,026 exact (instruction,input) dups removed | — |
| 6 | Normalized / near-duplicate detection | VERIFIED / MEASURED | `count_normalized_near_duplicates` | count reported | 0 normalized near-dups (post output-filter) | optional: add MinHash for paraphrase-level |
| 6 | Cross-split leakage check | VERIFIED / MEASURED | `_verify_no_overlap`, `leakage_report.json` | overlaps == 0 | `leakage_report.json`: all overlaps 0 | — |
| 6 | Group-aware splitting | N/A (documented) | `leakage_report.json` | justify N/A | dataset has no doc/group id | — |
| 6 | Malformed / missing values handled | VERIFIED | `load_raw_dataset` adds missing cols; `filter_empty_fields` | runs on real data | 1 empty-field row removed | — |
| 6 | Token statistics from real data | VERIFIED / MEASURED | `filter_by_token_length` | mean/median/p90/95/99/max + truncation rate | `stats.json.token_length_stats`: mean 152.8, p95 319, max 1013, trunc 1.22% | — |
| 6 | Prompt formatting / EOS / masking / truncation tested | VERIFIED | `prompt_template.py`, `training/data.py` | tests assert masking keeps only response | `test_training.py::TestLabelMasking` (real Qwen tokenizer) | — |
| 6 | Machine-readable dataset artifacts | VERIFIED | `save_splits` | files exist & parse | `stats.json`, `splits_manifest.json`, `leakage_report.json` committed | — |
| 7 | Model suitable / 1-4B / license / HF support | VERIFIED (doc) + PARTIAL (runtime) | `Qwen/Qwen2.5-1.5B` | loads, Apache-2.0, LoRA-compatible | tokenizer loads; 0.5B sibling LoRA plumbing runs | load 1.5B for real on GPU |
| 7 | Pin model + tokenizer + dataset revision | VERIFIED | `configs/base.yaml`, `make_dataset.py`, `trainer.py`, `predict.py`, `evaluate.py` | full commit SHAs recorded + used | `model.revision` `8faed761…`, `data.dataset_revision` `c88d3d5e…` threaded through every load; `test_training.py::test_model_and_dataset_revisions_are_pinned`; recorded in `stats.json` + eval reports | — |
| 7 | Pre-train resource estimates labelled | VERIFIED (doc) | README hardware table (ESTIMATED) | marked as estimates | README | replace with MEASURED after GPU run |
| 8 | Base model evaluated before improvement claims | VERIFIED (plumbing, 0.5B) | `evaluate.py --model-type base` | runs, saves predictions before any FT claim | PROVENANCE RUN 3 — base eval produced `base_predictions.jsonl` + report; **1.5B base eval still blocked (RAM)** | run on GPU/large-RAM |
| 8 | LoRA is a real execution path (opt steps, adapter save/reload) | VERIFIED (plumbing) | `trainer.py`, `training/data.py` | real optimizer steps, adapter saved & reloadable | `experiments/smoke/` LoRA run on Qwen2.5-0.5B, CPU (see PROVENANCE) | run at full scale on 1.5B |
| 8 | QLoRA training path | NOT_APPLICABLE (removed from scope) | — | — | `configs/qlora.yaml`, `Dockerfile.train`, `bitsandbytes` dep deleted; project targets LoRA | — |
| 8 | Config-driven hyperparameters | VERIFIED | `configs/*.yaml` + `load_config` defaults-merge | all knobs in YAML | `test_training.py::TestTrainingConfigs` | — |
| 9 | Controlled Base vs LoRA comparison | VERIFIED (plumbing) | `evaluate.py --compare-all` | identical sample_ids, prompt, decoding | PROVENANCE RUN 3 — base & lora scored on the same test rows, same greedy config, same test fingerprint `a7cc705c67b60c50` | run with the 1.5B adapter |
| 9 | Hyperparameter experiment / ablation | IMPLEMENTED_UNVERIFIED | `train.py --ablation`, `configs/ablation.yaml` | grid runs, failures visible | code; `_run_ablation` records per-run errors | run on GPU |
| 9 | MLflow tracking of every run | VERIFIED (plumbing) | `trainer.py`, `evaluate.py`, `callbacks.py` | params/metrics/artifacts logged | smoke run logged to `sqlite:///mlflow.db` | — |
| 10 | Task-appropriate metrics, not train loss | VERIFIED | `metrics.py` (ROUGE/BLEU/BERTScore/EM/len) | implemented + tested | `test_evaluation.py` (24 tests) | — |
| 10 | Per-example predictions saved; aggregates recomputable | VERIFIED | `evaluate.py` writes `*_predictions.jsonl`; `--from-predictions` | recompute matches | code + `_aggregate_from_predictions`; round-trip path | run on real model |
| 10 | Generation settings + sample count stored | VERIFIED | `evaluate.py` report | fields present | `eval_report.json.generation_config`, `dataset_provenance` | — |
| 10 | Metric implementations tested | VERIFIED | `tests/test_evaluation.py` | pass | 24 passed | — |
| 10 | Deterministic eval where appropriate | VERIFIED | default `temperature=0.0` (greedy) | code default | `evaluate.py` main | — |
| 10 | Confidence intervals / paired diffs | VERIFIED | `bootstrap_mean_ci`, `paired_bootstrap_diff` | CI computed, tested, used on real predictions | `test_evaluation.py::TestBootstrap` + PROVENANCE RUN 3 (`rougeL_ci95`, paired Δ with CI) | — |
| 10 | Latency: warm-up, repeats, median/p95, scope noted | VERIFIED (impl) | `predict.warmup`, `evaluate.py` p50/p95 + `latency_note` | code | `_aggregate_from_predictions` | real numbers pending |
| 10 | Memory reporting labelled (CPU/alloc/reserved/peak) | PARTIAL | `hardware.get_gpu_memory_usage` | labels allocated/reserved | code; only meaningful on GPU | measure on GPU |
| 11 | LLM-as-judge with rubric, raw judgments, bias controls | IMPLEMENTED_UNVERIFIED / BLOCKED (creds) | `judge.py`, `scripts/judge_eval.py` | blinded pairwise, A/B swap, not self-judge | code; writes `blocked_missing_credentials` report w/ rerun cmd | set `JUDGE_API_*` and run |
| 12 | Error analysis from real saved predictions | VERIFIED (plumbing) | `error_analysis.py`, `evaluate.py::_error_analysis_vs_base` | base vs ft from jsonl, both improvements & regressions | PROVENANCE RUN 3 — `error_analysis_lora_vs_base.json` generated from saved predictions (summary, error_distribution, top improvements/regressions, paired ROUGE-L diff) | run at scale on 1.5B outputs |
| 13 | Post-training inference quantization vs full precision | PARTIAL (CPU int8 measured; bnb int4/8 optional) | `predict.py` `dynamic_int8`, `scripts/quantize_compare.py` | load, run same test examples, compare quality/size/latency | PROVENANCE RUN 4 — fp32 vs torch dynamic-int8 on Qwen2.5-0.5B (n=3): size ×0.53, latency ×0.66, ROUGE-L −0.197 (**quality regressed**); bnb `int4` recorded as `unavailable` (no fallback) | run bnb int4/8 on CUDA + larger n |
| 14 | Clean inference module (base/LoRA/int8/quant, cached, structured) | VERIFIED (real, small model) | `src/inference/predict.py` | real generate call, adapter reload | `scripts/inference.py --real-smoke-test` PASSED (tiny-gpt2); PROVENANCE RUN 3 loaded + merged a real LoRA adapter | run with 1.5B + adapter |
| 14 | Adapter/model compatibility validation | VERIFIED (impl) | `_validate_adapter_compatibility` | checks adapter_config base | code + smoke adapter reload | — |
| 15 | FastAPI /health /generate /evaluate | VERIFIED | `src/api/main.py` | endpoints, validation, 503 semantics, real inference path | `tests/test_api.py` (24, mocked) + `tests/test_api_integration.py` (3, real `sshleifer/tiny-gpt2` through the lifespan: `/health` ready, `/generate` real forward pass, `/evaluate`) | run once with the 1.5B adapter |
| 15 | Health not "ok" when model failed to load | VERIFIED | lifespan records error, `/health` → 503 | test | `test_returns_503_when_no_model_loaded` | — |
| 16 | Docker build/run + endpoint calls + image size | BLOCKED (daemon down) | `Dockerfile`, `docker-compose.yml` | build, run, curl, size | static review + fixes only | run `docker build` on a host with the daemon |
| 17 | Meaningful tests + lint/type/format | VERIFIED | `tests/` (105), ruff, mypy | counts reported | `pytest` 105 passed; `ruff check .` clean; `mypy src scripts` clean | — |
| 17 | No hardcoded personal paths / secrets; `.env.example`, `.gitignore` | VERIFIED | repo-wide | grep clean | `.env.example`, `.gitignore`, `.dockerignore` updated; no absolute user paths in `src/`/`scripts/`/`configs/` | — |
| 17 | CLI robustness (Windows console UTF-8) | VERIFIED | `setup_logger` reconfigures stdio to utf-8/replace | `inference.py -i` prints non-latin1 output without crash | fixed a real `UnicodeEncodeError` on Windows cp1252 | — |
| 18 | README matches verified state | VERIFIED | `README.md` rewrite | claims ↔ evidence | README + this matrix + `docs/VALIDATION.md` | update results after GPU runs |
| 19 | Validation levels labelled | VERIFIED | `docs/VALIDATION.md` | L1/L2/L3 tagged | that file | — |
| 20 | Compute-limited behavior (largest meaningful validation, commands for pending) | VERIFIED | `docs/VALIDATION.md`, README "Reproducing" | reduced runs labelled; full commands given | those docs | — |
| 22 | Final acceptance gate | `IMPLEMENTATION COMPLETE — FULL EXPERIMENTS PENDING` | — | see gate list | 1.5B LoRA training + evaluation blocked by no-GPU (QLoRA out of scope) | GPU / high-RAM host |
