"""
FastAPI inference service for the Finance LLM fine-tuning project.

Endpoints:
    GET  /health     -- Health check and model info
    POST /generate   -- Generate a response for a financial question
    POST /evaluate   -- Compute evaluation metrics for a prediction/reference pair

Configuration via environment variables (see .env.example):
    INFERENCE_MODEL_PATH   -- Path to base model or HF model ID
    INFERENCE_ADAPTER_PATH -- Optional path to LoRA/QLoRA adapter
    LOAD_IN_4BIT           -- "true" to load the base model in 4-bit (QLoRA inference)
    API_SKIP_MODEL_LOAD    -- "true" to start the server without loading a model
                              (used by tests / readiness probes only)
    API_HOST               -- API host (default: 0.0.0.0)
    API_PORT               -- API port (default: 8000)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from src.data.prompt_template import format_for_inference
from src.evaluation.metrics import compute_bleu, compute_exact_match, compute_rouge
from src.inference.predict import FinanceLLMPredictor, GenerationConfig
from src.utils.logging_utils import setup_logger

# ---------------------------------------------------------------------------
# Global state (model is loaded once at startup)
# ---------------------------------------------------------------------------
_predictor: FinanceLLMPredictor | None = None
_model_load_error: str | None = None


def _load_predictor() -> None:
    """Load the predictor into module state. Records failures instead of raising."""
    global _predictor, _model_load_error

    model_path = os.environ.get("INFERENCE_MODEL_PATH", "Qwen/Qwen2.5-1.5B")
    revision = os.environ.get("INFERENCE_MODEL_REVISION") or None
    adapter_path = os.environ.get("INFERENCE_ADAPTER_PATH") or None
    load_in_4bit = os.environ.get("LOAD_IN_4BIT", "false").lower() == "true"

    model_id = "base-model"
    if adapter_path:
        model_id = "finetuned-qlora" if load_in_4bit else "finetuned-lora"

    logger.info(f"Loading model: {model_path} | adapter: {adapter_path} | 4bit: {load_in_4bit}")
    try:
        _predictor = FinanceLLMPredictor(
            model_path=model_path,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
            model_id=model_id,
            revision=revision,
        )
        _model_load_error = None
        logger.info("Model loaded successfully")
    except Exception as e:
        _predictor = None
        _model_load_error = f"{type(e).__name__}: {e}"
        logger.error(f"Failed to load model: {_model_load_error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, clean up on shutdown."""
    global _predictor, _model_load_error

    setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
    logger.info("Starting Finance LLM API...")

    if os.environ.get("API_SKIP_MODEL_LOAD", "false").lower() == "true":
        logger.warning("API_SKIP_MODEL_LOAD=true -- starting without a model (not ready to serve)")
        _predictor = None
        _model_load_error = "model loading skipped (API_SKIP_MODEL_LOAD=true)"
    elif _predictor is None:
        _load_predictor()

    yield  # Application runs here

    logger.info("Shutting down Finance LLM API")
    _predictor = None


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Finance LLM Inference API",
    description="Domain-specific financial Q&A powered by fine-tuned Qwen2.5-1.5B",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each incoming request with method, path, and response time.

    Only the method, path, status and duration are logged -- request bodies
    (which may contain user questions) are never written to the logs.
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -- "
        f"status={response.status_code} -- {elapsed_ms:.1f}ms"
    )
    return response


def _device_string() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health / readiness check.

    Returns HTTP 200 with status="ok" only when a model is loaded and ready.
    Returns HTTP 503 when the model is not loaded or failed to load.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=_model_load_error or "Model not loaded",
        )

    return HealthResponse(
        status="ok",
        model=_predictor.model_id,
        model_size_mb=_predictor.model_size_mb,
        device=_device_string(),
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a financial answer for the given instruction.

    The instruction is formatted using the same Alpaca-style prompt template
    used during fine-tuning, ensuring the model sees the expected input format.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail=_model_load_error or "Model not loaded")

    # Format prompt using the same template as training
    prompt = format_for_inference({
        "instruction": request.instruction,
        "input": request.input_context,
    })

    gen_config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.do_sample,
    )

    try:
        result = _predictor.generate(prompt, gen_config)
    except Exception as e:
        logger.error(f"Generation failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Generation error") from e

    return GenerateResponse(
        response=result.response,
        model=result.model_id,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """
    Compute text quality metrics for a prediction/reference pair.

    Useful for programmatic evaluation without needing to load datasets.
    Returns ROUGE, BLEU, and Exact Match scores. This endpoint does not
    require a loaded model.
    """
    try:
        rouge_scores = compute_rouge([request.candidate], [request.reference])
        bleu_scores = compute_bleu([request.candidate], [request.reference])
        em_scores = compute_exact_match([request.candidate], [request.reference])
    except Exception as e:
        logger.error(f"Metric computation failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Metric computation error") from e

    return EvaluateResponse(
        rouge1=rouge_scores["rouge1"],
        rouge2=rouge_scores["rouge2"],
        rougeL=rouge_scores["rougeL"],
        bleu4=bleu_scores["bleu4"],
        exact_match=em_scores["exact_match"],
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler: log server-side, return a generic message to the client."""
    logger.error(f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
