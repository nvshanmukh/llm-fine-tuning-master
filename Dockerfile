# =============================================================================
# Finance LLM Inference API - Production Docker Image (CPU)
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): install inference-only Python deps into a prefix
#   Stage 2 (runtime): lean image with a non-root user
#
# Build:
#   docker build -t finance-llm-api .
#
# Run (base model, downloaded at startup into a mounted cache):
#   docker run -p 8000:8000 \
#     -e INFERENCE_MODEL_PATH=Qwen/Qwen2.5-1.5B \
#     -v hf_cache:/app/.cache/huggingface \
#     finance-llm-api
#
# Run (local fine-tuned adapter, mounted read-only):
#   docker run -p 8000:8000 \
#     -e INFERENCE_MODEL_PATH=Qwen/Qwen2.5-1.5B \
#     -e INFERENCE_ADAPTER_PATH=/models/lora/final_model \
#     -v $(pwd)/experiments:/models:ro \
#     -v hf_cache:/app/.cache/huggingface \
#     finance-llm-api
# =============================================================================

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-inference.txt .
RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir --prefix=/install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements-inference.txt

# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY configs/ ./configs/

ENV INFERENCE_MODEL_PATH=Qwen/Qwen2.5-1.5B \
    INFERENCE_ADAPTER_PATH="" \
    LOAD_IN_4BIT=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1 \
    LOG_LEVEL=INFO \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user; owns the app dir and the HF cache it writes to.
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --home-dir /app --no-create-home appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Fails (unhealthy) whenever /health is not 200 -- i.e. whenever the model
# is not loaded. Long start period covers first-time model download.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["sh", "-c", "uvicorn src.api.main:app --host $API_HOST --port $API_PORT --workers $API_WORKERS"]
