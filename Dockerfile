# =============================================================================
# Finance LLM Inference API — Production Docker Image
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies in isolation
#   Stage 2 (runtime): Lean production image
#
# Build:
#   docker build -t finance-llm-api .
#
# Run (CPU mode, with local model):
#   docker run -p 8000:8000 \
#     -e INFERENCE_MODEL_PATH=Qwen/Qwen2.5-1.5B \
#     -e HF_TOKEN=hf_your_token \
#     finance-llm-api
#
# Run (GPU mode):
#   docker run --gpus all -p 8000:8000 \
#     -e INFERENCE_MODEL_PATH=/models/qlora \
#     -v /local/model:/models \
#     finance-llm-api
# =============================================================================

ARG PYTHON_VERSION=3.11

# -----------------------------------------------------------------------------
# Stage 1: Builder — install dependencies
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker layer caching
COPY requirements-inference.txt .

# Install Python dependencies into a local prefix
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir --prefix=/install -r requirements-inference.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime — lean production image
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app

# System runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY configs/ ./configs/

# Environment defaults (override at runtime)
ENV INFERENCE_MODEL_PATH=Qwen/Qwen2.5-1.5B \
    INFERENCE_ADAPTER_PATH="" \
    LOAD_IN_4BIT=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --no-create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["sh", "-c", \
     "uvicorn src.api.main:app --host $API_HOST --port $API_PORT --workers $API_WORKERS"]
