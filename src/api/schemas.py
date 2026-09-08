"""
Pydantic request/response schemas for the FastAPI inference service.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for the POST /generate endpoint."""

    instruction: str = Field(
        ...,
        description="The financial question or instruction.",
        min_length=5,
        max_length=2048,
        examples=["What is the difference between a stock and a bond?"],
    )
    input_context: str = Field(
        default="",
        description="Optional input context to accompany the instruction.",
        max_length=1024,
    )
    max_new_tokens: int = Field(
        default=256,
        ge=16,
        le=1024,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.01,
        le=2.0,
        description="Sampling temperature. Lower = more deterministic.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.01,
        le=1.0,
        description="Top-p nucleus sampling parameter.",
    )
    do_sample: bool = Field(
        default=True,
        description="Whether to use sampling (True) or greedy decoding (False).",
    )


class GenerateResponse(BaseModel):
    """Response body for the POST /generate endpoint."""

    response: str = Field(..., description="Generated response text.")
    model: str = Field(..., description="Model identifier used for generation.")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds.")
    input_tokens: int = Field(..., description="Number of tokens in the formatted prompt.")
    output_tokens: int = Field(..., description="Number of newly generated tokens.")


class EvaluateRequest(BaseModel):
    """Request body for the POST /evaluate endpoint (metrics only; no model needed)."""

    reference: str = Field(..., description="Ground-truth reference answer.", min_length=3)
    candidate: str = Field(..., description="Candidate model answer to evaluate.", min_length=3)
    instruction: str = Field(default="", description="Optional: the original instruction (context only).")
    input_context: str = Field(default="", description="Optional input context (context only).")


class EvaluateResponse(BaseModel):
    """Response body for the POST /evaluate endpoint."""

    rouge1: float
    rouge2: float
    rougeL: float
    bleu4: float
    exact_match: float


class HealthResponse(BaseModel):
    """Response body for the GET /health endpoint."""

    status: str = Field(..., examples=["ok"])
    model: str
    model_size_mb: float | None = None
    device: str
    version: str = "0.1.0"
