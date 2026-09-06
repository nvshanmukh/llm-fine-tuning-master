"""
Tests for the FastAPI inference service.

Tests are designed to run without a real LLM by mocking the global
predictor. This allows API routing, input validation, and response
schemas to be fully tested in CI without GPU resources.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_predictor():
    """
    A mock FinanceLLMPredictor returning deterministic, fixed responses.
    This avoids any model loading during API tests.
    """
    from src.inference.predict import InferenceResult

    mock = MagicMock()
    mock.model_id = "test-model"
    mock.model_size_mb = 3072.0
    mock.generate.return_value = InferenceResult(
        prompt="formatted test prompt",
        response="A mutual fund pools money from many investors to purchase a diversified portfolio.",
        model_id="test-model",
        latency_ms=42.0,
        input_tokens=55,
        output_tokens=18,
    )
    return mock


@pytest.fixture
def client(mock_predictor):
    """
    FastAPI TestClient with the global predictor replaced by the mock.
    Bypasses the lifespan model-loading step entirely.
    """
    import src.api.main as api_module

    api_module._predictor = mock_predictor
    from src.api.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_includes_model_name(self, client):
        data = client.get("/health").json()
        assert data["model"] == "test-model"

    def test_includes_model_size(self, client):
        data = client.get("/health").json()
        assert "model_size_mb" in data
        assert data["model_size_mb"] == pytest.approx(3072.0)

    def test_returns_503_when_no_model_loaded(self):
        """Health check must return 503 when the predictor is None."""
        import src.api.main as api_module

        original = api_module._predictor
        try:
            api_module._predictor = None
            from src.api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get("/health")
            assert response.status_code == 503
        finally:
            api_module._predictor = original


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    def test_returns_200_minimal_request(self, client):
        response = client.post("/generate", json={
            "instruction": "What is a mutual fund?",
        })
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/generate", json={
            "instruction": "What is a mutual fund?",
        }).json()
        assert "response" in data
        assert "model" in data
        assert "latency_ms" in data
        assert "input_tokens" in data
        assert "output_tokens" in data

    def test_response_text_is_non_empty(self, client):
        data = client.post("/generate", json={
            "instruction": "What is inflation?",
        }).json()
        assert len(data["response"]) > 0

    def test_model_name_matches_mock(self, client):
        data = client.post("/generate", json={
            "instruction": "What is a bond?",
        }).json()
        assert data["model"] == "test-model"

    def test_latency_is_non_negative(self, client):
        data = client.post("/generate", json={
            "instruction": "Define bear market.",
        }).json()
        assert data["latency_ms"] >= 0

    def test_accepts_all_optional_params(self, client):
        response = client.post("/generate", json={
            "instruction": "Explain compound interest.",
            "input_context": "Assume monthly compounding.",
            "max_new_tokens": 128,
            "temperature": 0.5,
            "top_p": 0.95,
            "do_sample": True,
        })
        assert response.status_code == 200

    def test_rejects_instruction_too_short(self, client):
        """Instructions shorter than 5 characters must be rejected with 422."""
        response = client.post("/generate", json={"instruction": "Hi"})
        assert response.status_code == 422

    def test_rejects_temperature_out_of_range(self, client):
        """Temperature must be in [0.01, 2.0]."""
        response = client.post("/generate", json={
            "instruction": "What is inflation?",
            "temperature": 5.0,
        })
        assert response.status_code == 422

    def test_rejects_max_new_tokens_too_low(self, client):
        """max_new_tokens must be >= 16."""
        response = client.post("/generate", json={
            "instruction": "What is inflation?",
            "max_new_tokens": 5,
        })
        assert response.status_code == 422

    def test_rejects_empty_body(self, client):
        response = client.post("/generate", json={})
        assert response.status_code == 422

    def test_returns_503_when_model_not_loaded(self):
        """Generate must return 503 if no model is loaded."""
        import src.api.main as api_module

        original = api_module._predictor
        try:
            api_module._predictor = None
            from src.api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.post("/generate", json={"instruction": "What is inflation?"})
            assert response.status_code == 503
        finally:
            api_module._predictor = original


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------

class TestEvaluateEndpoint:
    def test_returns_200(self, client):
        response = client.post("/evaluate", json={
            "instruction": "What is inflation?",
            "reference": "Inflation is the rate of increase in prices over time.",
            "candidate": "Inflation means rising prices in an economy.",
        })
        assert response.status_code == 200

    def test_returns_rouge_scores(self, client):
        data = client.post("/evaluate", json={
            "instruction": "What is inflation?",
            "reference": "Inflation is the rate of increase in prices over time.",
            "candidate": "Inflation is rising prices over time.",
        }).json()
        assert "rouge1" in data
        assert "rouge2" in data
        assert "rougeL" in data
        assert 0.0 <= data["rouge1"] <= 1.0
        assert 0.0 <= data["rougeL"] <= 1.0

    def test_returns_bleu_score(self, client):
        data = client.post("/evaluate", json={
            "instruction": "What is inflation?",
            "reference": "Inflation is rising prices.",
            "candidate": "Inflation is rising prices.",
        }).json()
        assert "bleu4" in data

    def test_returns_exact_match(self, client):
        data = client.post("/evaluate", json={
            "instruction": "What is inflation?",
            "reference": "Inflation is rising prices.",
            "candidate": "Inflation is rising prices.",
        }).json()
        assert "exact_match" in data

    def test_perfect_match_gives_high_rouge(self, client):
        text = "Inflation is the sustained increase in the general price level of goods and services."
        data = client.post("/evaluate", json={
            "instruction": "Define inflation.",
            "reference": text,
            "candidate": text,
        }).json()
        assert data["rouge1"] > 0.9
        assert data["exact_match"] == pytest.approx(1.0)

    def test_rejects_short_reference(self, client):
        """Reference shorter than 3 characters must be rejected."""
        response = client.post("/evaluate", json={
            "instruction": "What is inflation?",
            "reference": "X",
            "candidate": "Some reasonable candidate answer.",
        })
        assert response.status_code == 422

    def test_rejects_missing_reference(self, client):
        response = client.post("/evaluate", json={
            "instruction": "What is inflation?",
            "candidate": "Some answer.",
        })
        assert response.status_code == 422

    def test_accepts_optional_input_context(self, client):
        response = client.post("/evaluate", json={
            "instruction": "What is inflation?",
            "input_context": "In the context of the US economy.",
            "reference": "Inflation is rising prices over time.",
            "candidate": "Inflation is the increase in prices.",
        })
        assert response.status_code == 200
