"""
Integration test: exercise the REAL inference path through the FastAPI app
(no mocks) with a tiny model. Skipped automatically if the model can't be
fetched (offline) or torch is missing.

Run just this file:  pytest tests/test_api_integration.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def real_client():
    import os

    from fastapi.testclient import TestClient

    import src.api.main as api_module

    # Force a real load of the tiny model via the lifespan.
    os.environ["INFERENCE_MODEL_PATH"] = TINY_MODEL
    os.environ["INFERENCE_ADAPTER_PATH"] = ""
    os.environ["LOAD_IN_4BIT"] = "false"
    os.environ.pop("API_SKIP_MODEL_LOAD", None)
    api_module._predictor = None

    try:
        with TestClient(api_module.app) as client:
            if api_module._predictor is None:
                pytest.skip(f"could not load {TINY_MODEL}: {api_module._model_load_error}")
            yield client
    finally:
        api_module._predictor = None


class TestRealInferencePath:
    def test_health_reports_ready(self, real_client):
        data = real_client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_size_mb"] > 0
        assert data["device"] in ("cpu", "cuda")

    def test_generate_runs_a_real_forward_pass(self, real_client):
        resp = real_client.post("/generate", json={
            "instruction": "What is a mutual fund?",
            "max_new_tokens": 16,
            "do_sample": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["response"], str)
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0
        assert body["latency_ms"] >= 0

    def test_evaluate_endpoint_still_works(self, real_client):
        resp = real_client.post("/evaluate", json={
            "reference": "Inflation is a rise in prices over time.",
            "candidate": "Inflation is a rise in prices over time.",
        })
        assert resp.status_code == 200
        assert resp.json()["rouge1"] > 0.9
