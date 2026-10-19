"""HTTP contract for the text-only inference service (no weights needed)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def service():
    path = Path(__file__).resolve().parents[2] / "deploy/vercel/embeddinggemma-service/main.py"
    spec = importlib.util.spec_from_file_location("embeddinggemma_inference_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "   "},
        {"text": "x" * 2049},
        {"text": "query", "modality": "audio"},
        {"text": "query", "audio_url": "https://example.com/audio.wav"},
    ],
)
def test_rejects_non_text_queries_before_loading_weights(service, monkeypatch, payload):
    def unexpected_load():
        pytest.fail("Invalid requests must not load the model")

    monkeypatch.setattr(service, "_load_model", unexpected_load)
    with TestClient(service.app) as client:
        assert client.post("/embed", json=payload).status_code == 422


def test_text_query_returns_a_normalized_vector(service, monkeypatch):
    calls = []

    def encode(text, **kwargs):
        calls.append((text, kwargs))
        return np.ones(768, dtype=np.float32)

    monkeypatch.setattr(service, "_load_model", lambda: SimpleNamespace(encode=encode))
    with TestClient(service.app) as client:
        response = client.post("/embed", json={"text": "  inflación argentina  "})
    assert response.status_code == 200
    vector = np.asarray(response.json()["embedding"])
    assert vector.shape == (768,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert calls[0][0] == "inflación argentina"
    assert calls[0][1]["prompt_name"] == "SearchQuery"
