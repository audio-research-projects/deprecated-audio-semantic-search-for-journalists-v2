"""Reference EmbeddingGemma 2 inference service for Vercel query-side search.

Exposes ``POST /embed`` with the same contract expected by
``deploy/vercel/src/embeddings.ts`` (provider ``embeddinggemma`` / ``custom``):

    POST /embed   {"text": "...", "modality": "text" | "audio"}
    →             {"embedding": [float, ...]}   # L2-normalized, MRL-truncated

For ``modality="audio"`` the caller may pass ``audio_url`` instead of ``text``;
the service embeds it with EmbeddingGemma 2's audio encoder. Text queries use
the ``SearchQuery`` prompt so they land in the same space as the audio windows
ingested by ``src/embeddinggemma_embeddings.py``.

Deploy targets: Cloud Run (see Dockerfile) or any GPU/CPU host. Configure the
Vercel project with ``EMBEDDINGGEMMA_API_URL=<service base URL>``.
"""

from __future__ import annotations

import os

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_ID = os.getenv("EMBEDDINGGEMMA_MODEL", "google/embeddinggemma-2")
OUTPUT_DIM = int(os.getenv("EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY", "768"))
ALLOWED_DIMS = (768, 512, 256, 128)

if OUTPUT_DIM not in ALLOWED_DIMS:
    raise RuntimeError(f"EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY must be one of {ALLOWED_DIMS}")

app = FastAPI(title="embeddinggemma-2-inference")
_model = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_ID, config_kwargs={"vision_config": None})
    return _model


def _finalize(vector) -> list[float]:
    import numpy as np

    emb = np.asarray(vector, dtype=np.float32).reshape(-1)[:OUTPUT_DIM]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


class EmbedRequest(BaseModel):
    text: str | None = None
    modality: str = "text"
    audio_url: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "output_dimensionality": OUTPUT_DIM}


@app.post("/embed")
def embed(request: EmbedRequest) -> dict:
    model = _load_model()
    if request.modality == "audio":
        if not request.audio_url:
            raise HTTPException(400, "modality=audio requires audio_url")
        try:
            response = httpx.get(request.audio_url, timeout=60.0)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(502, f"audio download failed: {error}") from error
        import tempfile

        # sentence-transformers accepts file paths/URLs; write bytes to a temp
        # file so the decoder can sniff the format.
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        embedding = model.encode({"audio": tmp_path})
    else:
        if not (request.text or "").strip():
            raise HTTPException(400, "text is required for modality=text")
        embedding = model.encode(request.text, prompt_name="SearchQuery")
    return {"embedding": _finalize(embedding)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
