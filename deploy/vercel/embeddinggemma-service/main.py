"""Text-query embeddings compatible with the locally indexed audio corpus."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = os.getenv("EMBEDDINGGEMMA_MODEL", "google/embeddinggemma-2")
MODEL_REVISION = os.getenv(
    "EMBEDDINGGEMMA_REVISION", "914f7f89142e33e77833254d9c9b90c3cef7303b"
)
MODEL_PATH = Path(os.getenv("EMBEDDINGGEMMA_MODEL_PATH", str(Path(__file__).parent / "model")))
OUTPUT_DIM = int(os.getenv("EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY", "768"))
ALLOWED_DIMS = (768, 512, 256, 128)

if OUTPUT_DIM not in ALLOWED_DIMS:
    raise RuntimeError(f"EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY must be one of {ALLOWED_DIMS}")

app = FastAPI(title="embeddinggemma-2-text-inference")
_model = None
_inference_lock = Lock()


def _load_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(int(os.getenv("EMBEDDINGGEMMA_NUM_THREADS", "1")))
        if not (MODEL_PATH / "modules.json").is_file():
            raise RuntimeError("Bundled text model missing; run prepare_model.py before serving")
        _model = SentenceTransformer(
            str(MODEL_PATH),
            device="cpu",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": torch.float32},
            local_files_only=True,
        )
    return _model


def _finalize(vector) -> list[float]:
    import numpy as np

    emb = np.asarray(vector, dtype=np.float32).reshape(-1)
    if emb.size < OUTPUT_DIM or not np.isfinite(emb).all():
        raise RuntimeError("Model returned an invalid embedding")
    emb = emb[:OUTPUT_DIM]
    norm = np.linalg.norm(emb)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Model returned an invalid embedding norm")
    return (emb / norm).tolist()


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=2048)
    modality: Literal["text"] = "text"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "input_modality": "text",
        "output_dimensionality": OUTPUT_DIM,
        "model_loaded": _model is not None,
    }


@app.post("/embed")
def embed(request: EmbedRequest) -> dict:
    # Avoid duplicate model loads and overlapping inference memory peaks.
    with _inference_lock:
        embedding = _load_model().encode(
            request.text, prompt_name="SearchQuery", show_progress_bar=False
        )
        return {"embedding": _finalize(embedding)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
