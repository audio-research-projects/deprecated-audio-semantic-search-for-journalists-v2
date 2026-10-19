"""Single-process local UI and retrieval over the already processed corpus.

Run from the repository root with the pipeline's .env loaded. Audio queries
are translated to English locally through Ollama before embedding inference.
"""

# ruff: noqa: I001 -- SentenceTransformer must initialize Torch before FAISS.
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

# Load Torch before FAISS on macOS to avoid conflicting OpenMP initialization.
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, Field

from src.agent_service.search_engine import AudioSearchEngine
from src.query_translation import translate_to_english

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.getenv("DATASET_PATH", ROOT / "dataset/debate-embeddinggemma")).resolve()
MODEL = ROOT / "deploy/vercel/embeddinggemma-service/model"
app = FastAPI(title="Local audio dataset search")
_engine = None
_audio_index = None
_gemma = None
_lock = Lock()
LABELS = {
    "text": "Transcripciones",
    "audio": "Audio (EmbeddingGemma 2)",
    "yamnet": "YAMNet/AudioSet",
}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2048)
    k: int = Field(default=5, ge=1, le=50)
    include_text: bool = True
    include_clap: bool = False  # Legacy UI field means the audio index.
    include_yamnet: bool = False
    rewrite: bool = False
    dataset_id: str | None = None
    plan: dict | None = None


def engine():
    global _engine
    if _engine is None:
        _engine = AudioSearchEngine(str(DATASET))
    return _engine


def audio_search(query: str, k: int):
    global _audio_index, _gemma
    if _audio_index is None:
        _audio_index = faiss.read_index(str(DATASET / "indices/embeddinggemma_audio_index.faiss"))
    if _gemma is None:
        if not (MODEL / "modules.json").is_file():
            raise RuntimeError(
                "Preparar el modelo con python deploy/vercel/embeddinggemma-service/prepare_model.py"
            )
        _gemma = SentenceTransformer(
            str(MODEL),
            device="cpu",
            local_files_only=True,
            config_kwargs={"audio_config": None, "vision_config": None},
        )
    vector = np.asarray(_gemma.encode(query, prompt_name="SearchQuery"), dtype=np.float32)[
        : _audio_index.d
    ]
    vector /= np.linalg.norm(vector)
    scores, positions = _audio_index.search(vector.reshape(1, -1), k)
    # FAISS rows follow the dataset order, not necessarily segment_id.
    import pandas as pd

    frame = engine()._df
    assert isinstance(frame, pd.DataFrame)
    return [
        {
            "segment": engine().get_segment_info(int(frame.iloc[int(position)].segment_id)),
            "similarity": float(score),
        }
        for score, position in zip(scores[0], positions[0], strict=True)
        if position >= 0
    ]


def rewrite_plan(query: str, indexes: list[str]) -> dict:
    """Ask the local Ollama model for a search query plan in JSON."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3.8:27b")
    prompt = "\n".join(
        (
            "Reformulá la consulta para buscar en un archivo periodístico de audio.",
            f"Índices habilitados: {json.dumps(indexes)}. No agregues ni quites índices.",
            "Respondé exclusivamente JSON con original_query, indexes, rationale, text_query, audio_query y yamnet_query.",
            "Conservá el idioma original, nombres propios, citas y hechos. Hacé cada consulta concreta y breve, sin inventar detalles.",
            f"Consulta original: {json.dumps(query, ensure_ascii=False)}",
        )
    )
    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=httpx.Timeout(180.0, connect=3.0),
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        plan = json.loads(content)
        if not isinstance(plan, dict):
            raise ValueError("Ollama did not return a JSON object")
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise HTTPException(
            502,
            f"No se pudo reformular con Ollama ({model}). Verificá que Ollama esté iniciado y el modelo descargado.",
        ) from error

    plan["original_query"] = query
    plan["indexes"] = indexes
    for index, key in (("text", "text_query"), ("audio", "audio_query"), ("yamnet", "yamnet_query")):
        if index in indexes and not isinstance(plan.get(key), str):
            plan[key] = query
        if index not in indexes:
            plan.pop(key, None)
    plan.setdefault("rationale", "Consulta reformulada con Ollama.")
    return plan


def translate_local_query(query: str) -> str:
    try:
        return translate_to_english(query, provider="ollama")
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from error


@app.get("/api/datasets")
def datasets():
    with _lock:
        current = engine()
        return {
            "datasets": [
                {
                    "id": "local",
                    "label": "Debate Massa–Milei · 1201 segmentos",
                    "language": "es",
                    "active_indexes": ["text", "audio", "yamnet"],
                    "dataset_version": current.dataset_version,
                }
            ]
        }


@app.get("/api/health")
def health():
    return {"status": "ok", "dataset": str(DATASET)}


@app.post("/api/search")
def search(request: SearchRequest):
    if request.dataset_id not in (None, "local"):
        raise HTTPException(404, "Dataset desconocido")
    query = request.query.strip()
    indexes = [
        name
        for name, enabled in (
            ("text", request.include_text),
            ("audio", request.include_clap),
            ("yamnet", request.include_yamnet),
        )
        if enabled
    ]
    if not query or not indexes:
        raise HTTPException(422, "Indicar consulta y al menos una fuente")
    started = perf_counter()
    buckets = {}
    plan = rewrite_plan(query, indexes) if request.rewrite else {
        "original_query": query,
        "indexes": indexes,
        "rationale": "Búsqueda literal local.",
    }
    with _lock:
        for name in indexes:
            key = {"text": "text_query", "audio": "audio_query", "yamnet": "yamnet_query"}[name]
            effective = (request.plan or plan).get(key) or query
            plan[f"{name}_query"] = effective
            if name == "text":
                results = engine().search_semantic(effective, request.k)
            elif name == "audio":
                translated = translate_local_query(effective)
                plan["audio_query_en"] = translated
                results = audio_search(translated, request.k)
            else:
                # Translate before matching the stored English AudioSet labels.
                translated = translate_local_query(effective)
                plan["yamnet_query_en"] = translated
                results = engine().search_audio_by_classes(
                    translated, request.k, source_language="en"
                )
            normalized = []
            for rank, result in enumerate(results, 1):
                segment = dict(result["segment"])
                clip_name = segment.get("clip_file_name")
                if clip_name:
                    segment["clip_url"] = f"/api/clips/{Path(clip_name).name}"
                normalized.append(
                    {
                        **segment,
                        "rank": rank,
                        "search_index": name,
                        "search_index_label": LABELS[name],
                        "similarity": result["similarity"],
                    }
                )
            buckets[name] = {"available": True, "effective_query": effective, "results": normalized}
            if name in ("audio", "yamnet"):
                buckets[name]["translated_query"] = plan[f"{name}_query_en"]
    return {
        "query": query,
        "dataset_id": "local",
        "dataset_label": "Debate Massa–Milei",
        "plan": plan,
        "took_ms": round((perf_counter() - started) * 1000),
        "indexes": buckets,
    }


app.mount("/api/clips", StaticFiles(directory=DATASET / "segment_clips"), name="clips")
app.mount("/", StaticFiles(directory=ROOT / "deploy/vercel/dist-web", html=True), name="ui")
