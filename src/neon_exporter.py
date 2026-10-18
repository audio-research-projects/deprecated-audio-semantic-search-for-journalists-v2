"""Export a processed audio-search dataset to Neon Postgres (pgvector) and Vercel Blob."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from src.segment_clips import CLIP_DIR_NAME, CLIP_EXTENSION
from src.yamnet_inverted_index import build_yamnet_inverted_index

logger = logging.getLogger(__name__)

VERCEL_BLOB_API_URL = "https://blob.vercel-storage.com"
REQUIRED_DATASET_FILE = Path("final/complete_dataset.pkl")
SCHEMA_SQL_PATH = Path(__file__).resolve().parent.parent / "deploy" / "vercel" / "db" / "schema.sql"


@dataclass(frozen=True)
class PreparedSegmentRow:
    dataset_id: str
    segment_id: int
    text: str
    start_time: float
    end_time: float
    original_file_name: str
    language: str
    confidence: float
    clip_file_name: str | None
    clip_start_time: float | None
    clip_end_time: float | None
    clip_url: str | None
    sentiment_positive: float | None
    sentiment_negative: float | None
    sentiment_neutral: float | None
    dominant_sentiment: str | None
    yamnet_top_classes: list[dict[str, Any]]
    text_embedding: list[float] | None
    audio_embedding: list[float] | None


@dataclass(frozen=True)
class PreparedDatasetBundle:
    dataset_id: str
    label: str
    language: str
    dataset_version: str | None
    active_indexes: list[str]
    is_default: bool
    segments: list[PreparedSegmentRow]
    yamnet_postings: list[dict[str, Any]]


def format_pgvector(values: list[float] | np.ndarray | None) -> str | None:
    """Format a 1D float vector as a pgvector literal ``[v1,v2,...]``."""
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    formatted = ",".join(f"{float(v):.8g}" for v in array)
    return f"[{formatted}]"


def _clean_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _parse_classes(raw: object) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            parsed.append(
                {
                    "class_id": str(item["class_id"]),
                    "class_name": str(item["class_name"]),
                    "score": float(item["score"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


def _extract_vector(
    row: pd.Series,
    column_name: str,
    npy_dir: Path,
    segment_id: int,
) -> list[float] | None:
    if column_name in row and row[column_name] is not None:
        val = row[column_name]
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val, dtype=np.float32).reshape(-1)
            if arr.size > 0 and np.all(np.isfinite(arr)):
                return [float(x) for x in arr]
    npy_file = npy_dir / f"segment_{segment_id}.npy"
    if npy_file.is_file():
        arr = np.load(npy_file).astype(np.float32).reshape(-1)
        if arr.size > 0 and np.all(np.isfinite(arr)):
            return [float(x) for x in arr]
    return None


def upload_clip_to_vercel_blob(
    clip_path: Path,
    *,
    dataset_id: str,
    release: str,
    blob_token: str,
    client: httpx.Client | None = None,
) -> str:
    """Upload a single `.opus` clip to Vercel Blob and return its public URL."""
    pathname = f"audio-search/{dataset_id}/{release}/{CLIP_DIR_NAME}/{clip_path.name}"
    url = f"{VERCEL_BLOB_API_URL}/{pathname}"
    headers = {
        "Authorization": f"Bearer {blob_token}",
        "x-api-version": "7",
        "x-content-type": "audio/ogg",
        "x-add-random-suffix": "0",
    }
    own_client = client is None
    http = client or httpx.Client(timeout=60.0)
    try:
        response = http.put(url, content=clip_path.read_bytes(), headers=headers)
        response.raise_for_status()
        payload = response.json()
        blob_url = payload.get("url")
        if not isinstance(blob_url, str) or not blob_url:
            raise RuntimeError(f"Vercel Blob response missing 'url' for {clip_path.name}")
        return blob_url
    finally:
        if own_client:
            http.close()


def prepare_dataset_bundle(
    dataset_dir: str | Path,
    *,
    dataset_id: str = "default",
    label: str = "Default dataset",
    language: str = "es",
    release: str | None = None,
    is_default: bool = True,
    audio_embedding_source: str = "auto",
    clip_url_resolver: Callable[[int, str], str | None] | None = None,
) -> PreparedDatasetBundle:
    """Read a local processed dataset and prepare rows for Neon Postgres insertion.

    Args:
        dataset_dir: Directory containing `final/complete_dataset.pkl`.
        dataset_id: Dataset identifier in Neon (`datasets.id`).
        label: Human-readable dataset label.
        language: Dataset language code (`es` or `en`).
        release: Optional version string overriding manifest version.
        is_default: Whether to mark this dataset as the default in Neon.
        audio_embedding_source: Which audio vector to store in `audio_embedding`:
            `"auto"` (prefers `gemini`, then `embeddinggemma`, then `clap`),
            `"clap"`, `"gemini"`, or `"embeddinggemma"`.
        clip_url_resolver: Optional callback `(segment_id, clip_file_name) -> url`
            used to populate `clip_url` (e.g. after uploading to Vercel Blob).
    """
    root = Path(dataset_dir).expanduser().resolve()
    pkl_path = root / REQUIRED_DATASET_FILE
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {pkl_path}. Run ingestion first.")

    df = pd.read_pickle(pkl_path)
    manifest_path = root / "final" / "dataset_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset_version = (
        release
        or manifest.get("dataset_version")
        or manifest.get("release")
        or root.name
    )

    embeddings_root = root / "embeddings"
    segments: list[PreparedSegmentRow] = []
    raw_rows_for_yamnet: list[dict[str, Any]] = []

    has_text = False
    has_audio = False

    for _, row in df.iterrows():
        segment_id = int(row.get("segment_id", 0))
        yamnet_classes = _parse_classes(row.get("yamnet_top_classes", []))
        raw_rows_for_yamnet.append(
            {"segment_id": segment_id, "yamnet_top_classes": yamnet_classes}
        )

        text_vec = _extract_vector(
            row, "text_embedding", embeddings_root / "text", segment_id
        )
        gemini_vec = _extract_vector(
            row, "gemini_embedding", embeddings_root / "gemini", segment_id
        )
        embeddinggemma_vec = _extract_vector(
            row,
            "embeddinggemma_embedding",
            embeddings_root / "embeddinggemma",
            segment_id,
        )
        clap_vec = _extract_vector(
            row, "clap_embedding", embeddings_root / "clap", segment_id
        )

        if audio_embedding_source == "gemini":
            audio_vec = gemini_vec
        elif audio_embedding_source == "embeddinggemma":
            audio_vec = embeddinggemma_vec
        elif audio_embedding_source == "clap":
            audio_vec = clap_vec
        else:
            audio_vec = gemini_vec or embeddinggemma_vec or clap_vec

        if text_vec is not None:
            has_text = True
        if audio_vec is not None:
            has_audio = True

        clip_name = str(row.get("clip_file_name", "") or "").strip() or None
        clip_url = (
            clip_url_resolver(segment_id, clip_name)
            if (clip_name and clip_url_resolver is not None)
            else None
        )

        segments.append(
            PreparedSegmentRow(
                dataset_id=dataset_id,
                segment_id=segment_id,
                text=str(row.get("text", "")),
                start_time=float(row.get("start_time", 0.0)),
                end_time=float(row.get("end_time", 0.0)),
                original_file_name=str(row.get("original_file_name", "")),
                language=str(row.get("language", language)),
                confidence=_clean_float(row.get("confidence"), 0.0) or 0.0,
                clip_file_name=clip_name,
                clip_start_time=_clean_float(row.get("clip_start_time")),
                clip_end_time=_clean_float(row.get("clip_end_time")),
                clip_url=clip_url,
                sentiment_positive=_clean_float(row.get("sentiment_positive")),
                sentiment_negative=_clean_float(row.get("sentiment_negative")),
                sentiment_neutral=_clean_float(row.get("sentiment_neutral")),
                dominant_sentiment=(
                    str(row["dominant_sentiment"])
                    if pd.notna(row.get("dominant_sentiment")) and row.get("dominant_sentiment")
                    else None
                ),
                yamnet_top_classes=yamnet_classes,
                text_embedding=text_vec,
                audio_embedding=audio_vec,
            )
        )

    inverted = build_yamnet_inverted_index(raw_rows_for_yamnet)
    postings_map: dict[str, list[dict[str, Any]]] = inverted.get("postings", {})  # type: ignore[assignment]
    flat_postings: list[dict[str, Any]] = []
    for token, items in postings_map.items():
        for item in items:
            flat_postings.append(
                {
                    "dataset_id": dataset_id,
                    "token": token,
                    "segment_id": int(item["segment_id"]),
                    "class_id": str(item["class_id"]),
                    "class_name": str(item["class_name"]),
                    "score": float(item["score"]),
                    "class_rank": int(item["class_rank"]),
                }
            )

    active_indexes: list[str] = []
    if has_text:
        active_indexes.append("text")
    if has_audio:
        active_indexes.append("audio")
    if flat_postings:
        active_indexes.append("yamnet")

    return PreparedDatasetBundle(
        dataset_id=dataset_id,
        label=label,
        language=language,
        dataset_version=str(dataset_version) if dataset_version else None,
        active_indexes=active_indexes,
        is_default=is_default,
        segments=segments,
        yamnet_postings=flat_postings,
    )


def publish_bundle_to_neon(
    bundle: PreparedDatasetBundle,
    database_url: str | None = None,
) -> None:
    """Write a `PreparedDatasetBundle` into Neon Postgres using `psycopg`."""
    resolved_url = (database_url or os.getenv("DATABASE_URL", "")).strip()
    if not resolved_url:
        raise ValueError("DATABASE_URL is required to publish to Neon Postgres.")

    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError(
            "psycopg is required to publish to Neon. Run with: "
            "uv run --with 'psycopg[binary]' python scripts/publish_dataset_neon.py ..."
        ) from error

    schema_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(resolved_url) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            if bundle.is_default:
                cur.execute("UPDATE datasets SET is_default = FALSE WHERE id <> %s", (bundle.dataset_id,))

            cur.execute(
                """
                INSERT INTO datasets (id, label, language, dataset_version, active_indexes, total_segments, is_default, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    label = EXCLUDED.label,
                    language = EXCLUDED.language,
                    dataset_version = EXCLUDED.dataset_version,
                    active_indexes = EXCLUDED.active_indexes,
                    total_segments = EXCLUDED.total_segments,
                    is_default = EXCLUDED.is_default,
                    updated_at = NOW()
                """,
                (
                    bundle.dataset_id,
                    bundle.label,
                    bundle.language,
                    bundle.dataset_version,
                    json.dumps(bundle.active_indexes),
                    len(bundle.segments),
                    bundle.is_default,
                ),
            )

            cur.execute("DELETE FROM segment_yamnet_postings WHERE dataset_id = %s", (bundle.dataset_id,))
            cur.execute("DELETE FROM audio_segments WHERE dataset_id = %s", (bundle.dataset_id,))

            segment_params = [
                (
                    seg.dataset_id,
                    seg.segment_id,
                    seg.text,
                    seg.start_time,
                    seg.end_time,
                    seg.original_file_name,
                    seg.language,
                    seg.confidence,
                    seg.clip_file_name,
                    seg.clip_start_time,
                    seg.clip_end_time,
                    seg.clip_url,
                    seg.sentiment_positive,
                    seg.sentiment_negative,
                    seg.sentiment_neutral,
                    seg.dominant_sentiment,
                    json.dumps(seg.yamnet_top_classes, ensure_ascii=False),
                    format_pgvector(seg.text_embedding),
                    format_pgvector(seg.audio_embedding),
                )
                for seg in bundle.segments
            ]

            cur.executemany(
                """
                INSERT INTO audio_segments (
                    dataset_id, segment_id, text, start_time, end_time,
                    original_file_name, language, confidence,
                    clip_file_name, clip_start_time, clip_end_time, clip_url,
                    sentiment_positive, sentiment_negative, sentiment_neutral, dominant_sentiment,
                    yamnet_top_classes, text_embedding, audio_embedding
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s::jsonb, %s::vector, %s::vector
                )
                """,
                segment_params,
            )

            if bundle.yamnet_postings:
                posting_params = [
                    (
                        p["dataset_id"],
                        p["token"],
                        p["segment_id"],
                        p["class_id"],
                        p["class_name"],
                        p["score"],
                        p["class_rank"],
                    )
                    for p in bundle.yamnet_postings
                ]
                cur.executemany(
                    """
                    INSERT INTO segment_yamnet_postings (
                        dataset_id, token, segment_id, class_id, class_name, score, class_rank
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_id, token, segment_id, class_id) DO UPDATE SET
                        score = EXCLUDED.score,
                        class_rank = EXCLUDED.class_rank
                    """,
                    posting_params,
                )
        conn.commit()
    logger.info(
        "Published dataset '%s' to Neon: %d segments, %d YAMNet postings",
        bundle.dataset_id,
        len(bundle.segments),
        len(bundle.yamnet_postings),
    )


__all__ = [
    "CLIP_EXTENSION",
    "PreparedDatasetBundle",
    "PreparedSegmentRow",
    "format_pgvector",
    "prepare_dataset_bundle",
    "publish_bundle_to_neon",
    "upload_clip_to_vercel_blob",
]
