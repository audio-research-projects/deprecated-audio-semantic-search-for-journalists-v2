"""Unit tests for the Neon Postgres + Vercel Blob dataset exporter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.neon_exporter import (
    SCHEMA_SQL_PATH,
    format_pgvector,
    prepare_dataset_bundle,
    upload_clip_to_vercel_blob,
)


def _create_fake_dataset(
    root: Path, *, include_gemini: bool = True, include_embeddinggemma: bool = False
) -> Path:
    final_dir = root / "final"
    final_dir.mkdir(parents=True)
    clips_dir = root / "segment_clips"
    clips_dir.mkdir(parents=True)

    text_vec_0 = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    text_vec_1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    clap_vec_0 = np.array([1.0, 0.0], dtype=np.float32)
    clap_vec_1 = np.array([0.0, 1.0], dtype=np.float32)
    gemini_vec_0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    gemini_vec_1 = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32)
    embeddinggemma_vec_0 = np.array([0.4, 0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    embeddinggemma_vec_1 = np.array([-0.4, 0.4, -0.4, 0.4, -0.4], dtype=np.float32)

    rows = [
        {
            "segment_id": 0,
            "text": "Discurso inaugural en el Congreso",
            "start_time": 0.0,
            "end_time": 12.5,
            "original_file_name": "sesion_01.wav",
            "language": "es",
            "confidence": 0.94,
            "clip_file_name": "segment_0.opus",
            "clip_start_time": 0.0,
            "clip_end_time": 17.5,
            "sentiment_positive": 0.7,
            "sentiment_negative": 0.1,
            "sentiment_neutral": 0.2,
            "dominant_sentiment": "POS",
            "yamnet_top_classes": [
                {"class_id": "/m/09x0r", "class_name": "Speech", "score": 0.91},
                {"class_id": "/m/028ght", "class_name": "Applause", "score": 0.65},
            ],
            "text_embedding": text_vec_0,
            "clap_embedding": clap_vec_0,
            **({"gemini_embedding": gemini_vec_0} if include_gemini else {}),
            **(
                {"embeddinggemma_embedding": embeddinggemma_vec_0}
                if include_embeddinggemma
                else {}
            ),
        },
        {
            "segment_id": 1,
            "text": "Música de cierre",
            "start_time": 12.5,
            "end_time": 20.0,
            "original_file_name": "sesion_01.wav",
            "language": "es",
            "confidence": 0.88,
            "clip_file_name": "segment_1.opus",
            "clip_start_time": 7.5,
            "clip_end_time": 20.0,
            "sentiment_positive": 0.3,
            "sentiment_negative": 0.1,
            "sentiment_neutral": 0.6,
            "dominant_sentiment": "NEU",
            "yamnet_top_classes": [
                {"class_id": "/m/04rlf", "class_name": "Music", "score": 0.84},
            ],
            "text_embedding": text_vec_1,
            "clap_embedding": clap_vec_1,
            **({"gemini_embedding": gemini_vec_1} if include_gemini else {}),
            **(
                {"embeddinggemma_embedding": embeddinggemma_vec_1}
                if include_embeddinggemma
                else {}
            ),
        },
    ]

    df = pd.DataFrame(rows)
    df.to_pickle(final_dir / "complete_dataset.pkl")
    (final_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "2026-10-16-test",
                "active_embeddings": [
                    "text",
                    "clap",
                    *(["gemini"] if include_gemini else []),
                    *(["embeddinggemma"] if include_embeddinggemma else []),
                ],
                "active_classifiers": ["yamnet"],
                "total_segments": 2,
            }
        ),
        encoding="utf-8",
    )
    (clips_dir / "segment_0.opus").write_bytes(b"OggS-fake-opus-0")
    (clips_dir / "segment_1.opus").write_bytes(b"OggS-fake-opus-1")
    return root


def test_format_pgvector_formats_1d_arrays_and_handles_none():
    assert format_pgvector(None) is None
    assert format_pgvector([]) is None
    literal = format_pgvector([0.5, -0.25, 1.0])
    assert literal == "[0.5,-0.25,1]"


def test_schema_sql_exists_and_declares_required_tables():
    assert SCHEMA_SQL_PATH.is_file()
    sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS datasets" in sql
    assert "CREATE TABLE IF NOT EXISTS audio_segments" in sql
    assert "CREATE TABLE IF NOT EXISTS segment_yamnet_postings" in sql
    assert "CREATE TABLE IF NOT EXISTS feedback_logs" in sql


def test_prepare_dataset_bundle_extracts_segments_vectors_and_yamnet_postings(tmp_path: Path):
    dataset_dir = _create_fake_dataset(tmp_path / "dataset", include_gemini=True)

    bundle = prepare_dataset_bundle(
        dataset_dir,
        dataset_id="archivo-2026",
        label="Archivo 2026",
        language="es",
        audio_embedding_source="auto",
        clip_url_resolver=lambda _seg_id, name: f"https://blob.vercel-storage.com/clips/{name}",
    )

    assert bundle.dataset_id == "archivo-2026"
    assert bundle.dataset_version == "2026-10-16-test"
    assert bundle.active_indexes == ["text", "audio", "yamnet"]
    assert len(bundle.segments) == 2

    seg0 = bundle.segments[0]
    assert seg0.segment_id == 0
    assert seg0.clip_url == "https://blob.vercel-storage.com/clips/segment_0.opus"
    assert seg0.text_embedding is not None and len(seg0.text_embedding) == 3
    # "auto" prefers gemini (4d) when available over clap (2d)
    assert seg0.audio_embedding is not None and len(seg0.audio_embedding) == 4

    tokens = {posting["token"] for posting in bundle.yamnet_postings}
    assert {"speech", "applause", "music"} <= tokens


def test_prepare_dataset_bundle_supports_clap_selection(tmp_path: Path):
    dataset_dir = _create_fake_dataset(tmp_path / "dataset", include_gemini=True)
    bundle = prepare_dataset_bundle(
        dataset_dir,
        dataset_id="default",
        audio_embedding_source="clap",
    )
    assert bundle.segments[0].audio_embedding is not None
    assert len(bundle.segments[0].audio_embedding) == 2


def test_prepare_dataset_bundle_supports_embeddinggemma_selection(tmp_path: Path):
    dataset_dir = _create_fake_dataset(
        tmp_path / "dataset", include_gemini=True, include_embeddinggemma=True
    )
    bundle = prepare_dataset_bundle(
        dataset_dir,
        dataset_id="default",
        audio_embedding_source="embeddinggemma",
    )
    assert bundle.segments[0].audio_embedding is not None
    assert len(bundle.segments[0].audio_embedding) == 5


def test_prepare_dataset_bundle_auto_prefers_embeddinggemma_over_clap(tmp_path: Path):
    dataset_dir = _create_fake_dataset(
        tmp_path / "dataset", include_gemini=False, include_embeddinggemma=True
    )
    bundle = prepare_dataset_bundle(
        dataset_dir,
        dataset_id="default",
        audio_embedding_source="auto",
    )
    assert bundle.segments[0].audio_embedding is not None
    assert len(bundle.segments[0].audio_embedding) == 5


def test_upload_clip_to_vercel_blob_sends_put_request(tmp_path: Path):
    clip_file = tmp_path / "segment_0.opus"
    clip_file.write_bytes(b"OggS-audio-bytes")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "url": "https://example.public.blob.vercel-storage.com/audio-search/default/rel1/segment_clips/segment_0.opus"
    }
    mock_client = MagicMock()
    mock_client.put.return_value = mock_response

    url = upload_clip_to_vercel_blob(
        clip_file,
        dataset_id="default",
        release="rel1",
        blob_token="vercel_blob_rw_secret",
        client=mock_client,
    )

    assert url.endswith("/segment_0.opus")
    mock_client.put.assert_called_once()
    call_args = mock_client.put.call_args
    assert "audio-search/default/rel1/segment_clips/segment_0.opus" in call_args.args[0]
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer vercel_blob_rw_secret"


def test_prepare_dataset_bundle_raises_when_missing_pickle(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        prepare_dataset_bundle(tmp_path / "nonexistent")
