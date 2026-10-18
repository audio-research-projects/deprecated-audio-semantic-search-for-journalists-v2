"""Pipeline de ingesta: audio → dataset indexado para búsqueda semántica."""

import argparse
import hashlib
import json
import logging
import os
import shlex
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.audio_conversion import SUPPORTED_FORMATS, convert_audio
from src.audio_segmenting import build_audio_windows, extract_wav_window
from src.audio_transcription import ChunkingConfig, ChunkingProcessor, transcribe_files
from src.clap_audio_embeddings import CLAPConfig, CLAPEmbedding
from src.embedding_config import load_embedding_config, write_embedding_config_from_env
from src.embeddinggemma_embeddings import (
    EmbeddingGemmaConfig,
    EmbeddingGemmaMultimodalEmbedding,
)
from src.gemini_multimodal_embeddings import GeminiEmbeddingConfig, GeminiMultimodalEmbedding
from src.segment_clips import (
    CLIP_DIR_NAME,
    CLIP_EXTENSION,
    DEFAULT_CLIP_BITRATE,
    DEFAULT_CLIP_CONTEXT_SEC,
    build_clip_bounds,
    clip_file_name,
    export_segment_clip,
    prune_orphan_clips,
)
from src.sentiment_analysis import SentimentAnalyzer
from src.text_embeddings import TextEmbeddingModel
from src.vector_indexing import build_faiss_index
from src.yamnet_audio_classifier import (
    YAMNetAudioClassifier,
    YAMNetConfig,
    aggregate_yamnet_classes,
)
from src.yamnet_inverted_index import (
    YAMNET_INVERTED_INDEX_FILENAME,
    write_yamnet_inverted_index,
)

logger = logging.getLogger(__name__)


def _pool_audio_embeddings(window_embeddings: list[np.ndarray], dimension: int) -> np.ndarray:
    if not window_embeddings:
        return np.zeros(dimension, dtype=np.float32)
    pooled = np.mean(window_embeddings, axis=0)
    norm = np.linalg.norm(pooled)
    return (pooled / norm if norm > 0 else pooled).astype(np.float32)


def _write_process_run_log(
    output_path: Path,
    *,
    parameters: dict,
    embedding_config,
    command_argv: list[str] | None,
) -> None:
    """Persist the effective ingestion invocation next to its output dataset."""
    recorded_argv = command_argv or []
    log = {
        "version": "1.0",
        "completed_at": datetime.now(UTC).isoformat(),
        "working_directory": str(Path.cwd()),
        "command_argv": recorded_argv,
        "command_display": shlex.join(recorded_argv) if recorded_argv else None,
        "parameters": parameters,
        "embedding_configuration": {
            "active_embeddings": sorted(embedding_config.active_embeddings),
            "active_classifiers": sorted(embedding_config.active_classifiers),
            "text_model": embedding_config.text_model,
            "clap_model": embedding_config.clap_model,
            "gemini_model": embedding_config.gemini_model,
            "gemini_output_dimensionality": embedding_config.gemini_output_dimensionality,
            "embeddinggemma_model": embedding_config.embeddinggemma_model,
            "embeddinggemma_output_dimensionality": embedding_config.embeddinggemma_output_dimensionality,
            "yamnet_model": embedding_config.yamnet_model,
            "yamnet_top_k": embedding_config.yamnet_top_k,
        },
    }
    output_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")


def _file_fingerprint(path: Path) -> dict[str, int | str]:
    """Return a content fingerprint so same-name replacements are detected."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _pipeline_signature(
    *,
    whisper_model: str,
    language: str | None,
    chunk_strategy: str,
    chunk_duration_sec: float,
    chunk_overlap_sec: float,
    max_chunk_text_chars: int,
    audio_window_duration_sec: float,
    audio_window_overlap_sec: float,
    mock_audio: bool,
    embedding_config,
    segment_clips: bool = True,
    segment_clip_context_sec: float = DEFAULT_CLIP_CONTEXT_SEC,
    segment_clip_bitrate: str = DEFAULT_CLIP_BITRATE,
) -> dict:
    """Settings whose changes invalidate previously generated file artifacts."""
    return {
        "whisper_model": whisper_model,
        "language": language,
        "chunking": {
            "strategy": chunk_strategy,
            "duration_sec": chunk_duration_sec,
            "overlap_sec": chunk_overlap_sec,
            "max_text_chars": max_chunk_text_chars,
        },
        "audio_windows": {
            "duration_sec": audio_window_duration_sec,
            "overlap_sec": audio_window_overlap_sec,
        },
        "segment_clips": {
            "enabled": segment_clips,
            "context_sec": segment_clip_context_sec,
            "bitrate": segment_clip_bitrate,
        },
        "mock_audio": mock_audio,
        "embeddings": {
            "active": sorted(embedding_config.active_embeddings),
            "text_model": embedding_config.text_model,
            "clap_model": embedding_config.clap_model,
            "gemini_model": embedding_config.gemini_model,
            "gemini_output_dimensionality": embedding_config.gemini_output_dimensionality,
            "embeddinggemma_model": embedding_config.embeddinggemma_model,
            "embeddinggemma_output_dimensionality": embedding_config.embeddinggemma_output_dimensionality,
        },
        "classifiers": {
            "active": sorted(embedding_config.active_classifiers),
            "yamnet_model": embedding_config.yamnet_model,
            "yamnet_top_k": embedding_config.yamnet_top_k,
        },
    }


def _segment_clips_manifest(
    clips_dir: Path, *, enabled: bool, context_sec: float, bitrate: str
) -> dict:
    """Describe the playback-clip artifact so consumers can locate it in a release."""
    clips = sorted(clips_dir.glob(f"segment_*{CLIP_EXTENSION}")) if clips_dir.is_dir() else []
    return {
        "enabled": enabled,
        "directory": CLIP_DIR_NAME,
        "format": "opus",
        "codec": "libopus",
        "bitrate": bitrate,
        "context_sec": context_sec,
        "naming": f"segment_{{segment_id}}{CLIP_EXTENSION}",
        "count": len(clips),
        "total_bytes": sum(clip.stat().st_size for clip in clips),
        "served": "on-demand",
    }


def _load_incremental_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    input_dir: str,
    output_dir: str,
    whisper_model: str = "base",
    batch_size: int = 8,
    language: str | None = None,
    chunk_strategy: str = "whisper",
    chunk_duration_sec: float = 30.0,
    chunk_overlap_sec: float = 5.0,
    max_chunk_text_chars: int = 500,
    embeddings_config_path: str = "config/embeddings.toml",
    audio_window_duration_sec: float = 10.0,
    audio_window_overlap_sec: float = 2.0,
    segment_clips: bool = True,
    segment_clip_context_sec: float = DEFAULT_CLIP_CONTEXT_SEC,
    segment_clip_bitrate: str = DEFAULT_CLIP_BITRATE,
    mock_audio: bool = False,
    verbose: bool = False,
    command_argv: list[str] | None = None,
):
    """
    Ejecuta el pipeline completo de ingesta.

    Etapas:
    1. Conversión de audio a WAV 16kHz mono
    2. Transcripción con Whisper
    3. Generación de embeddings de texto
    4. Generación de embeddings de audio (CLAP)
    5. Clasificación opcional de eventos acústicos con YAMNet
    6. Clips de reproducción por segmento (Opus)
    7. Análisis de sentimiento
    8. Indexación vectorial FAISS
    9. Serialización del dataset final
    """
    if audio_window_duration_sec <= 0:
        raise ValueError("audio_window_duration_sec must be greater than zero")
    if not 0 <= audio_window_overlap_sec < audio_window_duration_sec:
        raise ValueError(
            "audio_window_overlap_sec must be non-negative and smaller than "
            "audio_window_duration_sec"
        )

    setup_logging(verbose)
    embedding_config = load_embedding_config(embeddings_config_path)
    logger.info(
        "Active embeddings: %s | classifiers: %s",
        ", ".join(sorted(embedding_config.active_embeddings)),
        ", ".join(sorted(embedding_config.active_classifiers)) or "none",
    )

    output = Path(output_dir)
    converted_dir = output / "converted"
    transcriptions_dir = output / "transcriptions"
    embeddings_dir = output / "embeddings"
    audio_segments_dir = output / "audio_segments"
    segment_clips_dir = output / CLIP_DIR_NAME
    indices_dir = output / "indices"
    final_dir = output / "final"

    for d in [
        converted_dir,
        transcriptions_dir,
        embeddings_dir,
        audio_segments_dir,
        segment_clips_dir,
        indices_dir,
        final_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    state_path = final_dir / "ingestion_state.json"
    signature = _pipeline_signature(
        whisper_model=whisper_model,
        language=language,
        chunk_strategy=chunk_strategy,
        chunk_duration_sec=chunk_duration_sec,
        chunk_overlap_sec=chunk_overlap_sec,
        max_chunk_text_chars=max_chunk_text_chars,
        audio_window_duration_sec=audio_window_duration_sec,
        audio_window_overlap_sec=audio_window_overlap_sec,
        mock_audio=mock_audio,
        embedding_config=embedding_config,
        segment_clips=segment_clips,
        segment_clip_context_sec=segment_clip_context_sec,
        segment_clip_bitrate=segment_clip_bitrate,
    )
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    source_files = [
        path for path in sorted(input_path.iterdir()) if path.suffix.lower() in SUPPORTED_FORMATS
    ]
    if not source_files:
        raise ValueError(f"No supported audio files found in {input_dir}")
    source_paths = {f"{path.stem}.wav": path for path in source_files}
    if len(source_paths) != len(source_files):
        raise ValueError("Audio files with the same basename cannot be processed together")
    source_fingerprints = {name: _file_fingerprint(path) for name, path in source_paths.items()}

    previous_state = _load_incremental_state(state_path)
    previous_dataset_path = final_dir / "complete_dataset.pkl"
    compatible = (
        previous_dataset_path.exists() and previous_state.get("pipeline_signature") == signature
    )
    stored_sources = previous_state.get("sources", {})
    previous_sources = stored_sources if compatible else {}
    changed_names = {
        name
        for name, fingerprint in source_fingerprints.items()
        if previous_sources.get(name) != fingerprint or not (converted_dir / name).exists()
    }
    removed_names = set(stored_sources) - set(source_fingerprints)
    if not compatible:
        changed_names = set(source_fingerprints)
        logger.info(
            "Incremental cache invalidated by missing state or changed processing configuration"
        )
        for embedding_dir in embeddings_dir.iterdir():
            if embedding_dir.is_dir():
                for artifact in embedding_dir.glob("segment_*.npy"):
                    artifact.unlink()
        for artifact in audio_segments_dir.glob("segment_*_window_*.wav"):
            artifact.unlink()
        prune_orphan_clips(segment_clips_dir, valid_segment_ids=set())

    previous_df = pd.read_pickle(previous_dataset_path) if compatible else pd.DataFrame()
    unchanged_df = (
        previous_df[~previous_df["original_file_name"].isin(changed_names | removed_names)].copy()
        if not previous_df.empty
        else pd.DataFrame()
    )
    logger.info(
        "Incremental ingestion: %d unchanged, %d new or modified, %d removed",
        len(source_fingerprints) - len(changed_names),
        len(changed_names),
        len(removed_names),
    )

    # --- Etapas 1 y 2: convertir y transcribir sólo archivos nuevos o modificados ---
    logger.info("=" * 60)
    logger.info("ETAPAS 1-2: Conversión y transcripción incremental")
    logger.info("=" * 60)
    changed_wavs = []
    for name in sorted(changed_names):
        changed_wavs.append(
            Path(convert_audio(str(source_paths[name]), str(converted_dir), force=True))
        )
    raw_segments = transcribe_files(changed_wavs, model_name=whisper_model, language=language)
    new_segments = ChunkingProcessor(
        ChunkingConfig(
            strategy=chunk_strategy,
            duration_sec=chunk_duration_sec,
            overlap_sec=chunk_overlap_sec,
            max_text_chars=max_chunk_text_chars,
        )
    ).process_segments(raw_segments)
    next_segment_id = 0 if unchanged_df.empty else int(unchanged_df["segment_id"].max()) + 1
    for offset, segment in enumerate(new_segments):
        segment.segment_id = next_segment_id + offset
    new_df = pd.DataFrame([asdict(segment) for segment in new_segments])
    if new_df.empty:
        new_df = pd.DataFrame(
            columns=[
                "segment_id",
                "text",
                "start_time",
                "end_time",
                "language",
                "confidence",
                "original_file_name",
            ]
        )
    logger.info("New segments after %s chunking: %d", chunk_strategy, len(new_df))

    texts = new_df["text"].tolist()
    embeddings: dict[str, np.ndarray] = {}
    if len(new_df) and embedding_config.is_embedding_active("text"):
        logger.info("Generating text embeddings (%s)", embedding_config.text_model)
        text_model = TextEmbeddingModel(embedding_config.text_model)
        embeddings["text"] = text_model.generate_embeddings(texts, batch_size=batch_size)

    audio_windows: dict[int, list[Path]] = {}
    if len(new_df) and (
        embedding_config.active_embeddings & {"clap", "gemini", "embeddinggemma"}
        or embedding_config.is_classifier_active("yamnet")
    ):
        for _, row in new_df.iterrows():
            source_path = converted_dir / row["original_file_name"]
            if not source_path.exists():
                audio_windows[int(row["segment_id"])] = []
                continue
            import soundfile as sf

            windows = build_audio_windows(
                row["start_time"],
                row["end_time"],
                sf.info(source_path).duration,
                window_duration_sec=audio_window_duration_sec,
                overlap_sec=audio_window_overlap_sec,
            )
            paths = []
            for window_number, (window_start, window_end) in enumerate(windows):
                clip_path = (
                    audio_segments_dir / f"segment_{row['segment_id']}_window_{window_number}.wav"
                )
                extract_wav_window(source_path, clip_path, window_start, window_end)
                paths.append(clip_path)
            audio_windows[int(row["segment_id"])] = paths
        new_df["audio_window_count"] = [
            len(audio_windows[int(segment_id)]) for segment_id in new_df["segment_id"]
        ]

    if len(new_df) and embedding_config.is_embedding_active("clap") and mock_audio:
        logger.warning("Using MOCK audio embeddings (testing mode)")
        clap_embeddings = np.random.randn(len(new_df), 512).astype(np.float32)
        clap_embeddings /= np.linalg.norm(clap_embeddings, axis=1, keepdims=True)
    elif len(new_df) and embedding_config.is_embedding_active("clap"):
        logger.info("Generating CLAP audio embeddings (%s)", embedding_config.clap_model)
        clap = CLAPEmbedding(CLAPConfig(model_name=embedding_config.clap_model))
        clap_embeddings = []
        for _, row in new_df.iterrows():
            window_embeddings = [
                clap.generate_embedding(str(path)) for path in audio_windows[int(row["segment_id"])]
            ]
            clap_embeddings.append(_pool_audio_embeddings(window_embeddings, 512))
        clap_embeddings = np.asarray(clap_embeddings, dtype=np.float32)
    if len(new_df) and embedding_config.is_embedding_active("clap"):
        embeddings["clap"] = clap_embeddings

    if len(new_df) and embedding_config.is_embedding_active("gemini"):
        logger.info("Generating Gemini native audio embeddings (%s)", embedding_config.gemini_model)
        gemini = GeminiMultimodalEmbedding(
            GeminiEmbeddingConfig(
                model_name=embedding_config.gemini_model,
                output_dimensionality=embedding_config.gemini_output_dimensionality,
            )
        )
        gemini_embeddings = []
        for _, row in new_df.iterrows():
            window_embeddings = [
                gemini.generate_audio_embedding(path)
                for path in audio_windows[int(row["segment_id"])]
            ]
            gemini_embeddings.append(
                _pool_audio_embeddings(
                    window_embeddings, embedding_config.gemini_output_dimensionality
                )
            )
        embeddings["gemini"] = np.asarray(gemini_embeddings, dtype=np.float32)

    if len(new_df) and embedding_config.is_embedding_active("embeddinggemma"):
        logger.info(
            "Generating EmbeddingGemma 2 audio embeddings (%s)",
            embedding_config.embeddinggemma_model,
        )
        embeddinggemma = EmbeddingGemmaMultimodalEmbedding(
            EmbeddingGemmaConfig(
                model_name=embedding_config.embeddinggemma_model,
                output_dimensionality=embedding_config.embeddinggemma_output_dimensionality,
            )
        )
        embeddinggemma_embeddings = []
        for _, row in new_df.iterrows():
            window_embeddings = [
                embeddinggemma.generate_audio_embedding(path)
                for path in audio_windows[int(row["segment_id"])]
            ]
            embeddinggemma_embeddings.append(
                _pool_audio_embeddings(
                    window_embeddings,
                    embedding_config.embeddinggemma_output_dimensionality,
                )
            )
        embeddings["embeddinggemma"] = np.asarray(
            embeddinggemma_embeddings, dtype=np.float32
        )

    if len(new_df) and embedding_config.is_classifier_active("yamnet"):
        logger.info("Classifying acoustic events with YAMNet (%s)", embedding_config.yamnet_model)
        yamnet = YAMNetAudioClassifier(
            YAMNetConfig(
                model_url=embedding_config.yamnet_model,
                top_k=embedding_config.yamnet_top_k,
            )
        )
        new_df["yamnet_top_classes"] = [
            aggregate_yamnet_classes(
                [yamnet.classify(path) for path in audio_windows[int(segment_id)]],
                top_k=embedding_config.yamnet_top_k,
            )
            for segment_id in new_df["segment_id"]
        ]

    for name, values in embeddings.items():
        embedding_dir = embeddings_dir / name
        embedding_dir.mkdir(exist_ok=True)
        for segment_id, embedding in zip(new_df["segment_id"], values, strict=True):
            np.save(embedding_dir / f"segment_{segment_id}.npy", embedding)
        new_df[f"{name}_embedding"] = list(values)

    # --- Etapa 6: Clips de reproducción por segmento ---
    logger.info("=" * 60)
    logger.info("ETAPA 6: Clips de reproducción por segmento")
    logger.info("=" * 60)
    if len(new_df) and segment_clips:
        import soundfile as sf

        source_durations: dict[str, float] = {}
        clip_names, clip_starts, clip_ends = [], [], []
        for _, row in new_df.iterrows():
            source_name = row["original_file_name"]
            source_path = converted_dir / source_name
            if not source_path.exists():
                logger.warning(
                    "No converted audio for %s; segment %s gets no clip",
                    source_name,
                    row["segment_id"],
                )
                clip_names.append("")
                clip_starts.append(float(row["start_time"]))
                clip_ends.append(float(row["end_time"]))
                continue
            if source_name not in source_durations:
                source_durations[source_name] = float(sf.info(source_path).duration)
            clip_start, clip_end = build_clip_bounds(
                row["start_time"],
                row["end_time"],
                source_durations[source_name],
                context_sec=segment_clip_context_sec,
            )
            name = clip_file_name(int(row["segment_id"]))
            export_segment_clip(
                source_path,
                segment_clips_dir / name,
                clip_start,
                clip_end,
                bitrate=segment_clip_bitrate,
            )
            clip_names.append(name)
            clip_starts.append(clip_start)
            clip_ends.append(clip_end)
        new_df["clip_file_name"] = clip_names
        new_df["clip_start_time"] = clip_starts
        new_df["clip_end_time"] = clip_ends
        logger.info(
            "Exported %d playback clips (±%.1fs context, %s) to %s",
            sum(1 for name in clip_names if name),
            segment_clip_context_sec,
            segment_clip_bitrate,
            segment_clips_dir,
        )
    elif not segment_clips:
        logger.info("Playback clip generation disabled (--no-segment-clips)")

    # --- Etapa 7: Análisis de sentimiento ---
    logger.info("=" * 60)
    logger.info("ETAPA 7: Análisis de sentimiento")
    logger.info("=" * 60)
    if len(new_df):
        sentiment_analyzer = SentimentAnalyzer()
        sentiments = sentiment_analyzer.analyze_batch(texts, batch_size=batch_size)
        new_df["sentiment_positive"] = [s.positive for s in sentiments]
        new_df["sentiment_negative"] = [s.negative for s in sentiments]
        new_df["sentiment_neutral"] = [s.neutral for s in sentiments]
        new_df["dominant_sentiment"] = [s.dominant for s in sentiments]
    logger.info("Sentiment analysis complete")

    df = (
        pd.concat([unchanged_df, new_df], ignore_index=True, sort=False)
        .sort_values("segment_id")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No transcription segments remain after processing the source audio")
    csv_path = transcriptions_dir / "segments_metadata.csv"
    df[[column for column in df.columns if "embedding" not in column]].to_csv(csv_path, index=False)
    logger.info("Saved transcription metadata to %s", csv_path)

    # --- Etapa 8: Indexación vectorial ---
    logger.info("=" * 60)
    logger.info("ETAPA 8: Indexación FAISS")
    logger.info("=" * 60)
    index_files = {
        "text": "text_index.faiss",
        "clap": "audio_index.faiss",
        "gemini": "gemini_audio_index.faiss",
        "embeddinggemma": "embeddinggemma_audio_index.faiss",
    }
    yamnet_index_path = indices_dir / YAMNET_INVERTED_INDEX_FILENAME
    yamnet_index_token_count = 0
    if embedding_config.is_classifier_active("yamnet"):
        if "yamnet_top_classes" not in df:
            df["yamnet_top_classes"] = [[] for _ in range(len(df))]
        yamnet_index_token_count = write_yamnet_inverted_index(
            yamnet_index_path,
            df[["segment_id", "yamnet_top_classes"]].to_dict("records"),
        )
        logger.info(
            "Saved YAMNet inverted index: %d tokens at %s",
            yamnet_index_token_count,
            yamnet_index_path,
        )
    elif yamnet_index_path.exists():
        yamnet_index_path.unlink()
        logger.info("Removed stale disabled YAMNet inverted index: %s", yamnet_index_path)
    all_embeddings = {
        name: np.asarray(df[f"{name}_embedding"].tolist(), dtype=np.float32)
        for name in index_files
        if name in embedding_config.active_embeddings
    }
    for name, filename in index_files.items():
        stale_index = indices_dir / filename
        if name not in embedding_config.active_embeddings and stale_index.exists():
            stale_index.unlink()
            logger.info("Removed stale disabled index: %s", stale_index)
    valid_segment_ids = {int(segment_id) for segment_id in df["segment_id"]}
    for embedding_dir in embeddings_dir.iterdir():
        if not embedding_dir.is_dir():
            continue
        for artifact in embedding_dir.glob("segment_*.npy"):
            segment_id = int(artifact.stem.removeprefix("segment_"))
            if (
                embedding_dir.name not in embedding_config.active_embeddings
                or segment_id not in valid_segment_ids
            ):
                artifact.unlink()
    for artifact in audio_segments_dir.glob("segment_*_window_*.wav"):
        segment_id = int(artifact.name.split("_", 2)[1])
        if segment_id not in valid_segment_ids:
            artifact.unlink()
    orphan_clips = prune_orphan_clips(segment_clips_dir, valid_segment_ids)
    if orphan_clips:
        logger.info("Removed %d playback clips for segments no longer in the dataset", orphan_clips)
    for removed_name in removed_names:
        converted_path = converted_dir / removed_name
        if converted_path.exists():
            converted_path.unlink()
    for name, values in all_embeddings.items():
        build_faiss_index(values, str(indices_dir / index_files[name]))

    # --- Etapa 9: Dataset final ---
    logger.info("=" * 60)
    logger.info("ETAPA 9: Serialización del dataset final")
    logger.info("=" * 60)

    # Save complete dataset
    pkl_path = final_dir / "complete_dataset.pkl"
    df.to_pickle(pkl_path)
    logger.info("Saved complete dataset: %s", pkl_path)

    # Save metadata CSV (without embeddings for inspection)
    meta_cols = [c for c in df.columns if "embedding" not in c]
    metadata_df = df[meta_cols].copy()
    if "yamnet_top_classes" in metadata_df:
        metadata_df["yamnet_top_classes"] = metadata_df["yamnet_top_classes"].map(
            lambda classes: json.dumps(classes, ensure_ascii=False)
        )
    metadata_df.to_csv(final_dir / "dataset_metadata.csv", index=False)

    # Save manifest
    manifest = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "total_segments": len(df),
        "total_audio_files": df["original_file_name"].nunique(),
        "whisper_model": whisper_model,
        "chunking": {
            "strategy": chunk_strategy,
            "duration_sec": chunk_duration_sec,
            "overlap_sec": chunk_overlap_sec,
            "max_text_chars": max_chunk_text_chars,
        },
        "active_embeddings": sorted(embedding_config.active_embeddings),
        "active_classifiers": sorted(embedding_config.active_classifiers),
        "embeddings": {
            name: {
                "model": getattr(embedding_config, f"{name}_model"),
                "dimension": int(values.shape[1]),
                "index": index_files[name],
            }
            for name, values in all_embeddings.items()
        },
        "classifiers": (
            {
                "yamnet": {
                    "model": embedding_config.yamnet_model,
                    "top_k": embedding_config.yamnet_top_k,
                    "window_aggregation": "max_score",
                    "inverted_index": f"indices/{YAMNET_INVERTED_INDEX_FILENAME}",
                    "inverted_index_token_count": yamnet_index_token_count,
                }
            }
            if embedding_config.is_classifier_active("yamnet")
            else {}
        ),
        "audio_window_duration_sec": audio_window_duration_sec,
        "audio_window_overlap_sec": audio_window_overlap_sec,
        "segment_clips": _segment_clips_manifest(
            segment_clips_dir,
            enabled=segment_clips,
            context_sec=segment_clip_context_sec,
            bitrate=segment_clip_bitrate,
        ),
        "languages": df["language"].value_counts().to_dict(),
        "incremental": {
            "new_or_modified_files": sorted(changed_names),
            "removed_files": sorted(removed_names),
            "reused_files": sorted(set(source_fingerprints) - changed_names),
        },
    }
    manifest_path = final_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Saved manifest: %s", manifest_path)

    state_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "updated_at": datetime.now(UTC).isoformat(),
                "pipeline_signature": signature,
                "sources": source_fingerprints,
                "last_run": {
                    "new_or_modified_files": sorted(changed_names),
                    "removed_files": sorted(removed_names),
                    "reused_files": sorted(set(source_fingerprints) - changed_names),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_process_run_log(
        final_dir / "process_run.json",
        parameters={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "whisper_model": whisper_model,
            "batch_size": batch_size,
            "language": language,
            "chunk_strategy": chunk_strategy,
            "chunk_duration_sec": chunk_duration_sec,
            "chunk_overlap_sec": chunk_overlap_sec,
            "max_chunk_text_chars": max_chunk_text_chars,
            "embeddings_config_path": str(embeddings_config_path),
            "audio_window_duration_sec": audio_window_duration_sec,
            "audio_window_overlap_sec": audio_window_overlap_sec,
            "segment_clips": segment_clips,
            "segment_clip_context_sec": segment_clip_context_sec,
            "segment_clip_bitrate": segment_clip_bitrate,
            "mock_audio": mock_audio,
            "verbose": verbose,
            "incremental": manifest["incremental"],
        },
        embedding_config=embedding_config,
        command_argv=command_argv,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Dataset: %s", output_dir)
    logger.info(
        "Segments: %d | Files: %d", manifest["total_segments"], manifest["total_audio_files"]
    )
    logger.info("=" * 60)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Pipeline de ingesta de audio")
    parser.add_argument(
        "--input",
        default=os.getenv("AUDIO_INPUT_DIR"),
        required=os.getenv("AUDIO_INPUT_DIR") is None,
        help="Directorio con archivos de audio (default: variable de entorno AUDIO_INPUT_DIR)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("DATASET_OUTPUT"),
        required=os.getenv("DATASET_OUTPUT") is None,
        help="Directorio de salida del dataset (default: variable de entorno DATASET_OUTPUT)",
    )
    parser.add_argument(
        "--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default=None, help="Forzar idioma (ej: es, en)")
    parser.add_argument(
        "--chunk-strategy", default="whisper", choices=["whisper", "fixed", "sentence", "paragraph"]
    )
    parser.add_argument("--chunk-duration", type=float, default=30.0)
    parser.add_argument("--chunk-overlap", type=float, default=5.0)
    parser.add_argument("--max-chunk-text-chars", type=int, default=500)
    parser.add_argument(
        "--embeddings-config",
        default=os.getenv("EMBEDDINGS_CONFIG_PATH", "config/embeddings.toml"),
        help="Ruta del TOML generado desde .env (default: EMBEDDINGS_CONFIG_PATH)",
    )
    parser.add_argument(
        "--audio-window-duration",
        type=float,
        default=10.0,
        help="Duración máxima (s) de cada ventana acústica",
    )
    parser.add_argument(
        "--audio-window-overlap",
        type=float,
        default=2.0,
        help="Solapamiento (s) entre ventanas de un segmento largo",
    )
    parser.add_argument(
        "--no-segment-clips",
        dest="segment_clips",
        action="store_false",
        help="No generar los clips Opus de reproducción por segmento",
    )
    parser.add_argument(
        "--clip-context",
        type=float,
        default=float(os.getenv("SEGMENT_CLIP_CONTEXT_SEC", DEFAULT_CLIP_CONTEXT_SEC)),
        help="Contexto (s) antes y después del segmento en el clip de reproducción",
    )
    parser.add_argument(
        "--clip-bitrate",
        default=os.getenv("SEGMENT_CLIP_BITRATE", DEFAULT_CLIP_BITRATE),
        help="Bitrate del encoder Opus para los clips (default: 32k)",
    )
    parser.add_argument("--mock-audio", action="store_true", help="Usar embeddings mock para CLAP")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    write_embedding_config_from_env(args.embeddings_config)

    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        language=args.language,
        chunk_strategy=args.chunk_strategy,
        chunk_duration_sec=args.chunk_duration,
        chunk_overlap_sec=args.chunk_overlap,
        max_chunk_text_chars=args.max_chunk_text_chars,
        embeddings_config_path=args.embeddings_config,
        audio_window_duration_sec=args.audio_window_duration,
        audio_window_overlap_sec=args.audio_window_overlap,
        segment_clips=args.segment_clips,
        segment_clip_context_sec=args.clip_context,
        segment_clip_bitrate=args.clip_bitrate,
        mock_audio=args.mock_audio,
        verbose=args.verbose,
        command_argv=[sys.executable, *sys.argv],
    )


if __name__ == "__main__":
    main()
