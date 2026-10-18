"""Ingesta end-to-end para el deploy de Vercel: YouTube → dataset → Neon.

Flujo:
1. Descarga el audio de una o más URLs de YouTube con ``yt-dlp`` (``src/youtube_download.py``).
2. Corre el pipeline de ingesta estándar (``src/simple_dataset_pipeline.py``):
   conversión a WAV 16 kHz, transcripción Whisper, segmentación, embeddings
   (texto/CLAP/Gemini/EmbeddingGemma 2 según ``EMBEDDINGS_ACTIVE``), YAMNet,
   clips Opus e índices FAISS.
3. Publica el dataset en Neon Postgres (pgvector) y sube los clips a Vercel
   Blob (misma lógica que ``scripts/publish_dataset_neon.py``).

Uso:
    EMBEDDINGS_ACTIVE=text,embeddinggemma \\
    uv run python scripts/ingest_youtube_neon.py \\
        --youtube-url "https://www.youtube.com/watch?v=DVt_mzCOGtA" \\
        --dataset ./dataset-youtube \\
        --dataset-id youtube \\
        --label "YouTube ingest"

Las descargas van a ``--download-dir`` (default ``./youtube-audio``) y se
reutilizan entre corridas: el pipeline es incremental sobre ese directorio.

Requiere: grupo ``ml`` (yt-dlp, whisper, embeddings), ``ffmpeg`` en PATH,
``DATABASE_URL`` y ``BLOB_READ_WRITE_TOKEN`` (o ``--clips-base-url``) para la
publicación.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.embedding_config import write_embedding_config_from_env
from src.neon_exporter import publish_bundle_to_neon
from src.simple_dataset_pipeline import run_pipeline
from src.youtube_download import download_youtube_audio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_dataset_neon import publish_processed_dataset

logger = logging.getLogger("ingest_youtube_neon")


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--youtube-url",
        nargs="+",
        required=True,
        help="Una o más URLs de YouTube a ingerir",
    )
    parser.add_argument(
        "--download-dir",
        default=os.getenv("YOUTUBE_AUDIO_DIR", "./youtube-audio"),
        help="Directorio persistente para los audios descargados (default: ./youtube-audio)",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("DATASET_OUTPUT", "./dataset-youtube"),
        help="Directorio de salida del dataset procesado (default: DATASET_OUTPUT o ./dataset-youtube)",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("DEFAULT_DATASET_ID", "youtube"),
        help="ID del dataset en Neon (default: 'youtube')",
    )
    parser.add_argument("--label", default="YouTube ingest", help="Nombre legible del dataset")
    parser.add_argument(
        "--language",
        default=os.getenv("QUERY_LANGUAGE", "es"),
        help="Idioma del dataset / forzado para Whisper (default: es)",
    )
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument(
        "--embeddings-config",
        default=os.getenv("EMBEDDINGS_CONFIG_PATH", "config/embeddings.toml"),
        help="TOML generado desde .env (EMBEDDINGS_ACTIVE decide text/clap/gemini/embeddinggemma)",
    )
    parser.add_argument(
        "--audio-embedding-source",
        choices=("auto", "gemini", "embeddinggemma", "clap"),
        default="auto",
        help="Vector acústico a publicar en audio_embedding (default: auto)",
    )
    parser.add_argument(
        "--clips-base-url",
        default=os.getenv("CLIPS_PUBLIC_BASE_URL", ""),
        help="URL base pública si los clips ya están en un bucket externo (saltea Vercel Blob)",
    )
    parser.add_argument("--skip-clips", action="store_true", help="No subir clips")
    parser.add_argument("--skip-publish", action="store_true", help="Solo descargar + procesar, no publicar en Neon")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Saltear la ingesta (el dataset ya fue procesado) y solo publicar",
    )
    parser.add_argument("--no-default", action="store_true", help="No marcar como dataset default")
    parser.add_argument("--mock-audio", action="store_true", help="Embeddings mock (testing)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Descargar + procesar + preparar bundle, sin escribir en Neon ni subir clips",
    )
    args = parser.parse_args()

    download_dir = Path(args.download_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset).expanduser().resolve()

    # --- 1. Descarga ---
    if not args.skip_pipeline:
        logger.info("Descargando %d video(s) de YouTube a %s", len(args.youtube_url), download_dir)
        for url in args.youtube_url:
            downloaded = download_youtube_audio(url, download_dir)
            print(f"  ✓ {downloaded.title} → {downloaded.path.name}")

    # --- 2. Pipeline de ingesta ---
    if not args.skip_pipeline:
        write_embedding_config_from_env(args.embeddings_config)
        logger.info("Ejecutando pipeline de ingesta sobre %s → %s", download_dir, dataset_dir)
        run_pipeline(
            input_dir=str(download_dir),
            output_dir=str(dataset_dir),
            whisper_model=args.whisper_model,
            language=args.language or None,
            embeddings_config_path=args.embeddings_config,
            mock_audio=args.mock_audio,
            verbose=args.verbose,
            command_argv=[sys.executable, *sys.argv],
        )
    elif not (dataset_dir / "final" / "complete_dataset.pkl").is_file():
        logger.error("--skip-pipeline requiere final/complete_dataset.pkl en %s", dataset_dir)
        return 1

    # --- 3. Publicación en Neon ---
    if args.skip_publish:
        logger.info("Publicación omitida (--skip-publish). Dataset local en %s", dataset_dir)
        return 0

    bundle = publish_processed_dataset(
        dataset_dir,
        dataset_id=args.dataset_id,
        label=args.label,
        language=args.language,
        audio_embedding_source=args.audio_embedding_source,
        clips_base_url=args.clips_base_url,
        skip_clips=args.skip_clips,
        is_default=not args.no_default,
        dry_run=args.dry_run,
    )

    print()
    print(f"Dataset preparado: {bundle.dataset_id} ({bundle.label})")
    print(f"Segmentos:         {len(bundle.segments)}")
    print(f"Índices activos:   {', '.join(bundle.active_indexes) or 'ninguno'}")
    print(f"Clips con URL:     {sum(1 for s in bundle.segments if s.clip_url)}")

    if args.dry_run:
        print("\n[dry-run] No se realizaron cambios en Neon ni en Vercel Blob.")
        return 0

    publish_bundle_to_neon(bundle)
    print("\nIngesta YouTube publicada en Neon Postgres.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
