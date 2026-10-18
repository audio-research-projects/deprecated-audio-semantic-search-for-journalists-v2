"""Publica un dataset procesado en Neon Postgres (pgvector) y Vercel Blob para el deploy en Vercel.

Paralelo a ``scripts/publish_dataset_release.py`` (que publica en Cloud Storage para GCP),
este script toma el directorio local generado por ``src/simple_dataset_pipeline.py`` y:

1. Sube opcionalmente ``segment_clips/*.opus`` a Vercel Blob (si se define ``BLOB_READ_WRITE_TOKEN``)
   o construye URLs públicas desde ``--clips-base-url``.
2. Crea las tablas e índices ``pgvector`` en Neon (``deploy/vercel/db/schema.sql``).
3. Inserta los metadatos de segmentos, vectores de texto/audio y el índice invertido de YAMNet.

Uso:
    uv run --with 'psycopg[binary]' python scripts/publish_dataset_neon.py \\
        --dataset ./dataset \\
        --dataset-id default \\
        --label "Archivo Periodístico"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.neon_exporter import (
    PreparedDatasetBundle,
    prepare_dataset_bundle,
    publish_bundle_to_neon,
    upload_clip_to_vercel_blob,
)
from src.publish_dataset_release_utils import default_release_name
from src.segment_clips import CLIP_DIR_NAME

logger = logging.getLogger("publish_dataset_neon")


def publish_processed_dataset(
    dataset_dir: Path,
    *,
    dataset_id: str,
    label: str,
    language: str,
    release: str | None = None,
    audio_embedding_source: str = "auto",
    clips_base_url: str = "",
    skip_clips: bool = False,
    is_default: bool = True,
    dry_run: bool = False,
) -> PreparedDatasetBundle:
    """Upload playback clips and build the Neon bundle for a processed dataset.

    Returns the prepared bundle; the caller decides whether to write it to
    Neon with ``publish_bundle_to_neon`` (``dry_run`` only skips the Blob
    upload here).
    """
    release = release or default_release_name(dataset_dir)
    clips_dir = dataset_dir / CLIP_DIR_NAME
    blob_token = os.getenv("BLOB_READ_WRITE_TOKEN", "").strip()
    clips_base_url = clips_base_url.strip().rstrip("/")

    clip_urls: dict[str, str] = {}
    if not skip_clips and clips_dir.is_dir():
        if blob_token and not dry_run:
            clips = sorted(clips_dir.glob("segment_*.opus"))
            logger.info("Subiendo %d clips a Vercel Blob...", len(clips))
            with httpx.Client(timeout=60.0) as http_client:
                for clip_path in clips:
                    clip_urls[clip_path.name] = upload_clip_to_vercel_blob(
                        clip_path,
                        dataset_id=dataset_id,
                        release=release,
                        blob_token=blob_token,
                        client=http_client,
                    )
        elif clips_base_url:
            for clip_path in sorted(clips_dir.glob("segment_*.opus")):
                clip_urls[clip_path.name] = f"{clips_base_url}/{clip_path.name}"
        elif blob_token and dry_run:
            for clip_path in sorted(clips_dir.glob("segment_*.opus")):
                clip_urls[clip_path.name] = (
                    f"https://blob.vercel-storage.com/audio-search/{dataset_id}/{release}/{CLIP_DIR_NAME}/{clip_path.name}"
                )
        else:
            logger.warning(
                "Sin BLOB_READ_WRITE_TOKEN ni --clips-base-url: los segmentos se publicarán sin clip_url."
            )

    return prepare_dataset_bundle(
        dataset_dir,
        dataset_id=dataset_id,
        label=label,
        language=language,
        release=release,
        is_default=is_default,
        audio_embedding_source=audio_embedding_source,
        clip_url_resolver=lambda _seg_id, name: clip_urls.get(name),
    )


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("DATASET_OUTPUT") or os.getenv("DATASET_PATH", "./dataset"),
        help="Directorio del dataset procesado (default: DATASET_OUTPUT o DATASET_PATH)",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("DEFAULT_DATASET_ID", "default"),
        help="ID estable del dataset en Neon (default: 'default')",
    )
    parser.add_argument(
        "--label",
        default="Archivo periodístico",
        help="Nombre legible del dataset para la UI",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("QUERY_LANGUAGE", "es"),
        help="Idioma principal del dataset (default: es)",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="Identificador de release (default: fecha + hash del dataset)",
    )
    parser.add_argument(
        "--audio-embedding-source",
        choices=("auto", "gemini", "embeddinggemma", "clap"),
        default="auto",
        help="Vector acústico a publicar en audio_embedding (default: auto, prefiere gemini → embeddinggemma → clap)",
    )
    parser.add_argument(
        "--clips-base-url",
        default=os.getenv("CLIPS_PUBLIC_BASE_URL", ""),
        help="URL base pública si los clips ya fueron subidos a un bucket S3/R2/Blob externo",
    )
    parser.add_argument(
        "--skip-clips",
        action="store_true",
        help="No subir segment_clips/ a Vercel Blob",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="No marcar este dataset como default en Neon",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preparar el bundle y validar sin escribir en Neon ni en Vercel Blob",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    if not (dataset_dir / "final" / "complete_dataset.pkl").is_file():
        logger.error("No se encontró final/complete_dataset.pkl en %s", dataset_dir)
        return 1

    bundle = publish_processed_dataset(
        dataset_dir,
        dataset_id=args.dataset_id,
        label=args.label,
        language=args.language,
        release=args.release,
        audio_embedding_source=args.audio_embedding_source,
        clips_base_url=args.clips_base_url,
        skip_clips=args.skip_clips,
        is_default=not args.no_default,
        dry_run=args.dry_run,
    )

    print()
    print(f"Dataset preparado: {bundle.dataset_id} ({bundle.label})")
    print(f"Release:           {bundle.dataset_version}")
    print(f"Segmentos:         {len(bundle.segments)}")
    print(f"Índices activos:   {', '.join(bundle.active_indexes) or 'ninguno'}")
    print(f"Postings YAMNet:   {len(bundle.yamnet_postings)}")
    print(f"Clips con URL:     {sum(1 for s in bundle.segments if s.clip_url)}")

    if args.dry_run:
        print("\n[dry-run] No se realizaron cambios en Neon ni en Vercel Blob.")
        return 0

    publish_bundle_to_neon(bundle)
    print("\nPublicado correctamente en Neon Postgres.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
