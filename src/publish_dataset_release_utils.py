"""Shared release naming utility for GCS and Neon dataset publishing scripts."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_FILE = Path("final/complete_dataset.pkl")


def default_release_name(dataset_dir: Path) -> str:
    """Return a date + content digest identifier for a processed dataset directory."""
    digest = hashlib.sha256()
    digest.update((dataset_dir / REQUIRED_FILE).read_bytes())
    indices_dir = dataset_dir / "indices"
    if indices_dir.is_dir():
        for index in sorted(indices_dir.glob("*.faiss")):
            digest.update(index.read_bytes())
    return f"{datetime.now(UTC):%Y-%m-%d}-{digest.hexdigest()[:7]}"
