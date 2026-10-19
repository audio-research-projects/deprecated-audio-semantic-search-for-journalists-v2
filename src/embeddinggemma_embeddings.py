"""EmbeddingGemma 2 multimodal embeddings (text + audio in one shared space).

EmbeddingGemma 2 (``google/embeddinggemma-2``) replaces CLAP as the local
audio-embedding backend: audio windows and text queries are mapped into the
same vector space, so cross-modal retrieval is plain cosine similarity.
It runs through ``sentence-transformers>=6.1`` and accepts WAV paths directly
(16 kHz mono is applied internally), so it fits the same per-window pooling
strategy the pipeline already uses for CLAP/Gemini.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.query_translation import translate_to_english

logger = logging.getLogger(__name__)

EMBEDDING_GEMMA_DIM = 768
# Matryoshka Representation Learning: vectors stay valid when truncated.
MRL_DIMENSIONS = (768, 512, 256, 128)


@dataclass(frozen=True)
class EmbeddingGemmaConfig:
    model_name: str = "google/embeddinggemma-2"
    output_dimensionality: int = EMBEDDING_GEMMA_DIM
    device: str | None = field(
        default_factory=lambda: os.environ.get("EMBEDDINGGEMMA_DEVICE") or None
    )
    # Text + audio workloads only need ~570M params; the vision encoder adds
    # ~170M and is skipped unless explicitly enabled.
    include_vision: bool = False


class EmbeddingGemmaMultimodalEmbedding:
    """Embed WAV windows and text queries into EmbeddingGemma 2's shared space."""

    def __init__(self, config: EmbeddingGemmaConfig | None = None):
        self.config = config or EmbeddingGemmaConfig()
        if self.config.output_dimensionality not in MRL_DIMENSIONS:
            raise ValueError(
                "output_dimensionality must be one of "
                f"{MRL_DIMENSIONS} (Matryoshka truncation levels)"
            )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        import torch
        from sentence_transformers import SentenceTransformer

        config_kwargs = {} if self.config.include_vision else {"vision_config": None}
        logger.info(
            "Loading EmbeddingGemma 2: %s (dim: %d, vision: %s)",
            self.config.model_name,
            self.config.output_dimensionality,
            self.config.include_vision,
        )
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
            config_kwargs=config_kwargs,
            # CPU inference needs float32; the checkpoint defaults to bfloat16.
            # float16 is incompatible with EmbeddingGemma 2's activation range.
            model_kwargs={"torch_dtype": torch.float32},
        )
        logger.info("EmbeddingGemma 2 loaded successfully")

    @property
    def embedding_dim(self) -> int:
        return self.config.output_dimensionality

    def _finalize(self, values) -> np.ndarray:
        """MRL-truncate to the configured dimensionality and L2-normalize."""
        embedding = np.asarray(values, dtype=np.float32).reshape(-1)
        embedding = embedding[: self.config.output_dimensionality]
        norm = np.linalg.norm(embedding)
        return (embedding / norm if norm > 0 else embedding).astype(np.float32)

    def generate_audio_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Embed one audio window (WAV/MP3/FLAC; resampled to 16 kHz internally)."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self._finalize(self.model.encode({"audio": str(path)}))

    def generate_batch_audio_embeddings(
        self, audio_paths: list[str | Path], batch_size: int = 8
    ) -> np.ndarray:
        """Embed multiple audio files; returns shape (n_files, embedding_dim)."""
        embeddings = []
        for index, audio_path in enumerate(audio_paths):
            if index % batch_size == 0:
                logger.info(
                    "EmbeddingGemma audio %d/%d", index + 1, len(audio_paths)
                )
            embeddings.append(self.generate_audio_embedding(audio_path))
        return np.asarray(embeddings, dtype=np.float32)

    def generate_query_embedding(
        self, query: str, *, source_language: str | None = None
    ) -> np.ndarray:
        """Translate a text query to English, then encode it for audio retrieval."""
        query = translate_to_english(query, source_language)
        return self._finalize(self.model.encode(query, prompt_name="SearchQuery"))

    # Alias matching the CLAPEmbedding contract used by query-side callers.
    def generate_text_embedding(
        self, text: str, *, source_language: str | None = None
    ) -> np.ndarray:
        return self.generate_query_embedding(text, source_language=source_language)
