"""CLAP audio embeddings module for cross-modal audio-text search."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.query_translation import translate_to_english

logger = logging.getLogger(__name__)

CLAP_EMBEDDING_DIM = 512


@dataclass
class CLAPConfig:
    model_name: str = "laion/clap-htsat-unfused"
    device: str = field(default_factory=lambda: os.environ.get("CLAP_DEVICE", "cpu"))
    cache_dir: str = "/tmp/clap_cache"
    # Language of incoming text queries. If not English, the query is translated
    # to English before CLAP's text encoder, because laion/clap-htsat-unfused
    # uses a RoBERTa-base text encoder trained on English.
    query_language: str = field(default_factory=lambda: os.environ.get("QUERY_LANGUAGE", "es"))


class CLAPEmbedding:
    """CLAP model wrapper for generating audio and text embeddings in shared space."""

    def __init__(self, config: CLAPConfig | None = None):
        self.config = config or CLAPConfig()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Lazy-load the CLAP model."""
        import laion_clap

        logger.info(
            "Loading CLAP model: %s (device: %s)", self.config.model_name, self.config.device
        )
        self._model = laion_clap.CLAP_Module(enable_fusion=False, device=self.config.device)
        self._model.load_ckpt()
        logger.info("CLAP model loaded successfully")

    @property
    def embedding_dim(self) -> int:
        return CLAP_EMBEDDING_DIM

    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera embedding de audio (512-dim, normalizado).

        Args:
            audio_path: Path to audio file

        Returns:
            Normalized numpy array of shape (512,)
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        embedding = self.model.get_audio_embedding_from_filelist(
            [str(audio_file)], use_tensor=False
        )
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def generate_text_embedding(
        self, text: str, *, source_language: str | None = None
    ) -> np.ndarray:
        """
        Genera embedding de texto en espacio CLAP (512-dim, normalizado).

        If ``self.config.query_language`` is not English, the query is first
        translated to English, because the underlying text encoder
        (RoBERTa-base) was trained on English.

        Args:
            text: Text query for cross-modal search
            source_language: Optional language override for this query. Use
                ``en`` when the caller already translated the text.

        Returns:
            Normalized numpy array of shape (512,)
        """
        query_language = self.config.query_language if source_language is None else source_language
        text = translate_to_english(text, query_language)
        embedding = self.model.get_text_embedding([text], use_tensor=False)
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def generate_text_embeddings_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Generate normalized text embeddings for a batch of queries.

        Args:
            texts: List of text queries.
            batch_size: Optional chunk size to avoid CPU/RAM spikes with very
                large lists. Defaults to processing all at once.

        Returns:
            Normalized numpy array of shape (len(texts), 512).
        """
        translated = [translate_to_english(t, self.config.query_language) for t in texts]
        if batch_size is None:
            embeddings = self.model.get_text_embedding(translated, use_tensor=False)
        else:
            chunks = [translated[i : i + batch_size] for i in range(0, len(translated), batch_size)]
            embeddings = np.concatenate(
                [self.model.get_text_embedding(chunk, use_tensor=False) for chunk in chunks]
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    def generate_batch_audio_embeddings(
        self, audio_paths: list[str], batch_size: int = 8
    ) -> np.ndarray:
        """
        Generate embeddings for multiple audio files.

        Args:
            audio_paths: List of audio file paths
            batch_size: Processing batch size

        Returns:
            Array of shape (n_files, 512)
        """
        all_embeddings = []

        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            logger.info(
                "CLAP batch %d/%d (%d files)",
                i // batch_size + 1,
                (len(audio_paths) + batch_size - 1) // batch_size,
                len(batch),
            )
            embeddings = self.model.get_audio_embedding_from_filelist(batch, use_tensor=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings).astype(np.float32)
        logger.info("Generated %d CLAP audio embeddings", len(result))
        return result
