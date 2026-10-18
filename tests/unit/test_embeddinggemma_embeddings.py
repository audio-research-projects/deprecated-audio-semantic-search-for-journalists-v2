"""Unit tests for the EmbeddingGemma 2 wrapper (no model download required)."""

import numpy as np
import pytest

from src.embeddinggemma_embeddings import (
    EmbeddingGemmaConfig,
    EmbeddingGemmaMultimodalEmbedding,
)


def test_config_rejects_non_matryoshka_dimensionality():
    with pytest.raises(ValueError, match="Matryoshka"):
        EmbeddingGemmaMultimodalEmbedding(EmbeddingGemmaConfig(output_dimensionality=300))


def test_finalize_truncates_and_normalizes():
    embedder = EmbeddingGemmaMultimodalEmbedding(
        EmbeddingGemmaConfig(output_dimensionality=128)
    )
    values = np.ones(768, dtype=np.float32)

    result = embedder._finalize(values)

    assert result.shape == (128,)
    assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)


def test_finalize_keeps_full_dim_and_handles_zero_vector():
    embedder = EmbeddingGemmaMultimodalEmbedding()

    result = embedder._finalize(np.zeros(768, dtype=np.float32))

    assert result.shape == (768,)
    assert np.all(result == 0)


def test_generate_audio_embedding_missing_file():
    embedder = EmbeddingGemmaMultimodalEmbedding()

    with pytest.raises(FileNotFoundError):
        embedder.generate_audio_embedding("/nonexistent/audio.wav")
