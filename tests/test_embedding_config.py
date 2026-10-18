import pytest

from src.embedding_config import (
    embedding_config_from_env,
    load_embedding_config,
    write_embedding_config_from_env,
)


def test_load_embedding_config_reads_embeddings_and_classifiers(tmp_path):
    config_path = tmp_path / "embeddings.toml"
    config_path.write_text(
        "[embeddings]\nactive = ['text', 'gemini']\n"
        "[embeddings.gemini]\noutput_dimensionality = 768\n"
        "[classifiers]\nactive = ['yamnet']\n"
    )

    config = load_embedding_config(config_path)

    assert config.active_embeddings == frozenset({"text", "gemini"})
    assert config.active_classifiers == frozenset({"yamnet"})
    assert config.gemini_output_dimensionality == 768


def test_load_embedding_config_rejects_unknown_embedding(tmp_path):
    config_path = tmp_path / "embeddings.toml"
    config_path.write_text("[embeddings]\nactive = ['unknown']\n")

    with pytest.raises(ValueError, match="Unsupported"):
        load_embedding_config(config_path)


def test_load_embedding_config_migrates_legacy_yamnet_entry(tmp_path):
    config_path = tmp_path / "embeddings.toml"
    config_path.write_text(
        "[embeddings]\nactive = ['text', 'yamnet']\n"
        "[embeddings.yamnet]\nmodel = 'https://example.test/yamnet'\ntop_k = 3\n"
    )

    config = load_embedding_config(config_path)

    assert config.active_embeddings == frozenset({"text"})
    assert config.is_classifier_active("yamnet")
    assert config.yamnet_model == "https://example.test/yamnet"
    assert config.yamnet_top_k == 3


def test_embedding_config_from_env_reads_enabled_models():
    config = embedding_config_from_env(
        {
            "EMBEDDINGS_ACTIVE": "text, clap, gemini",
            "CLASSIFIERS_ACTIVE": "yamnet",
            "GEMINI_EMBEDDING_MODEL": "gemini-embedding-2",
            "GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY": "768",
        }
    )

    assert config.active_embeddings == frozenset({"text", "clap", "gemini"})
    assert config.active_classifiers == frozenset({"yamnet"})
    assert config.gemini_output_dimensionality == 768


def test_write_embedding_config_from_env_generates_loadable_toml(tmp_path):
    config_path = tmp_path / "embeddings.toml"

    write_embedding_config_from_env(
        config_path,
        {
            "EMBEDDINGS_ACTIVE": "gemini",
            "CLASSIFIERS_ACTIVE": "yamnet",
            "GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY": "1536",
        },
    )

    loaded = load_embedding_config(config_path)
    assert loaded.active_embeddings == frozenset({"gemini"})
    assert loaded.active_classifiers == frozenset({"yamnet"})
    assert loaded.gemini_output_dimensionality == 1536


def test_embedding_config_from_env_rejects_yamnet_as_an_embedding():
    with pytest.raises(ValueError, match="EMBEDDINGS_ACTIVE"):
        embedding_config_from_env({"EMBEDDINGS_ACTIVE": "text,yamnet"})


def test_embedding_config_from_env_reads_embeddinggemma():
    config = embedding_config_from_env(
        {
            "EMBEDDINGS_ACTIVE": "text,embeddinggemma",
            "EMBEDDINGGEMMA_MODEL": "google/embeddinggemma-2",
            "EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY": "256",
        }
    )

    assert config.active_embeddings == frozenset({"text", "embeddinggemma"})
    assert config.embeddinggemma_model == "google/embeddinggemma-2"
    assert config.embeddinggemma_output_dimensionality == 256


def test_embedding_config_from_env_rejects_invalid_embeddinggemma_dimensionality():
    with pytest.raises(ValueError, match="EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY"):
        embedding_config_from_env(
            {"EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY": "300"}
        )


def test_load_embedding_config_reads_embeddinggemma(tmp_path):
    config_path = tmp_path / "embeddings.toml"
    config_path.write_text(
        "[embeddings]\nactive = ['text', 'embeddinggemma']\n"
        "[embeddings.embeddinggemma]\n"
        "model = 'google/embeddinggemma-2'\noutput_dimensionality = 512\n"
    )

    config = load_embedding_config(config_path)

    assert config.active_embeddings == frozenset({"text", "embeddinggemma"})
    assert config.embeddinggemma_model == "google/embeddinggemma-2"
    assert config.embeddinggemma_output_dimensionality == 512


def test_write_embedding_config_from_env_includes_embeddinggemma_section(tmp_path):
    config_path = tmp_path / "embeddings.toml"

    write_embedding_config_from_env(
        config_path,
        {
            "EMBEDDINGS_ACTIVE": "text,embeddinggemma",
            "EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY": "128",
        },
    )

    loaded = load_embedding_config(config_path)
    assert loaded.active_embeddings == frozenset({"text", "embeddinggemma"})
    assert loaded.embeddinggemma_output_dimensionality == 128
