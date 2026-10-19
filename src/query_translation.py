"""Gateway query translator for CLAP, EmbeddingGemma and AudioSet/YAMNet search.

`laion/clap-htsat-unfused` uses a RoBERTa-base text encoder that was trained on
English text. Queries in other languages should be translated before generating
the CLAP text embedding or matching the English AudioSet class taxonomy.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

ENGLISH_ALIASES = {"en", "english", "ingles", "inglés"}


@lru_cache(maxsize=1)
def _openai_client():
    try:
        import openai
    except ImportError as error:
        raise RuntimeError(
            "Translation requires the 'openai' package for the Gateway client."
        ) from error
    token = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_OIDC_TOKEN")
    if not token:
        raise RuntimeError("Query translation requires AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN")
    return openai.OpenAI(api_key=token, base_url="https://ai-gateway.vercel.sh/v1")


def _is_english(language: str) -> bool:
    return language.strip().lower().split("_")[0].split("-")[0] in ENGLISH_ALIASES


def _translate_with_ollama(text: str) -> str:
    """Translate locally through the Ollama chat API."""
    import httpx

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3.8:27b")
    prompt = (
        "Translate the user's query into English for semantic audio search. "
        "Preserve proper nouns, quoted phrases, and numbers. Return only the translation.\n\n"
        f"Query: {text}"
    )
    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=httpx.Timeout(120.0, connect=3.0),
        )
        response.raise_for_status()
        translated = response.json().get("message", {}).get("content", "").strip()
    except (httpx.HTTPError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"No se pudo traducir con Ollama ({model}). Verificá que Ollama esté iniciado y el modelo descargado."
        ) from error
    if not translated:
        raise RuntimeError(f"Ollama ({model}) devolvió una traducción vacía")
    return translated


def translate_to_english(
    text: str, source_language: str | None = None, *, provider: str = "gateway"
) -> str:
    """Translate a query to English if the configured source language is not English.

    Args:
        text: Original query text.
        source_language: Source language code or name. If None, read from
            ``QUERY_LANGUAGE`` environment variable, defaulting to "es".

    Returns:
        The original text if already English or if translation is disabled;
        otherwise the English translation.
    """
    if source_language is None:
        source_language = os.environ.get("QUERY_LANGUAGE", "es")
    if _is_english(source_language):
        return text
    if not text or not text.strip():
        return text

    if provider == "ollama":
        return _translate_with_ollama(text)
    if provider != "gateway":
        raise ValueError(f"Unknown query translation provider: {provider}")

    client = _openai_client()
    model = os.environ.get("TEXT_GENERATION_MODEL", "openai/gpt-4o-mini")
    logger.debug("Translating query from %s to English with %s", source_language, model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator for an audio retrieval system. "
                    "Translate the user's query to English. Keep proper nouns, "
                    "brand names, quoted phrases and numbers unchanged. "
                    "Return only the translation, no explanations."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    translated = (response.choices[0].message.content or "").strip()
    if not translated:
        raise RuntimeError("Gateway returned an empty English translation")
    logger.debug("Translated query: %r -> %r", text, translated)
    return translated
