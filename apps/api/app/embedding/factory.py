"""Embedding provider factory.

Cached so the same instance is reused across the request lifecycle.
Tests can call :func:`set_provider_for_tests` to swap in a mock
without touching the cache.
"""

from __future__ import annotations

import logging
from threading import Lock

from app.config import Settings, get_settings
from app.embedding.base import EmbeddingProvider
from app.embedding.hash_provider import HashEmbeddingProvider
from app.embedding.local_provider import LocalEmbeddingProvider
from app.embedding.minimax_provider import MiniMaxEmbeddingProvider
from app.embedding.openai_provider import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)

_instance: EmbeddingProvider | None = None
_lock = Lock()


def _build(settings: Settings) -> EmbeddingProvider:
    name = settings.embedding_provider.lower().strip()
    if name == "hash":
        return HashEmbeddingProvider(
            dimension=settings.embedding_dimension,
            model=settings.embedding_model,
        )
    if name == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.embedding_openai_api_key,
            base_url=settings.embedding_openai_base_url,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout_seconds=settings.embedding_timeout_seconds,
            dimensions=(
                settings.embedding_dimension
                if settings.embedding_openai_send_dimensions
                else None
            ),
            request_interval_seconds=settings.embedding_openai_request_interval,
            rpm_limit=settings.embedding_openai_rpm_limit,
            tpm_limit=settings.embedding_openai_tpm_limit,
        )
    if name == "minimax":
        return MiniMaxEmbeddingProvider(
            api_key=settings.embedding_minimax_api_key,
            group_id=settings.embedding_minimax_group_id,
            base_url=settings.embedding_minimax_base_url,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    if name == "local":
        return LocalEmbeddingProvider(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            query_instruction=settings.embedding_local_query_instruction,
            fp16=settings.embedding_local_fp16,
        )
    raise ValueError(
        f"unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}; "
        "expected one of: hash, openai, minimax, local"
    )


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    global _instance
    with _lock:
        if _instance is None:
            settings = settings or get_settings()
            _instance = _build(settings)
            logger.info(
                "embedding.provider.ready provider=%s model=%s dimension=%d",
                _instance.name, _instance.model, _instance.dimension,
            )
        return _instance


def set_provider_for_tests(provider: EmbeddingProvider) -> None:
    global _instance
    with _lock:
        _instance = provider


def reset_provider_for_tests() -> None:
    global _instance
    with _lock:
        _instance = None