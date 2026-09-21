"""Reranker factory — cached singleton, env-driven (rule 8 style)."""

from __future__ import annotations

import logging
from threading import Lock

from app.config import Settings, get_settings
from app.rerank.base import RerankProvider
from app.rerank.siliconflow_provider import SiliconFlowRerankProvider

logger = logging.getLogger(__name__)

_instance: RerankProvider | None = None
_lock = Lock()


def _build(settings: Settings) -> RerankProvider:
    return SiliconFlowRerankProvider(
        api_key=settings.embedding_openai_api_key,
        base_url=settings.embedding_openai_base_url,
        model=settings.search_rerank_model,
        timeout_seconds=settings.embedding_timeout_seconds,
        request_interval_seconds=settings.embedding_openai_request_interval,
    )


def get_rerank_provider(settings: Settings | None = None) -> RerankProvider:
    global _instance
    with _lock:
        if _instance is None:
            _instance = _build(settings or get_settings())
            logger.info("rerank.provider.ready model=%s", _instance.model)
        return _instance


def set_rerank_provider_for_tests(provider: RerankProvider) -> None:
    global _instance
    with _lock:
        _instance = provider


def reset_rerank_provider_for_tests() -> None:
    global _instance
    with _lock:
        _instance = None
