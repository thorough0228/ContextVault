"""LLM provider factory.

Cached so the same instance is reused across the request lifecycle.
Tests can call :func:`set_provider_for_tests` to swap in a mock
without touching the cache.
"""

from __future__ import annotations

import logging
from threading import Lock

from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.hash_provider import HashLLMProvider
from app.llm.openai_provider import OpenAILLMProvider

logger = logging.getLogger(__name__)

_instance: LLMProvider | None = None
_lock = Lock()


def _build(settings: Settings) -> LLMProvider:
    name = settings.llm_provider.lower().strip()
    if name == "hash":
        return HashLLMProvider(model=settings.llm_model)
    if name == "openai":
        return OpenAILLMProvider(
            api_key=settings.llm_openai_api_key,
            base_url=settings.llm_openai_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_context_tokens=settings.llm_max_context_tokens,
            strip_think=settings.llm_strip_think,
        )
    raise ValueError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r}; "
        "expected one of: hash, openai"
    )


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    global _instance
    with _lock:
        if _instance is None:
            settings = settings or get_settings()
            _instance = _build(settings)
            logger.info(
                "llm.provider.ready provider=%s model=%s",
                _instance.name, _instance.model,
            )
        return _instance


def set_provider_for_tests(provider: LLMProvider) -> None:
    global _instance
    with _lock:
        _instance = provider


def reset_provider_for_tests() -> None:
    global _instance
    with _lock:
        _instance = None