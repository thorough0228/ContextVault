"""LLM package — provider abstraction and concrete implementations.

The chat service talks to :func:`get_llm_provider`, never to a
specific vendor SDK. To add a new provider (Azure, Anthropic,
self-hosted), implement :class:`app.llm.base.LLMProvider` and
register it in :mod:`app.llm.factory`.
"""

from __future__ import annotations

from app.llm.base import (
    LLMProvider,
    LLMPermanentError,
    LLMTransientError,
)
from app.llm.factory import (
    get_llm_provider,
    reset_provider_for_tests,
    set_provider_for_tests,
)

__all__ = [
    "LLMProvider",
    "LLMPermanentError",
    "LLMTransientError",
    "get_llm_provider",
    "set_provider_for_tests",
    "reset_provider_for_tests",
]