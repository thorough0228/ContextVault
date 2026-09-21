"""Embedding provider abstraction (Phase 4).

Following AGENTS.md rule 8 (\"all external AI providers must be called
through an abstract interface\"), every code path that turns text
into a vector goes through :class:`EmbeddingProvider` and is selected
by :func:`get_embedding_provider` based on the configured name.

The interface is intentionally narrow:

* ``embed_text(text)`` → single ``list[float]``
* ``embed_texts(texts)`` → ``list[list[float]]`` (batched)

Implementations live alongside this package — ``hash_provider`` ships
a deterministic stub for tests / offline dev, ``openai_provider``
ships an OpenAI-compatible stub that callers can flesh out when they
have an API key.
"""

from app.embedding.base import (
    EmbeddingProvider,
    EmbeddingTransientError,
    EmbeddingPermanentError,
)
from app.embedding.factory import (
    get_embedding_provider,
    reset_provider_for_tests,
    set_provider_for_tests,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingTransientError",
    "EmbeddingPermanentError",
    "get_embedding_provider",
    "set_provider_for_tests",
    "reset_provider_for_tests",
]