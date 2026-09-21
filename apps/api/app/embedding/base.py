"""Embedding provider interface + error taxonomy.

The two error types map directly to the Celery retry policy in
``app/tasks.py``: ``EmbeddingTransientError`` triggers retry,
``EmbeddingPermanentError`` (the default) marks the document
``FAILED`` immediately.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


class EmbeddingTransientError(Exception):
    """Embedding provider hiccup — worth retrying."""


class EmbeddingPermanentError(Exception):
    """Embedding failure that won't fix itself (bad input, auth, etc.)."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-agnostic embedding interface.

    Implementations MUST honour the configured ``dimension`` — the
    factory rejects any provider that returns vectors of the wrong
    length.
    """

    @property
    def name(self) -> str:  # logical name (e.g. "hash", "openai")
        ...

    @property
    def model(self) -> str:  # model identifier (e.g. "text-embedding-3-small")
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_text(self, text: str) -> List[float]:
        """Vectorize one string. Raise :class:`EmbeddingTransientError`
        on recoverable failures and :class:`EmbeddingPermanentError`
        on permanent ones (bad input, auth, dimension mismatch)."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batched variant. Default impl loops over ``embed_text``;
        providers should override for throughput.
        """
        ...