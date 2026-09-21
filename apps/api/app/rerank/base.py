"""Reranker base — Protocol + error taxonomy.

A reranker takes a query and candidate documents and returns a
relevance score per document. Used by the hybrid search pipeline to
precision-reorder fused candidates; failures are non-fatal (the
search service falls back to the fused ranking).
"""

from __future__ import annotations

from typing import List, Protocol


class RerankPermanentError(Exception):
    """Bad request / auth / model unavailable — retrying won't help."""


class RerankTransientError(Exception):
    """Timeout / 5xx / rate limit — worth retrying."""


class RerankProvider(Protocol):
    name: str
    model: str

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Return one relevance score per document, same order as
        ``documents`` (not re-sorted — the caller keeps its own order
        and applies the scores positionally)."""
        ...
