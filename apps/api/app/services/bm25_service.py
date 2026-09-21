"""BM25 keyword scoring over a RAG's chunks (Phase 12 hybrid search).

Per-RAG in-memory BM25Okapi index, built lazily from the persisted
``document_chunks.tokenized`` column (already jieba-segmented at
ingestion time — building the index is a plain split + IDF pass,
~1-3s for a 10k-chunk RAG, once per process).

Invalidation: callers (ingestion completion, document deletion) invoke
:func:`invalidate` so the next query rebuilds from current rows.
Scoring: ``BM25Okapi.get_scores`` over the whole RAG corpus, then the
top-k hits are max-normalised to [0, 1] to match the wire contract.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Dict[str, "_RagIndex"] = {}


class _RagIndex:
    __slots__ = ("bm25", "order", "meta")

    def __init__(self, bm25, order: List[str], meta: Dict[str, dict]):
        self.bm25 = bm25
        self.order = order  # chunk ids aligned with the bm25 corpus rows
        self.meta = meta


def invalidate(rag_id: str) -> None:
    """Drop the cached BM25 index for ``rag_id`` (rebuilds lazily)."""

    with _lock:
        _cache.pop(rag_id, None)


def invalidate_all() -> None:
    with _lock:
        _cache.clear()


async def bm25_keyword_hits(
    db: AsyncSession,
    *,
    rag_id: str,
    query: str,
    top_k: int,
) -> Optional[List[dict]]:
    """Top-k keyword hits via BM25, or ``None`` when BM25 is
    unavailable (rank_bm25 missing / empty corpus). Scores are
    max-normalised to [0, 1], descending."""

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("bm25.unavailable reason=rank_bm25 not installed")
        return None

    with _lock:
        cached = _cache.get(rag_id)
    if cached is None:
        built = await _build(db, rag_id)
        if built is None:
            return None
        with _lock:
            # Another coroutine may have built it while we queried —
            # either index is valid; keep whichever landed first.
            _cache.setdefault(rag_id, built)
        cached = _cache.get(rag_id)

    from app.services.tokenization import tokenize

    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = cached.bm25.get_scores(query_tokens)
    ranked = sorted(
        zip(cached.order, scores), key=lambda pair: pair[1], reverse=True
    )[:top_k]
    max_score = ranked[0][1] if ranked else 0.0

    hits: List[dict] = []
    for chunk_id, raw in ranked:
        meta = cached.meta[chunk_id]
        hits.append(
            {
                "chunk_id": chunk_id,
                "document_id": meta["document_id"],
                "filename": meta["filename"],
                "chunk_text": meta["chunk_text"],
                "score": raw / max_score if max_score > 0 else 0.0,
                "page_number": meta["page_number"],
                "metadata": meta["metadata"],
            }
        )
    return hits


async def _build(db: AsyncSession, rag_id: str) -> Optional[_RagIndex]:
    from app.models import Document, DocumentChunk
    from app.services.tokenization import tokenize

    rows = (
        await db.execute(
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.chunk_text,
                DocumentChunk.tokenized,
                DocumentChunk.page_number,
                DocumentChunk.metadata_json,
                Document.filename,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.rag_id == rag_id,
                Document.superseded_by.is_(None),
            )
        )
    ).all()
    if not rows:
        return None

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("bm25.unavailable reason=rank_bm25 not installed")
        return None

    order: List[str] = []
    corpus: List[List[str]] = []
    meta: Dict[str, dict] = {}
    for cid, document_id, text, tokenized, page, metadata, filename in rows:
        tokens = (tokenized or "").split() or tokenize(text or "")
        order.append(cid)
        corpus.append(tokens)
        meta[cid] = {
            "document_id": document_id,
            "filename": filename,
            "chunk_text": text,
            "page_number": page,
            "metadata": metadata,
        }

    bm25 = BM25Okapi(corpus)
    logger.info("bm25.index.built rag_id=%s chunks=%d", rag_id, len(corpus))
    return _RagIndex(bm25, order, meta)
