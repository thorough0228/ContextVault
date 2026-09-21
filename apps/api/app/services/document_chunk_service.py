"""DocumentChunk service: idempotent upsert + retrieval helpers.

The Phase 4 worker uses :func:`replace_document_chunks` to mirror the
Phase 3 chunking output into the searchable index. Search uses
:func:`cosine_search` for SQLite (in-Python) and the SQL-native
``<=>`` operator for PostgreSQL — selection happens in
``search_service.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Sequence, Tuple

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document, DocumentChunk
from app.services.tokenization import tokenize_to_string

logger = logging.getLogger(__name__)


async def replace_document_chunks(
    db: AsyncSession,
    *,
    document_id: str,
    rag_id: str,
    chunks: Sequence[Chunk],
    embeddings: Sequence[List[float]],
) -> int:
    """Replace the searchable index for ``document_id`` with one row per
    chunk. Idempotent — re-running the worker against the same
    document yields the same end state.

    Returns the number of rows inserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
        )

    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    await db.flush()

    rows: list[DocumentChunk] = []
    for chunk, vector in zip(chunks, embeddings):
        # ``chunk`` is the dataclass ``Chunk`` produced by the chunker
        # (see ``app.ingest.chunk_pages``), which exposes ``metadata``
        # as the attribute name — not the SQLAlchemy ORM ``Chunk`` row's
        # ``metadata_json``. The ORM row maps the JSON column under
        # ``metadata_json`` so the persistence side stays consistent.
        rows.append(
            DocumentChunk(
                document_id=document_id,
                rag_id=rag_id,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
                page_number=chunk.page_number,
                metadata_json=chunk.metadata,
                tokenized=tokenize_to_string(chunk.text),
                embedding=list(vector),
            )
        )
    if rows:
        db.add_all(rows)
        await db.flush()
    return len(rows)


async def delete_document_chunks_for_document(
    db: AsyncSession, *, document_id: str
) -> int:
    """Used by the document delete router when a Document row is
    removed — DocumentChunk rows cascade on the FK, but we keep this
    helper available for explicit cleanup jobs."""
    result = await db.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    return int(result.rowcount or 0)


async def iter_rag_chunks(
    db: AsyncSession, *, rag_id: str
) -> Iterable[DocumentChunk]:
    """Stream all searchable chunks for one RAG — used by the in-Python
    cosine branch when running on SQLite."""
    result = await db.scalars(
        select(DocumentChunk).where(DocumentChunk.rag_id == rag_id)
    )
    return result.all()


async def list_rag_chunks(
    db: AsyncSession, *, rag_id: str
) -> List[DocumentChunk]:
    rows = await db.scalars(
        select(DocumentChunk).where(DocumentChunk.rag_id == rag_id)
    )
    return list(rows.all())


# ---- cosine similarity (in-Python, SQLite-friendly) -----------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]. Embedding providers are expected to
    return L2-normalised vectors so the denominator is 1 and this is
    just a dot product."""
    denom = math.sqrt(_dot(a, a) * _dot(b, b))
    if denom == 0:
        return 0.0
    return _dot(a, b) / denom


def rank_by_cosine(
    query_vec: Sequence[float],
    candidates: Sequence[Tuple[str, str, str, int, str, dict | None, list[float]]],
    *,
    top_k: int,
) -> List[Tuple[float, str, str, str, int, str, dict | None]]:
    """Pure-Python cosine ranking.

    ``candidates`` is a sequence of tuples in the same order as
    :class:`DocumentChunk`'s searchable columns
    (chunk_id, document_id, filename, page_number, chunk_text, metadata, embedding).
    Returns hits sorted by descending similarity, capped at ``top_k``.
    """
    scored: List[Tuple[float, str, str, str, int, str, dict | None]] = []
    for chunk_id, document_id, filename, page_number, chunk_text, meta, vec in candidates:
        if not vec:
            continue
        score = cosine_similarity(query_vec, vec)
        scored.append(
            (score, chunk_id, document_id, filename, page_number, chunk_text, meta)
        )
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored[:top_k]