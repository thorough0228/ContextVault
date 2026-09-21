"""Search service: rag-scoped, user-isolated vector retrieval.

Two execution paths, picked at runtime based on the database dialect:

* **PostgreSQL** — push ``rag_id`` filter and ``<=>`` (cosine
  distance) ordering into SQL. Uses pgvector's HNSW index from
  migration ``0004_phase4_pgvector``.
* **SQLite** — pull all rows for the rag and rank them in-Python
  (sqlite doesn't have pgvector). Fine for the test suite and the
  hundreds-of-rows-per-rag scale.

Both paths enforce the same rule: **filter by ``rag_id`` and verify
``rag_id`` belongs to the caller before running the query.** A bug
that drops the filter cannot leak other users' chunks because the
service refuses to run without it.
"""

from __future__ import annotations

import logging
import math
from typing import List

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.embedding import (
    EmbeddingPermanentError,
    EmbeddingProvider,
    EmbeddingTransientError,
    get_embedding_provider,
)
from app.models import Document, DocumentChunk
from app.services import rag_service

logger = logging.getLogger(__name__)


class SearchError(Exception):
    """Wraps provider / DB errors with a stable code for the router."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def search_rag(
    db: AsyncSession,
    *,
    user_id: str,
    rag_id: str,
    query: str,
    top_k: int,
    provider: EmbeddingProvider | None = None,
) -> List[dict]:
    """Top-k semantic search inside one RAG owned by ``user_id``.

    Returns a list of dicts ready for :class:`SearchHit`:

    ``{chunk_id, document_id, filename, chunk_text, score,
    page_number, metadata}``

    Raises :class:`SearchError` with codes:

    * ``empty_query`` — caller-supplied query is blank.
    * ``rag_not_found`` — rag doesn't exist or belongs to another user.
    * ``embedding_failed`` — provider raised permanent or transient error.
    * ``search_failed`` — DB query failed.
    """
    if not query or not query.strip():
        raise SearchError("empty_query", "query must be non-empty")

    # Mandatory boundary check — same as the documents router.
    await rag_service.get_user_rag(db, user_id=user_id, rag_id=rag_id)

    settings = get_settings()
    effective_top_k = top_k if top_k > 0 else settings.search_default_top_k
    effective_top_k = min(effective_top_k, settings.search_max_top_k)

    provider = provider or get_embedding_provider()
    try:
        query_vec = provider.embed_text(query)
    except EmbeddingPermanentError as exc:
        raise SearchError("embedding_failed", str(exc)) from exc
    except EmbeddingTransientError as exc:
        # Transient — caller (the route) treats this as 503 + retry.
        raise SearchError("embedding_transient", str(exc)) from exc

    if len(query_vec) != provider.dimension:
        raise SearchError(
            "embedding_failed",
            f"provider returned {len(query_vec)} dims, expected {provider.dimension}",
        )

    bind = db.get_bind()
    dialect = bind.dialect.name if bind is not None else ""
    if settings.search_mode == "hybrid":
        return await _hybrid_search(
            db,
            query=query,
            rag_id=rag_id,
            top_k=effective_top_k,
            query_vec=query_vec,
            dialect=dialect,
        )
    if dialect == "postgresql":
        return await _search_pgvector(
            db, rag_id=rag_id, query_vec=query_vec, top_k=effective_top_k
        )
    return await _search_python(
        db, rag_id=rag_id, query_vec=query_vec, top_k=effective_top_k
    )


# ---------------------------------------------------------------------------
# Hybrid search (Phase 12): vector + keyword fusion + optional rerank
# ---------------------------------------------------------------------------


def _rrf_fuse(
    primary: List[dict],
    secondary: List[dict],
    *,
    k: int,
    top_k: int,
) -> List[dict]:
    """Reciprocal-rank fusion of two hit lists (same hit-dict shape).

    score(d) = Σ 1 / (k + rank_i(d)) over the lists containing d, then
    max-normalised to [0, 1] so the wire-format score contract
    (descending, in range) is preserved. Documents absent from one
    list simply contribute only the other list's term.
    """

    by_id: dict[str, dict] = {}
    for hits in (primary, secondary):
        for rank, hit in enumerate(hits, start=1):
            by_id.setdefault(hit["chunk_id"], hit)

    fused: list[tuple[float, dict]] = []
    for chunk_id, hit in by_id.items():
        score = 0.0
        for hits in (primary, secondary):
            ranks = {h["chunk_id"]: r for r, h in enumerate(hits, start=1)}
            if chunk_id in ranks:
                score += 1.0 / (k + ranks[chunk_id])
        fused.append((score, hit))

    fused.sort(key=lambda pair: pair[0], reverse=True)
    top = fused[:top_k]
    max_score = top[0][0] if top else 0.0
    out: List[dict] = []
    for score, hit in top:
        merged = dict(hit)
        merged["score"] = score / max_score if max_score > 0 else 0.0
        out.append(merged)
    return out


async def _hybrid_search(
    db: AsyncSession,
    *,
    query: str,
    rag_id: str,
    top_k: int,
    query_vec: List[float],
    dialect: str,
) -> List[dict]:
    """Vector + keyword RRF fusion, then an optional reranker pass.

    The reranker is best-effort: any provider failure logs a warning
    and keeps the fused ranking (hybrid retrieval never fails because
    the precision pass is unavailable).
    """

    settings = get_settings()
    candidates = max(top_k * max(1, settings.search_hybrid_candidate_multiplier), top_k)

    bind = db.get_bind()
    dialect = bind.dialect.name if bind is not None else ""
    if dialect == "postgresql":
        vector_hits = await _search_pgvector(
            db, rag_id=rag_id, query_vec=query_vec, top_k=candidates
        )
    else:
        vector_hits = await _search_python(
            db, rag_id=rag_id, query_vec=query_vec, top_k=candidates
        )
    keyword_hits = await _keyword_hits(db, rag_id=rag_id, query=query, top_k=candidates)

    fused = _rrf_fuse(
        vector_hits, keyword_hits, k=settings.search_hybrid_rrf_k, top_k=candidates
    )

    if settings.search_rerank_enabled and fused:
        from app.rerank.base import RerankPermanentError, RerankTransientError
        from app.rerank.factory import get_rerank_provider

        window = fused[: max(top_k, settings.search_rerank_candidates)]
        try:
            scores = get_rerank_provider().rerank(
                query, [h["chunk_text"] for h in window]
            )
        except (RerankPermanentError, RerankTransientError) as exc:
            logger.warning("search.rerank_degraded reason=%s", str(exc)[:120])
        else:
            reranked = []
            for hit, raw in sorted(
                zip(window, scores), key=lambda pair: pair[1], reverse=True
            ):
                out = dict(hit)
                # Map the raw logit to (0, 1) — sigmoid keeps ordering
                # and satisfies the score-range contract.
                out["score"] = 1.0 / (1.0 + math.exp(-raw))
                reranked.append(out)
            return reranked[:top_k]

    return fused[:top_k]


async def _keyword_hits(
    db: AsyncSession, *, rag_id: str, query: str, top_k: int
) -> List[dict]:
    """Keyword leg of hybrid search.

    Primary: BM25 (rank_bm25) over a per-RAG cached index built from
    the persisted tokenized column — proper IDF/length-normalised
    scoring across the whole RAG corpus. Falls back to the dialect
    recall paths (PG tsvector recall / SQLite token overlap) when
    BM25 cannot serve (dependency missing, empty corpus, cold build
    failure).
    """

    settings = get_settings()
    if settings.search_bm25_enabled:
        try:
            from app.services import bm25_service

            hits = await bm25_service.bm25_keyword_hits(
                db, rag_id=rag_id, query=query, top_k=top_k
            )
            if hits is not None:
                return hits
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "search.bm25_degraded rag_id=%s err=%s", rag_id, str(exc)[:120]
            )

    bind = db.get_bind()
    dialect = bind.dialect.name if bind is not None else ""
    if dialect == "postgresql":
        return await _keyword_pgvector(
            db, rag_id=rag_id, query=query, top_k=top_k
        )
    return await _keyword_python(
        db, rag_id=rag_id, query=query, top_k=top_k
    )


async def _keyword_pgvector(
    db: AsyncSession,
    *,
    rag_id: str,
    query: str,
    top_k: int,
) -> List[dict]:
    """Keyword leg: tsquery OR over the tokenized column, ranked by
    ts_rank_cd. Returns an empty list when the query tokenizes to
    nothing."""

    from app.services.tokenization import tokenize

    tokens = tokenize(query)
    if not tokens:
        return []
    tsquery = " | ".join(tokens)
    sql = text(
        """
        SELECT
            dc.id            AS chunk_id,
            dc.document_id   AS document_id,
            d.filename       AS filename,
            dc.chunk_text    AS chunk_text,
            dc.page_number   AS page_number,
            dc.metadata      AS metadata,
            ts_rank_cd(to_tsvector('simple', dc.tokenized),
                       to_tsquery('simple', :tsq)) AS rank
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE dc.rag_id = :rag_id
          AND d.superseded_by IS NULL
          AND dc.tokenized IS NOT NULL
          AND to_tsvector('simple', dc.tokenized) @@ to_tsquery('simple', :tsq)
        ORDER BY rank DESC
        LIMIT :top_k
        """
    )
    rows = await db.execute(
        sql, {"tsq": tsquery, "rag_id": rag_id, "top_k": top_k}
    )
    hits: List[dict] = []
    for row in rows.mappings():
        rank = float(row["rank"] or 0.0)
        hits.append(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_text": row["chunk_text"],
                # ts_rank_cd has no fixed ceiling — normalise per-list
                # so both legs contribute comparably to the RRF ranks.
                "score": rank,
                "page_number": row["page_number"],
                "metadata": row["metadata"],
            }
        )
    max_rank = max((h["score"] for h in hits), default=0.0)
    if max_rank > 0:
        for h in hits:
            h["score"] = h["score"] / max_rank
    return hits


async def _keyword_python(
    db: AsyncSession,
    *,
    rag_id: str,
    query: str,
    top_k: int,
) -> List[dict]:
    """SQLite keyword leg: token-overlap scoring over the rag's rows."""

    from app.services.tokenization import tokenize

    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []
    rows = await db.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.rag_id == rag_id,
            Document.superseded_by.is_(None),
        )
        .limit(2000)
    )
    scored: List[dict] = []
    for dc, filename in rows.all():
        doc_tokens = set(tokenize(dc.chunk_text))
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            continue
        scored.append(
            {
                "chunk_id": dc.id,
                "document_id": dc.document_id,
                "filename": filename,
                "chunk_text": dc.chunk_text,
                "score": overlap / len(query_tokens),
                "page_number": dc.page_number,
                "metadata": dc.metadata_json,
            }
        )
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]


async def _search_pgvector(
    db: AsyncSession,
    *,
    rag_id: str,
    query_vec: List[float],
    top_k: int,
) -> List[dict]:
    """Native pgvector cosine distance, ordered ASC (smaller = closer).

    ``vector(N) <=> vector(N)`` returns cosine distance in [0, 2].
    The client sees similarity = 1 - distance/2 mapped to [0, 1] so
    "higher is better" matches the JSON API.
    """
    sql = text(
        """
        SELECT
            dc.id            AS chunk_id,
            dc.document_id   AS document_id,
            d.filename       AS filename,
            dc.chunk_text    AS chunk_text,
            dc.page_number   AS page_number,
            dc.metadata      AS metadata,
            dc.embedding <=> CAST(:qv AS vector) AS distance
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE dc.rag_id = :rag_id
          AND dc.embedding IS NOT NULL
          AND d.superseded_by IS NULL
        ORDER BY distance ASC
        LIMIT :top_k
        """
    )
    # asyncpg has no codec for the vector type — bind the pgvector text
    # literal and let the CAST coerce it. (A Python list cannot be bound
    # here, contrary to what an earlier comment claimed.)
    qv = "[" + ",".join(repr(float(x)) for x in query_vec) + "]"
    rows = await db.execute(
        sql,
        {"qv": qv, "rag_id": rag_id, "top_k": top_k},
    )
    hits: List[dict] = []
    for row in rows.mappings():
        distance = float(row["distance"])
        # Cosine distance in [0, 2]; map to similarity in [0, 1] with
        # higher-is-better so the wire format matches the SQLite branch.
        similarity = max(0.0, 1.0 - distance / 2.0)
        hits.append(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_text": row["chunk_text"],
                "score": similarity,
                "page_number": row["page_number"],
                "metadata": row["metadata"],
            }
        )
    return hits


async def _search_python(
    db: AsyncSession,
    *,
    rag_id: str,
    query_vec: List[float],
    top_k: int,
) -> List[dict]:
    """SQLite fallback: pull all rows for the rag and rank in Python.

    Pull is bounded by ``top_k * 50`` candidates so a rag with 100k
    chunks doesn't OOM the worker. For real search traffic on SQLite,
    swap to a smaller test corpus or run on PostgreSQL.
    """
    # Soft cap on candidates pulled into memory: 50× top_k.
    candidate_cap = max(50, top_k * 50)
    rows = await db.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.rag_id == rag_id,
            Document.superseded_by.is_(None),
        )
        .limit(candidate_cap)
    )
    candidates: List[dict] = []
    for dc, filename in rows.all():
        if not dc.embedding:
            continue
        # Map raw cosine [-1, 1] to [0, 1] so the wire format
        # matches the pgvector branch (which also normalises).
        raw = _cosine(query_vec, list(dc.embedding))
        score = (raw + 1.0) / 2.0
        candidates.append(
            {
                "chunk_id": dc.id,
                "document_id": dc.document_id,
                "filename": filename,
                "chunk_text": dc.chunk_text,
                "score": score,
                "page_number": dc.page_number,
                "metadata": dc.metadata_json,
            }
        )
    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates[:top_k]


def _cosine(a: List[float], b: List[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom