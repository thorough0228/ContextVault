"""Wiki page hot cache (Phase 17) — read-through Redis, degrade-safe.

Caches each RAG's non-stale wiki pages as one JSON payload
(``wikicache:v1:{rag_id}``, TTL ``WIKI_CACHE_TTL_SECONDS``). Every
write path (summary upsert, staleness marking, document delete,
maintenance) calls :func:`invalidate`, so the TTL is only a safety net
for a lost DEL — cache/DB agreement is event-driven, not polled.

Failure posture: the cache is NEVER allowed to break chat. Any backend
error → ``wiki.cache degraded`` warning + the plain DB query (same
shape as the BM25/rerank degradations). The client is created lazily
on first use — import time must stay side-effect-free so tests and
evals boot without a Redis (the celery-client precedent).
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.metrics import Counter

logger = logging.getLogger(__name__)

_KEY_PREFIX = "wikicache:v1:"
# Match the health probe's budgets — a down Redis must cost each chat
# turn at most ~1.5s, not a full TCP retry storm.
_SOCKET_TIMEOUT_SECONDS = 1.5

# Test seam: when set, used instead of the real redis client.
_backend_override = None
_redis_client = None


def _key(rag_id: str) -> str:
    return f"{_KEY_PREFIX}{rag_id}"


def _get_backend():
    """Return the cache backend, or ``None`` when caching is off. The
    enabled flag gates everything — the test override only swaps the
    transport, it cannot re-enable a disabled cache."""

    global _redis_client
    from app.config import get_settings

    settings = get_settings()
    if not settings.wiki_cache_enabled:
        return None
    if _backend_override is not None:
        return _backend_override
    if _redis_client is None:
        import redis.asyncio

        _redis_client = redis.asyncio.Redis.from_url(
            settings.redis_url,
            socket_timeout=_SOCKET_TIMEOUT_SECONDS,
            socket_connect_timeout=_SOCKET_TIMEOUT_SECONDS,
        )
    return _redis_client


def set_cache_for_tests(backend) -> None:
    """Swap in a stub backend (dict-style async get/set/delete)."""
    global _backend_override
    _backend_override = backend


def reset_cache_for_tests() -> None:
    """Drop the stub AND the lazy redis client (test isolation)."""
    global _backend_override, _redis_client
    _backend_override = None
    _redis_client = None


async def load_pages(db: AsyncSession, *, rag_id: str) -> List[dict]:
    """Non-stale wiki pages for a rag — cache first, DB as fallback.

    Returned dicts carry ``slug``/``title``/``content``/``source_refs``
    /``notes`` (scoring is query-dependent and stays with the caller).
    Never raises.
    """

    backend = _get_backend()
    if backend is None:
        return await _query_pages(db, rag_id=rag_id)

    raw = None
    try:
        raw = await backend.get(_key(rag_id))
    except Exception as exc:  # noqa: BLE001
        _degraded(rag_id, exc)
        return await _query_pages(db, rag_id=rag_id)

    if raw is not None:
        try:
            pages = json.loads(raw)
        except ValueError:
            pages = None  # corrupted payload → treat as a miss
        if isinstance(pages, list):
            Counter("wiki_cache_total").inc(result="hit")
            logger.info("wiki.cache hit rag_id=%s pages=%d", rag_id, len(pages))
            return pages

    pages = await _query_pages(db, rag_id=rag_id)
    Counter("wiki_cache_total").inc(result="miss")
    logger.info("wiki.cache miss rag_id=%s pages=%d", rag_id, len(pages))
    try:
        from app.config import get_settings

        await backend.set(
            _key(rag_id),
            json.dumps(pages, ensure_ascii=False),
            ex=get_settings().wiki_cache_ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        _degraded(rag_id, exc)
    return pages


async def invalidate(rag_id: str) -> None:
    """Best-effort eviction after any wiki write. Never raises."""

    backend = _get_backend()
    if backend is None:
        return
    try:
        await backend.delete(_key(rag_id))
        logger.info("wiki.cache invalidated rag_id=%s", rag_id)
    except Exception as exc:  # noqa: BLE001
        _degraded(rag_id, exc)


async def _query_pages(db: AsyncSession, *, rag_id: str) -> List[dict]:
    from sqlalchemy import select

    from app.models import WikiPage

    rows = await db.scalars(
        select(WikiPage).where(
            WikiPage.rag_id == rag_id,
            WikiPage.stale.is_(False),
        )
    )
    return [
        {
            "slug": p.slug,
            "kind": p.kind,
            "title": p.title,
            "content": p.content,
            "source_refs": p.source_refs or [],
            "notes": p.notes,
            "meta": p.meta,
        }
        for p in rows.all()
    ]


def _degraded(rag_id: str, exc: Exception) -> None:
    Counter("wiki_cache_total").inc(result="degraded")
    logger.warning(
        "wiki.cache degraded rag_id=%s err=%s", rag_id, str(exc)[:120]
    )
