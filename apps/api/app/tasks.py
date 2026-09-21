"""Celery tasks that both the API and the worker import.

Phase 3 ships ``process_document``: download a freshly uploaded
object, parse it, chunk it, persist chunks, and update the Document
status. The task uses ``@shared_task`` so it binds to whichever
Celery app is active in the importing process (the API only sends
``.delay()``; the worker actually runs the body).

Phase 4 extends ``process_document`` so the pipeline runs to completion:

    upload → parse → chunk → embed → write document_chunks → READY

Idempotency: re-running the task against the same ``document_id``
deletes pre-existing chunks before inserting the new list. The
status flip ``CREATED/PROCESSING → READY/FAILED`` is safe to repeat.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import List

from celery import shared_task

from app.celery_client import celery_app as _celery_app
from app.config import get_settings
from app.embedding import (
    EmbeddingPermanentError,
    EmbeddingProvider,
    EmbeddingTransientError,
    get_embedding_provider,
)
from app.ingest import chunk_pages, parse_document
from app.ingest.file_type import FileTypeError
from app.models import Document
from app.services import document_chunk_service, document_service
from app.storage import get_storage
from app.db import dispose_engine, get_sessionmaker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)


class _EmbeddingFailedPermanently(Exception):
    """Internal marker — the provider raised :class:`EmbeddingPermanentError`.
    We translate this into a doc-level ``FAILED`` status."""


class _EmbeddingFailedTransently(Exception):
    """Internal marker — the provider raised :class:`EmbeddingTransientError`
    and we want the task harness to schedule a Celery retry."""


async def _embed_chunks_with_retry(
    provider: EmbeddingProvider,
    chunks,
    *,
    document_id: str,
) -> List[List[float]]:
    """Batched embedding with internal retry on transient errors.

    Translates provider errors into our internal markers so the task
    harness can distinguish "permanently broken document" from
    "transient provider hiccup".
    """
    settings = get_settings()
    batch_size = max(1, settings.embedding_batch_size)
    max_chars = settings.embedding_max_chars_per_input

    out: List[List[float]] = []
    texts: List[str] = [c.text for c in chunks]

    # Defensive truncation — single inputs longer than the provider's
    # context window are a permanent failure (we won't silently split).
    for text in texts:
        if len(text) > max_chars:
            raise _EmbeddingFailedPermanently(
                f"chunk length {len(text)} > limit {max_chars}"
            )

    # Reject empty chunk text — the provider would either fail or
    # return an all-zero vector that pollutes the index.
    if not texts:
        return []
    if any(not t.strip() for t in texts):
        raise _EmbeddingFailedPermanently("empty chunk text encountered")

    try:
        # One call when small; otherwise batch.
        if len(texts) <= batch_size:
            embeddings = list(provider.embed_texts(texts))
        else:
            # Manual batching — providers handle one chunk at a time if
            # they don't override embed_texts.
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                out.extend(provider.embed_texts(batch))
            embeddings = out
    except EmbeddingPermanentError as exc:
        logger.warning(
            "process_document.embedding_permanent document_id=%s err=%s",
            document_id, exc,
        )
        raise _EmbeddingFailedPermanently(str(exc)) from exc
    except EmbeddingTransientError as exc:
        logger.warning(
            "process_document.embedding_transient document_id=%s err=%s",
            document_id, exc,
        )
        raise _EmbeddingFailedTransently(str(exc)) from exc

    # Guard: dimensions must match the configured provider dimension.
    # Catching this here means a buggy provider fails one document
    # rather than corrupting the index with mixed-dim vectors.
    expected = provider.dimension
    for i, vec in enumerate(embeddings):
        if not isinstance(vec, list) or len(vec) != expected:
            raise _EmbeddingFailedPermanently(
                f"provider returned {len(vec) if hasattr(vec, '__len__') else '?'} dims "
                f"at index {i}, expected {expected}"
            )
    return embeddings


def _transient(exc: Exception) -> bool:
    """Heuristic: which exceptions should trigger a Celery retry?

    Anything embedding- or storage-flavored that's not a permanent
    config bug gets a retry. Permanent errors include malformed
    documents, missing users, dimension mismatches, and explicit
    ``BadRequestError`` / ``FileTypeError`` raises.
    """
    from app.exceptions import BadRequestError, NotFoundError

    if isinstance(exc, (BadRequestError, FileTypeError, NotFoundError)):
        return False
    if isinstance(exc, EmbeddingPermanentError):
        return False
    name = exc.__class__.__name__
    if name in {"FileDataError", "EmptyFileError", "ValueError"}:
        return False
    if isinstance(exc, EmbeddingTransientError):
        return True
    return True


async def _run_with_session(document_id: str, fn) -> None:
    """Open a fresh session, run ``fn(session)``, commit/rollback."""
    # The celery app's ``database_url`` is set at boot from the live
    # Settings. Tests may override it via the app's config — see
    # ``tests/conftest.py``.
    url = getattr(_celery_app.conf, "database_url", None) or get_settings().async_database_url
    engine = create_async_engine(url, future=True)
    sm = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    try:
        async with sm() as session:
            try:
                await fn(session)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    finally:
        await engine.dispose()


def _call_async(coro) -> None:
    """Run a coroutine either in a fresh event loop or in a worker
    thread if the caller already has one (e.g. inside a pytest-asyncio
    test)."""
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        asyncio.run(coro)
        return

    # Already inside a running loop — schedule on a fresh loop in a
    # worker thread so the test event loop isn't disturbed.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(asyncio.run, coro)
        future.result()


@shared_task(
    name="app.tasks.process_document",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def process_document(self, document_id: str, on_ready_supersede: str | None = None) -> dict:
    """End-to-end ingestion for one document.

    Steps:
    1. Look up the Document. If missing → permanent fail (logged + ack).
    2. Mark PROCESSING.
    3. Download the original from object storage.
    4. Parse by file_type.
    5. Chunk with the configured window/overlap.
    6. Replace existing chunks (idempotency).
    7. Mark READY (or FAILED + error_message).

    Retry policy: transient errors (network, DB) → ``self.retry(...)``
    with exponential backoff up to ``max_retries``. Permanent errors
    (bad file, missing object) → ``FAILED`` and we move on.
    """
    # Phase 6: pick up the X-Request-Id the API set on the task
    # headers (when present) and push it into the worker's
    # ContextVar so every log line in this task carries the same
    # correlation id. Always reset in finally so the worker process
    # doesn't leak the id into the next task.
    from app.middleware.request_id import (
        set_request_id as _set_rid,
        reset_request_id as _reset_rid,
    )
    inbound_rid = (
        (self.request.headers or {}).get("request_id")
        if hasattr(self.request, "headers")
        else None
    )
    rid_token = _set_rid(inbound_rid) if inbound_rid else None
    try:
        return _process_document_impl(self, document_id, on_ready_supersede)
    finally:
        if rid_token is not None:
            _reset_rid(rid_token)


def _process_document_impl(self, document_id: str, on_ready_supersede: str | None = None) -> dict:
    """Inner body of :func:`process_document` — kept separate so the
    request-id context is always reset even on early returns."""

    async def _do(session: AsyncSession) -> None:
        settings = get_settings()
        doc = await session.get(Document, document_id)
        if doc is None:
            logger.warning(
                "process_document.missing document_id=%s attempt=%s",
                document_id, self.request.retries + 1,
            )
            return

        await document_service.mark_processing(session, document_id=document_id)

        storage = get_storage()
        try:
            obj = storage.download(key=doc.storage_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "process_document.download_failed document_id=%s err=%s",
                document_id, exc,
            )
            if _transient(exc):
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"download failed: {exc}",
            )
            return

        try:
            pages = parse_document(
                file_type=doc.file_type,
                stream=io.BytesIO(obj.body),
            )
        except FileTypeError as exc:
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"invalid file: {exc}",
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "process_document.parse_failed document_id=%s err=%s",
                document_id, exc,
            )
            if _transient(exc):
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"parse failed: {exc}",
            )
            return

        try:
            chunks = chunk_pages(
                pages,
                document_id=document_id,
                rag_id=doc.rag_id,
                size=settings.chunk_size_chars,
                overlap=settings.chunk_overlap_chars,
            )
        except Exception as exc:  # noqa: BLE001
            if _transient(exc):
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"chunk failed: {exc}",
            )
            return

        try:
            chunk_count = await document_service.replace_chunks(
                session,
                document_id=document_id,
                rag_id=doc.rag_id,
                chunks=chunks,
            )
        except Exception as exc:  # noqa: BLE001
            if _transient(exc):
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"persist failed: {exc}",
            )
            return

        # ---- Phase 4: embedding + vector storage ------------------------
        # Embedding happens AFTER chunks are persisted so we know the
        # count is stable. If embedding fails permanently the doc
        # goes FAILED; transient errors retry with exponential backoff.
        embeddings: List[List[float]] | None = None
        try:
            provider = get_embedding_provider()
            embeddings = await _embed_chunks_with_retry(
                provider, chunks, document_id=document_id,
            )
        except _EmbeddingFailedPermanently as exc:
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"embedding failed: {exc}",
            )
            return
        except _EmbeddingFailedTransently as exc:
            # Schedule a Celery retry; the transient mark is done
            # by the worker harness.
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        except Exception as exc:  # noqa: BLE001
            if _transient(exc):
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"embedding failed: {exc}",
            )
            return

        if embeddings is None:
            # Empty document — nothing to embed but document is still
            # considered READY for search (will return []).
            embeddings = []

        try:
            await document_chunk_service.replace_document_chunks(
                session,
                document_id=document_id,
                rag_id=doc.rag_id,
                chunks=chunks,
                embeddings=embeddings,
            )
        except Exception as exc:  # noqa: BLE001
            if _transient(exc):
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            await document_service.mark_failed(
                session,
                document_id=document_id,
                error_message=f"vector store failed: {exc}",
            )
            return

        await document_service.mark_ready(
            session,
            document_id=document_id,
            page_count=len(pages),
            chunk_count=chunk_count,
        )

        # Hybrid search: the BM25 index for this RAG is now stale.
        from app.services import bm25_service

        bm25_service.invalidate(doc.rag_id)

        # Bluegreen update (Phase 13): the new document ingested
        # successfully — hide the old version from retrieval. Done in
        # the SAME transaction as mark_ready, so there is no window
        # where both versions are visible.
        if on_ready_supersede:
            old_doc = await session.get(Document, on_ready_supersede)
            if old_doc is not None:
                old_doc.superseded_by = document_id
                logger.info(
                    "process_document.superseded old=%s new=%s",
                    on_ready_supersede, document_id,
                )
            else:
                logger.warning(
                    "process_document.supersede_target_missing old=%s",
                    on_ready_supersede,
                )

    try:
        _call_async(_run_with_session(document_id, _do))
    except Exception as exc:  # noqa: BLE001
        # Last-resort safety net: if Celery didn't already convert this
        # to a retry, mark FAILED instead of letting the task go red.
        if _transient(exc):
            logger.exception(
                "process_document.transient document_id=%s", document_id,
            )
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        # Permanent — log and mark FAILED so the row stops bouncing.
        logger.exception(
            "process_document.permanent document_id=%s err=%s",
            document_id, exc,
        )
        # Schedule a small async best-effort FAILED update; if it
        # fails too, swallow it (nothing else we can do here).
        try:
            _call_async(
                _run_with_session(
                    document_id,
                    lambda s: document_service.mark_failed(
                        s,
                        document_id=document_id,
                        error_message=f"{exc.__class__.__name__}: {exc}",
                    ),
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("process_document.mark_failed.best_effort_failed")
        return {
            "document_id": document_id,
            "status": "FAILED",
            "error": str(exc),
        }

    return {
        "document_id": document_id,
        "status": "READY",
        "finished_at": datetime.now(tz=timezone.utc).isoformat(),
    }

@shared_task(
    name="app.tasks.extract_memory",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def extract_memory(
    self,
    *,
    conversation_id: str,
    user_id: str,
    rag_id: str,
    user_message: str,
    assistant_message: str,
) -> dict:
    """Phase 12 long-term memory: extract durable facts from one chat
    turn and store them for (user, rag).

    The LLM must answer with a JSON array of short fact strings; a
    non-JSON answer (e.g. from the deterministic hash provider used in
    tests) yields no facts and completes cleanly. Deduplication is by
    exact content within (user, rag)."""

    from app.llm import get_llm_provider
    from app.models import MemoryFact
    from app.config import get_settings as _settings

    settings = _settings()
    max_facts = max(1, settings.memory_extract_max_facts)

    async def _do(session) -> dict:
        llm = get_llm_provider()
        prompt = (
            "Extract up to "
            f"{max_facts} durable facts about the user from this chat turn "
            "(preferences, constraints, project background, explicit "
            "requests to remember). Skip small talk and anything already "
            "obvious from the question alone. Answer with ONLY a JSON "
            "array of short fact strings, e.g. "
            '["User prefers lightweight blades", ...]. If there is '
            "nothing worth remembering, answer with [].\n\n"
            f"User: {user_message[:2000]}\n\n"
            f"Assistant: {assistant_message[:2000]}"
        )
        raw = await llm.chat([{"role": "user", "content": prompt}])
        facts = _parse_fact_array(raw)
        if not facts:
            return {"conversation_id": conversation_id, "facts": 0}

        # Self-contained: the API side already verified ownership before
        # enqueuing, so the task trusts (user_id, rag_id) and does NOT
        # read the conversation — a fresh session here cannot see the
        # still-uncommitted stream transaction anyway.
        existing = set(
            await session.scalars(
                select(MemoryFact.content).where(
                    MemoryFact.user_id == user_id,
                    MemoryFact.rag_id == rag_id,
                )
            )
        )
        added = 0
        for fact in facts[:max_facts]:
            content = fact.strip()
            if not content or content in existing:
                continue
            session.add(
                MemoryFact(
                    user_id=user_id,
                    rag_id=rag_id,
                    content=content[:500],
                    source_conversation_id=conversation_id,
                )
            )
            existing.add(content)
            added += 1
        await session.commit()
        logger.info(
            "memory.extract.ok conversation_id=%s facts=%d", conversation_id, added
        )
        return {"conversation_id": conversation_id, "facts": added}

    try:
        _call_async(_run_with_session(conversation_id, _do))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "memory.extract.failed conversation_id=%s err=%s",
            conversation_id, str(exc)[:160],
        )
        raise self.retry(exc=exc)
    return {"conversation_id": conversation_id}


def _parse_fact_array(raw: str) -> List[str]:
    """Best-effort strict-JSON array extraction from the LLM answer."""
    import json as _json
    import re as _re

    text = (raw or "").strip()
    fence = _re.search(r"```(?:json)?\s*(.+?)```", text, _re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = _json.loads(text[start : end + 1])
    except _json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if str(x).strip()]
