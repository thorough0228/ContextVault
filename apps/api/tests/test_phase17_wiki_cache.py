"""Phase 17: wiki hot cache — read-through, invalidation, degradation.

The cache is exercised through a stub backend (dict-style async
get/set/delete) — no real Redis in tests, per the conftest
determinism contract. ``WIKI_CACHE_ENABLED`` is pinned off there;
these tests opt in per-test via monkeypatch.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.llm.hash_provider import HashLLMProvider
from app.models import Chunk, Document, WikiPage
from app.services import wiki_cache, wiki_service

pytestmark = pytest.mark.offline


class _FakeBackend:
    """Dict-backed stub with an optional failure switch."""

    def __init__(self) -> None:
        self.store: dict = {}
        self.fail = False
        self.sets = 0
        self.deletes = 0

    async def get(self, key: str):
        if self.fail:
            raise RuntimeError("redis down")
        return self.store.get(key)

    async def set(self, key: str, value: str, *, ex: int):
        if self.fail:
            raise RuntimeError("redis down")
        self.store[key] = value
        self.sets += 1

    async def delete(self, key: str):
        if self.fail:
            raise RuntimeError("redis down")
        self.store.pop(key, None)
        self.deletes += 1


class _SummaryLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield "cached summary body\n\n## Key Points\n- marker-z"


def _enable_cache(monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend) -> None:
    monkeypatch.setenv("WIKI_CACHE_ENABLED", "true")
    monkeypatch.setenv("WIKI_CACHE_TTL_SECONDS", "300")
    get_settings.cache_clear()
    wiki_cache.set_cache_for_tests(backend)


def _disable_cache() -> None:
    wiki_cache.reset_cache_for_tests()
    get_settings.cache_clear()


async def _seed_doc_with_page(db_session, *, rag_id="r1", doc_id="d1", filename="doc.txt"):
    doc = Document(
        id=doc_id, rag_id=rag_id, filename=filename,
        file_type="txt", file_size=10, storage_key="k",
        status="READY", content_hash="h1",
    )
    db_session.add(doc)
    db_session.add(
        Chunk(
            document_id=doc.id, rag_id=doc.rag_id,
            page_number=1, chunk_index=0, text="body text",
        )
    )
    await db_session.commit()
    page = await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_SummaryLLM()
    )
    assert page is not None
    return doc


async def test_load_pages_miss_then_hit(client_with_db, db_session, monkeypatch):
    backend = _FakeBackend()
    _enable_cache(monkeypatch, backend)
    try:
        await _seed_doc_with_page(db_session)

        pages = await wiki_cache.load_pages(db_session, rag_id="r1")
        assert len(pages) == 1
        assert backend.sets == 1

        again = await wiki_cache.load_pages(db_session, rag_id="r1")
        assert again == pages
        assert backend.sets == 1, "second read must come from the cache"
    finally:
        _disable_cache()


async def test_invalidate_forces_refill(client_with_db, db_session, monkeypatch):
    backend = _FakeBackend()
    _enable_cache(monkeypatch, backend)
    try:
        await _seed_doc_with_page(db_session)

        await wiki_cache.load_pages(db_session, rag_id="r1")
        await wiki_cache.invalidate("r1")
        assert not backend.store, "invalidate must evict"
        await wiki_cache.load_pages(db_session, rag_id="r1")
        assert backend.sets == 2, "post-invalidation read must refill"
    finally:
        _disable_cache()


async def test_degraded_backend_falls_back_to_db(client_with_db, db_session, monkeypatch):
    backend = _FakeBackend()
    backend.fail = True
    _enable_cache(monkeypatch, backend)
    try:
        await _seed_doc_with_page(db_session)

        pages = await wiki_cache.load_pages(db_session, rag_id="r1")
        assert len(pages) == 1, "DB fallback must still serve the pages"
        await wiki_cache.invalidate("r1")  # must not raise either
    finally:
        _disable_cache()


async def test_disabled_flag_never_touches_backend(client_with_db, db_session):
    backend = _FakeBackend()
    backend.store["wikicache:v1:r1"] = "[]"
    wiki_cache.set_cache_for_tests(backend)
    # WIKI_CACHE_ENABLED stays pinned off by conftest
    try:
        get_settings.cache_clear()
        await _seed_doc_with_page(db_session)
        pages = await wiki_cache.load_pages(db_session, rag_id="r1")
        assert len(pages) == 1, "flag off → plain DB path"
    finally:
        _disable_cache()


async def test_mark_stale_invalidates_cache(client_with_db, db_session, monkeypatch):
    backend = _FakeBackend()
    _enable_cache(monkeypatch, backend)
    try:
        doc = await _seed_doc_with_page(db_session)

        await wiki_cache.load_pages(db_session, rag_id="r1")
        assert backend.store, "precondition: cache populated"

        doc.content_hash = "h2"
        await db_session.commit()
        await wiki_service.mark_stale_for_document(db_session, document_id=doc.id)
        assert not backend.store, "stale marking must evict the rag's cache"
    finally:
        _disable_cache()


async def test_document_delete_prunes_page(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    _enable_cache(monkeypatch, _FakeBackend())
    try:
        headers = await auth_headers_factory("delcache@example.com")
        rid = (
            await client_with_db.post(
                "/api/v1/rags", headers=headers, json={"name": "delcache"}
            )
        ).json()["id"]
        up = await client_with_db.post(
            f"/api/v1/rags/{rid}/documents",
            headers=headers,
            files={"file": ("gone.txt", b"delete me body", "text/plain")},
        )
        assert up.status_code == 201, up.text
        doc_id = up.json()["id"]

        # Phase 19: ingest no longer auto-generates wiki pages — seed
        # one explicitly so the delete path has something to prune
        # (commit: an open write txn on db_session would lock the
        # request's DELETE against the file-backed SQLite)
        doc = (
            await db_session.execute(
                select(Document).where(Document.id == doc_id)
            )
        ).scalars().first()
        page = await wiki_service.generate_doc_summary_page(
            db_session, doc=doc, llm=_SummaryLLM()
        )
        assert page is not None, "seeded page must exist"
        await db_session.commit()

        resp = await client_with_db.delete(
            f"/api/v1/documents/{doc_id}", headers=headers
        )
        assert resp.status_code in (200, 204), resp.text

        remaining = (
            await db_session.execute(
                select(WikiPage).where(
                    WikiPage.rag_id == rid,
                    WikiPage.kind == wiki_service.KIND_DOC_SUMMARY,
                )
            )
        ).scalars().all()
        assert remaining == [], "the deleted document's page must not survive"
    finally:
        _disable_cache()
