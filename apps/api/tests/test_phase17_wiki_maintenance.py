"""Phase 17/19: wiki auto-maintenance — orphan lint, contradictions.

(The Phase 17 rescan retired in Phase 19: the wiki is query-driven, so
a missing page is the normal state, not a fault — stale provenance is
downgraded live by the router instead.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from app.llm.hash_provider import HashLLMProvider
from app.models import Chunk, Document, WikiPage
from app.services import wiki_service

pytestmark = pytest.mark.offline

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"


class _SummaryLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield "summary body"


def _doc(db, *, doc_id, rag_id="r1", filename="doc.txt", status="READY", content_hash="h1"):
    doc = Document(
        id=doc_id, rag_id=rag_id, filename=filename,
        file_type="txt", file_size=10, storage_key="k",
        status=status, content_hash=content_hash,
    )
    db.add(doc)
    db.add(
        Chunk(
            document_id=doc_id, rag_id=rag_id,
            page_number=1, chunk_index=0, text="body text",
        )
    )
    return doc


def _page(db, *, rag_id="r1", slug, doc_id, content_hash="h1", stale=False, notes=None):
    page = WikiPage(
        id=f"wp-{slug}-{doc_id[:6]}", rag_id=rag_id, kind=wiki_service.KIND_DOC_SUMMARY,
        slug=slug, title=f"Summary: {slug}", content="page body",
        source_refs=[{"document_id": doc_id, "content_hash": content_hash, "filename": slug}],
        stale=stale, notes=notes,
    )
    db.add(page)
    return page


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


async def test_lint_deletes_dead_provenance_and_rename_leftovers(client_with_db, db_session):
    # live docs only — the "dead" provenance below points at a doc_id
    # with NO row (as after a real delete)
    _doc(db_session, doc_id="d-live", filename="live.txt")
    _doc(db_session, doc_id="d-other", filename="other.txt")
    await db_session.commit()

    # page whose ONLY source has no document row → delete
    _page(db_session, slug="gone.txt", doc_id="ghost-dead")
    # rename leftover: slug matches no live filename → delete
    _page(db_session, slug="old-name.txt", doc_id="d-live")
    # the live doc's own page → kept
    _page(db_session, slug="live.txt", doc_id="d-live")
    await db_session.commit()

    result = await wiki_service.lint_orphan_pages(db_session)
    assert result["deleted"] == 2

    remaining = (
        await db_session.execute(
            select(WikiPage).where(
                WikiPage.kind == wiki_service.KIND_DOC_SUMMARY
            )
        )
    ).scalars().all()
    assert [p.slug for p in remaining] == ["live.txt"]


async def test_lint_prunes_partial_dead_refs(client_with_db, db_session):
    _doc(db_session, doc_id="d1", filename="one.txt")
    await db_session.commit()
    # a page (multi-source shape) with one live + one dead ref
    page = WikiPage(
        id="wp-multi", rag_id="r1", kind=wiki_service.KIND_DOC_SUMMARY,
        slug="one.txt", title="Summary: one.txt", content="body",
        source_refs=[
            {"document_id": "d1", "content_hash": "h1", "filename": "one.txt"},
            {"document_id": "ghost", "content_hash": "hx", "filename": "gone.txt"},
        ],
        stale=False, notes=None,
    )
    db_session.add(page)
    await db_session.commit()

    result = await wiki_service.lint_orphan_pages(db_session)
    assert result["pruned"] == 1
    await db_session.refresh(page)
    assert [r["document_id"] for r in page.source_refs] == ["d1"]


# ---------------------------------------------------------------------------
# contradiction scan
# ---------------------------------------------------------------------------


class _ContradictionLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield (
            '```json\n[{"slug_a": "a-txt", "slug_b": "b-txt",'
            ' "summary": "thrust ratings disagree"}]\n```'
        )


class _GarbageLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield "I cannot comply, the pages look fine to me."


async def _two_pages(db_session):
    _doc(db_session, doc_id="d1", filename="a.txt")
    _doc(db_session, doc_id="d2", filename="b.txt")
    await db_session.commit()
    pa = _page(db_session, slug="a-txt", doc_id="d1")
    pb = _page(db_session, slug="b-txt", doc_id="d2")
    await db_session.commit()
    return pa, pb


async def test_contradictions_recorded_in_notes(client_with_db, db_session):
    pa, pb = await _two_pages(db_session)

    findings = await wiki_service.detect_contradictions(
        db_session, rag_id="r1", llm=_ContradictionLLM()
    )
    assert len(findings) == 1
    await db_session.refresh(pa)
    await db_session.refresh(pb)
    assert pa.notes["contradictions"][0]["summary"] == "thrust ratings disagree"
    assert pb.notes == pa.notes


async def test_contradictions_clear_notes_when_clean(client_with_db, db_session):
    pa, pb = await _two_pages(db_session)
    pa.notes = {"contradictions": [{"slug_a": "a-txt", "slug_b": "b-txt", "summary": "old"}]}

    class _EmptyLLM(HashLLMProvider):
        async def stream_chat(self, messages):
            yield "[]"

    findings = await wiki_service.detect_contradictions(
        db_session, rag_id="r1", llm=_EmptyLLM()
    )
    assert findings == []
    await db_session.refresh(pa)
    assert pa.notes is None, "a clean scan clears stale findings"


async def test_contradictions_parse_failure_leaves_notes_untouched(
    client_with_db, db_session
):
    pa, _pb = await _two_pages(db_session)
    pa.notes = {"contradictions": [{"summary": "previous finding"}]}
    await db_session.commit()

    findings = await wiki_service.detect_contradictions(
        db_session, rag_id="r1", llm=_GarbageLLM()
    )
    assert findings == []
    await db_session.refresh(pa)
    assert pa.notes == {"contradictions": [{"summary": "previous finding"}]}


# ---------------------------------------------------------------------------
# beat schedule registration
# ---------------------------------------------------------------------------


def test_beat_schedule_registered():
    """The worker's celery app must carry both Phase 17 beat entries."""
    import importlib
    import os
    import sys

    worker_dir = str(WORKER_DIR)
    api_dir = str(Path(__file__).resolve().parents[1])
    if api_dir in sys.path:
        sys.path.remove(api_dir)
    sys.path.insert(0, worker_dir)
    saved = os.environ.get("CELERY_BROKER_URL")
    os.environ["CELERY_BROKER_URL"] = "memory://"
    try:
        for mod in ("app", "app.celery_app"):
            sys.modules.pop(mod, None)
        mod = importlib.import_module("app.celery_app")
        schedule = mod.celery_app.conf.beat_schedule
        tasks = {v["task"] for v in schedule.values()}
        assert "app.tasks.wiki_maintenance" in tasks
        assert "app.tasks.wiki_contradiction_scan" in tasks
    finally:
        if saved is None:
            os.environ.pop("CELERY_BROKER_URL", None)
        else:
            os.environ["CELERY_BROKER_URL"] = saved
        for mod in ("app", "app.celery_app"):
            sys.modules.pop(mod, None)
        sys.path.remove(worker_dir)
        sys.path.insert(0, api_dir)
