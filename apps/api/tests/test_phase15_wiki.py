"""Phase 15: wiki layer tests — generation, staleness, routing."""

from __future__ import annotations

import json

import pytest

from app.llm.hash_provider import HashLLMProvider
from app.models import Document, WikiPage
from app.services import wiki_service
from app.services.chat_service import _build_prompt_messages
from sqlalchemy import select

pytestmark = pytest.mark.offline

DOC_FILENAME = "report.txt"


class _SummaryLLM(HashLLMProvider):
    """Hash LLM that emits a recognizable wiki summary."""

    async def stream_chat(self, messages):
        yield "WIKI SUMMARY CONTENT\n\n## Key Points\n- marker-x"


def _doc(doc_id="d1", rag_id="r1", filename=DOC_FILENAME, content_hash="h1"):
    return Document(
        id=doc_id, rag_id=rag_id, filename=filename,
        file_type="txt", file_size=100, storage_key="k",
        status="READY", content_hash=content_hash,
    )


async def _seed_doc(db, doc):
    """Insert the Document row + one Chunk row (the summary generator
    reads its body from the chunks table)."""
    from app.models import Chunk

    db.add(doc)
    db.add(Chunk(
        document_id=doc.id, rag_id=doc.rag_id,
        page_number=1, chunk_index=0,
        text=f"Body content of {doc.filename}: key facts and numbers.",
    ))
    await db.commit()


# ---------------------------------------------------------------------------
# wiki_service
# ---------------------------------------------------------------------------


async def test_generate_doc_summary_creates_page(client_with_db, db_session):
    doc = _doc()
    await _seed_doc(db_session, doc)

    page = await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_SummaryLLM()
    )
    assert page is not None
    assert page.kind == wiki_service.KIND_DOC_SUMMARY
    assert "WIKI SUMMARY CONTENT" in page.content
    assert page.source_refs[0]["document_id"] == doc.id
    assert page.source_refs[0]["content_hash"] == "h1"
    assert page.stale is False


async def test_generate_upsert_is_idempotent(client_with_db, db_session):
    doc = _doc()
    await _seed_doc(db_session, doc)

    await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_SummaryLLM()
    )
    doc.content_hash = "h2"  # hash changed (update)
    page2 = await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_SummaryLLM()
    )
    pages = (
        await db_session.execute(
            select(WikiPage).where(WikiPage.slug == page2.slug)
        )
    ).scalars().all()
    assert len(pages) == 1, "upsert must not duplicate"
    assert pages[0].source_refs[0]["content_hash"] == "h2"


from sqlalchemy import select  # noqa: E402


async def test_index_page_rebuilds_from_summaries(client_with_db, db_session):
    d1 = _doc("d1", "r1", "a.txt")
    d2 = _doc("d2", "r1", "b.txt")
    await _seed_doc(db_session, d1)
    await _seed_doc(db_session, d2)
    for d in (d1, d2):
        await wiki_service.generate_doc_summary_page(
            db_session, doc=d, llm=_SummaryLLM()
        )
    page = await wiki_service.rebuild_index_page(db_session, rag_id="r1")
    assert page is not None
    assert "Summary: a.txt" in page.content
    assert "Summary: b.txt" in page.content


async def test_llm_failure_returns_none(client_with_db, db_session):
    class _Failing(HashLLMProvider):
        async def stream_chat(self, messages):
            raise RuntimeError("llm down")
            yield ""

    doc = _doc()
    db_session.add(doc)
    await db_session.commit()
    page = await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_Failing()
    )
    assert page is None


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------


async def test_stale_marked_when_hash_changes(client_with_db, db_session):
    doc = _doc()
    await _seed_doc(db_session, doc)
    await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_SummaryLLM()
    )
    # document content changed (hash updated — as the update path does)
    doc.content_hash = "h2"
    await db_session.commit()

    marked = await wiki_service.mark_stale_for_document(
        db_session, document_id=doc.id
    )
    assert marked == 1
    page = (
        await db_session.execute(select(WikiPage).where(WikiPage.slug.is_not(None)))
    ).scalars().first()
    assert page.stale is True


async def test_stale_not_marked_when_hash_unchanged(client_with_db, db_session):
    doc = _doc()
    db_session.add(doc)
    await db_session.commit()
    await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_SummaryLLM()
    )
    marked = await wiki_service.mark_stale_for_document(
        db_session, document_id=doc.id
    )
    assert marked == 0


# ---------------------------------------------------------------------------
# routing: wiki context in prompt
# ---------------------------------------------------------------------------


def test_prompt_includes_wiki_block():
    history = []
    msgs = _build_prompt_messages(
        system_prompt="SYS",
        context="chunk ctx",
        history=history,
        user_message="q",
        wiki_pages=[{
            "slug": "summary-report-txt",
            "title": "Summary: report.txt",
            "content": "digest content",
            "score": 0.9,
        }],
    )
    wiki_blocks = [
        m for m in msgs
        if m["role"] == "system" and "digest" in m["content"]
    ]
    assert len(wiki_blocks) == 1
    assert "digest content" in wiki_blocks[0]["content"]


def test_prompt_without_wiki_pages_unchanged():
    base = _build_prompt_messages(
        system_prompt="SYS", context="ctx", history=[], user_message="q",
    )
    with_wiki_none = _build_prompt_messages(
        system_prompt="SYS", context="ctx", history=[], user_message="q",
        wiki_pages=None,
    )
    assert len(base) == len(with_wiki_none)
