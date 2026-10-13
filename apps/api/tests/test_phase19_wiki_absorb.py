"""Phase 19: query-driven wiki — QA-note absorption lifecycle.

The wiki is a notebook, not a per-document distillation: reliable,
chunk-backed answers are absorbed asynchronously into one QA-notes
page per RAG, and the next similar question routes wiki_only with the
ORIGINAL chunk citations.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.llm.hash_provider import HashLLMProvider
from app.models import Document, WikiPage
from app.services import wiki_router, wiki_service

pytestmark = pytest.mark.offline

RAG = "r1"


def _citation(i=0, score=0.8, doc_id="d1"):
    return {
        "chunk_id": f"chunk-{i}",
        "document_id": doc_id,
        "filename": "alpha.txt",
        "page_number": 2,
        "chunk_text": f"chunk body {i}: the Zephyr-9 warranty is 14,000 hours",
        "retrieval_score": score,
        "metadata": None,
    }


async def _doc(db):
    db.add(
        Document(
            id="d1", rag_id=RAG, filename="alpha.txt",
            file_type="txt", file_size=10, storage_key="k",
            status="READY", content_hash="h1",
        )
    )
    await db.commit()


class _DistillLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield (
            '```json\n{"question": "How long is the Zephyr-9 warranty?",'
            ' "note": "The Zephyr-9 warranty is 14,000 hours of burn time."}\n```'
        )


class _GarbageLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield "sorry, I cannot distill that."


async def _absorb(db, llm, *, min_score=0.35, max_entries=50, citations=None,
                  question="What is the Zephyr-9 warranty?", answer="14,000 hours.",
                  rag_id=RAG):
    return await wiki_service.absorb_qa_note(
        db, rag_id=rag_id, question=question, answer=answer,
        citations=citations if citations is not None else [_citation()],
        llm=llm, min_score=min_score, max_entries=max_entries,
    )


async def _notes_page(db):
    return (
        await db.execute(
            select(WikiPage).where(
                WikiPage.rag_id == RAG,
                WikiPage.kind == wiki_service.KIND_QA_NOTES,
            )
        )
    ).scalars().first()


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


async def test_gate_no_citations_skips(client_with_db, db_session):
    result = await _absorb(db_session, _DistillLLM(), citations=[])
    assert result == {"absorbed": False, "reason": "no_citations"}
    assert await _notes_page(db_session) is None


async def test_gate_low_score_skips(client_with_db, db_session):
    result = await _absorb(
        db_session, _DistillLLM(),
        citations=[_citation(score=0.2)], min_score=0.35,
    )
    assert result == {"absorbed": False, "reason": "low_score"}


async def test_gate_llm_parse_failure_skips(client_with_db, db_session):
    result = await _absorb(db_session, _GarbageLLM())
    assert result["absorbed"] is False
    assert result["reason"] in ("parse_failed", "llm_failed")
    assert await _notes_page(db_session) is None


# ---------------------------------------------------------------------------
# absorption mechanics
# ---------------------------------------------------------------------------


async def test_absorb_creates_page_with_chunk_refs(client_with_db, db_session):
    await _doc(db_session)
    result = await _absorb(db_session, _DistillLLM())
    assert result["absorbed"] is True
    assert result["replaced"] is False
    assert result["entries"] == 1

    page = await _notes_page(db_session)
    assert page.slug == "qa-notes"
    assert "### Q: How long is the Zephyr-9 warranty?" in page.content
    assert "14,000 hours of burn time" in page.content
    assert page.stale is False

    # chunk-level provenance captured with the LIVE document hash
    ref = page.source_refs[0]
    assert ref["chunk_id"] == "chunk-0"
    assert ref["document_id"] == "d1"
    assert ref["content_hash"] == "h1"
    assert ref["page_number"] == 2

    # meta carries the structured index + version
    assert page.meta["version"] == 1
    assert page.meta["items"][0]["q_norm"] == "how long is the zephyr-9 warranty?"


async def test_absorb_same_question_replaces(client_with_db, db_session):
    await _doc(db_session)
    await _absorb(db_session, _DistillLLM())
    result = await _absorb(db_session, _DistillLLM(), answer="Updated: 14,500 hours.")
    assert result["absorbed"] is True
    assert result["replaced"] is True
    assert result["entries"] == 1, "same question must not duplicate"
    assert result["version"] == 2

    page = await _notes_page(db_session)
    assert len(page.meta["items"]) == 1
    assert page.meta["items"][0]["q_norm"] == "how long is the zephyr-9 warranty?"


class _CounterDistillLLM(HashLLMProvider):
    """Distills a DIFFERENT question each call (real LLMs vary too)."""

    counter = 0

    async def stream_chat(self, messages):
        _CounterDistillLLM.counter += 1
        n = _CounterDistillLLM.counter
        yield (
            f'```json\n{{"question": "Question {n}?", "note": "note {n}"}}\n```'
        )


async def test_absorb_fifo_cap_evicts_oldest(client_with_db, db_session):
    await _doc(db_session)
    await _absorb(db_session, _CounterDistillLLM(), max_entries=2)
    await _absorb(db_session, _CounterDistillLLM(), max_entries=2)
    await _absorb(db_session, _CounterDistillLLM(), max_entries=2)

    page = await _notes_page(db_session)
    assert [it["question"] for it in page.meta["items"]] == [
        "Question 2?",
        "Question 3?",
    ], "oldest entry (1) must retire under the FIFO cap"
    assert "Question 1?" not in page.content


async def test_refs_dedupe_across_absorptions(client_with_db, db_session):
    await _doc(db_session)
    await _absorb(db_session, _DistillLLM())
    await _absorb(db_session, _DistillLLM(), question="Another question about zephyr?")
    page = await _notes_page(db_session)
    assert len(page.source_refs) == 1, "same chunk cited twice → one ref"


async def test_chunk_level_citations_from_qa_notes(client_with_db, db_session):
    await _doc(db_session)
    await _absorb(db_session, _DistillLLM())
    page = await _notes_page(db_session)

    pages = [
        {
            "slug": page.slug,
            "kind": page.kind,
            "title": page.title,
            "content": page.content,
            "source_refs": page.source_refs,
            "notes": page.notes,
            "meta": page.meta,
            "score": 0.9,
        }
    ]
    cites = wiki_service.wiki_citations_from_pages(pages)
    assert len(cites) == 1
    c = cites[0]
    assert c["chunk_id"] == "chunk-0", "qa_notes refs must cite the ORIGINAL chunk"
    assert c["page_number"] == 2
    assert c["retrieval_score"] == 0.8
    assert c["metadata"]["source"] == "wiki"


async def test_absorb_invalidates_cache(client_with_db, db_session, monkeypatch):
    from app.services import wiki_cache

    class _Backend:
        def __init__(self):
            self.deleted = []
            self.store = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, *, ex):
            self.store[key] = value

        async def delete(self, key):
            self.deleted.append(key)
            self.store.pop(key, None)

    backend = _Backend()
    monkeypatch.setenv("WIKI_CACHE_ENABLED", "true")
    get_settings.cache_clear()
    wiki_cache.set_cache_for_tests(backend)
    try:
        await _doc(db_session)
        await _absorb(db_session, _DistillLLM())
        assert (
            f"wikicache:v1:{RAG}" in backend.deleted
        ), "absorption must evict the rag's hot cache"
    finally:
        wiki_cache.reset_cache_for_tests()
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# end-to-end money test: ask → absorb → ask again routes wiki_only
# ---------------------------------------------------------------------------


async def test_ask_absorb_ask_again_routes_wiki_only(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    monkeypatch.setenv("WIKI_ROUTE_ENABLED", "true")
    monkeypatch.setenv("WIKI_ABSORB_ENABLED", "true")
    get_settings.cache_clear()
    # the conftest pins WIKI_LLM_MODEL="" (wiki → hash provider) for
    # hermetic tests — the hash output can't be distilled, so point the
    # absorb LLM at a controlled stub
    monkeypatch.setattr(wiki_service, "_wiki_llm", lambda: _DistillLLM())
    try:
        headers = await auth_headers_factory("absorb-e2e@example.com")
        rid = (
            await client_with_db.post(
                "/api/v1/rags", headers=headers, json={"name": "absorb"}
            )
        ).json()["id"]
        up = await client_with_db.post(
            f"/api/v1/rags/{rid}/documents",
            headers=headers,
            files={"file": (
                "alpha.txt",
                b"The Zephyr-9 warranty is 14,000 hours of burn time.",
                "text/plain",
            )},
        )
        assert up.status_code == 201, up.text

        QUESTION = "What is the Zephyr-9 warranty?"

        # turn 1: no wiki → rag_only, and the turn is absorbed
        events = await _chat(client_with_db, headers, rid, QUESTION)
        assert next(e for e in events if e["type"] == "route")["route"] == "rag_only"

        page = (
            await db_session.execute(
                select(WikiPage).where(
                    WikiPage.rag_id == rid,
                    WikiPage.kind == wiki_service.KIND_QA_NOTES,
                )
            )
        ).scalars().first()
        assert page is not None, "the answered turn must be absorbed"
        # prove the absorbed note carries the live source hash
        doc_hash = (
            await db_session.execute(
                select(Document.content_hash).where(
                    Document.id == page.source_refs[0]["document_id"]
                )
            )
        ).scalar()
        assert page.source_refs[0]["content_hash"] == doc_hash

        # turn 2: the same question now routes wiki_only with chunk cites
        events = await _chat(client_with_db, headers, rid, QUESTION)
        route_ev = next(e for e in events if e["type"] == "route")
        assert route_ev["route"] == "wiki_only"
        done = next(e for e in events if e["type"] == "done")
        assert done["route"] == "wiki_only"
        cites = done["citations"]
        assert cites and cites[0]["metadata"]["source"] == "wiki"
        # the citation must be the ORIGINAL absorbed chunk (a real
        # ingestion chunk id, not a wiki-synthesised document-level one)
        assert not cites[0]["chunk_id"].startswith("wiki:")
        assert cites[0]["chunk_id"] == page.source_refs[0]["chunk_id"]
        assert cites[0]["document_id"] == page.source_refs[0]["document_id"]
    finally:
        get_settings.cache_clear()


async def _chat(client_with_db, headers, rid, message):
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": message},
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        events = []
        async for raw in resp.aiter_lines():
            if raw:
                events.append(json.loads(raw))
        return events
