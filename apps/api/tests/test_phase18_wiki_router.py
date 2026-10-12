"""Phase 18: wiki boundary router — decision table, meta profile, abstain.

Every routing rule is a checkable condition; these tests pin the whole
decision table plus the page meta (boundary profile) plumbing and the
deterministic ABSTAIN flow.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.llm.hash_provider import HashLLMProvider
from app.models import Chunk, Document, WikiPage
from app.services import wiki_router, wiki_service

pytestmark = pytest.mark.offline

PAGE_BODY = "zephyr thruster warranty 14000 hours millinewton rating"


def _enable_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIKI_ROUTE_ENABLED", "true")
    get_settings.cache_clear()


def _disable_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIKI_ROUTE_ENABLED", raising=False)
    get_settings.cache_clear()


async def _seed(db_session, *, page_notes=None, page_meta=None, doc_hash="h1",
                page_hash="h1", filename="doc.txt", with_doc=True):
    if with_doc:
        doc = Document(
            id="d1", rag_id="r1", filename=filename,
            file_type="txt", file_size=10, storage_key="k",
            status="READY", content_hash=doc_hash,
        )
        db_session.add(doc)
        db_session.add(
            Chunk(document_id="d1", rag_id="r1", page_number=1,
                  chunk_index=0, text="body")
        )
    page = WikiPage(
        id=f"wp-{abs(hash(page_hash + filename)) % 10**8}", rag_id="r1",
        kind=wiki_service.KIND_DOC_SUMMARY, slug=filename,
        title=f"Summary: {filename}", content=PAGE_BODY,
        source_refs=[{"document_id": "d1", "content_hash": page_hash,
                      "filename": filename}],
        stale=False, notes=page_notes, meta=page_meta,
    )
    db_session.add(page)
    await db_session.commit()
    return page


async def _route(db_session, query):
    return await wiki_router.route_for_query(
        db_session, rag_id="r1", query=query
    )


# ---------------------------------------------------------------------------
# decision table
# ---------------------------------------------------------------------------


async def test_no_relevant_page_routes_rag_only(client_with_db, db_session, monkeypatch):
    _enable_route(monkeypatch)
    try:
        await _seed(db_session)
        decision = await _route(db_session, "what is the capital of France")
        assert decision.route == wiki_router.ROUTE_RAG_ONLY
        assert decision.reason == "no_relevant_page"
    finally:
        _disable_route(monkeypatch)


async def test_routing_disabled_forces_rag_only(client_with_db, db_session, monkeypatch):
    # conftest pins WIKI_ROUTE_ENABLED off — the flag gates everything
    try:
        await _seed(db_session)
        decision = await _route(db_session, "zephyr thruster warranty")
        assert decision.route == wiki_router.ROUTE_RAG_ONLY
        assert decision.reason == "routing_disabled"
    finally:
        get_settings.cache_clear()


async def test_hash_diverged_routes_wiki_plus_rag(client_with_db, db_session, monkeypatch):
    _enable_route(monkeypatch)
    try:
        # page built against an old hash; doc already moved on
        await _seed(db_session, doc_hash="h2", page_hash="h1")
        decision = await _route(db_session, "zephyr thruster warranty")
        assert decision.route == wiki_router.ROUTE_WIKI_PLUS_RAG
        assert decision.reason == "source_stale_or_diverged"
    finally:
        _disable_route(monkeypatch)


async def test_deleted_source_routes_wiki_plus_rag(client_with_db, db_session, monkeypatch):
    _enable_route(monkeypatch)
    try:
        # provenance points at a document row that does not exist
        await _seed(db_session, with_doc=False)
        decision = await _route(db_session, "zephyr thruster warranty")
        assert decision.route == wiki_router.ROUTE_WIKI_PLUS_RAG
        assert decision.reason == "source_stale_or_diverged"
    finally:
        _disable_route(monkeypatch)


async def test_fully_conflicted_routes_abstain(client_with_db, db_session, monkeypatch):
    _enable_route(monkeypatch)
    try:
        await _seed(db_session, page_notes={
            "contradictions": [{"slug_a": "doc.txt", "slug_b": "other",
                                "summary": "warranty numbers disagree"}],
        })
        decision = await _route(db_session, "zephyr thruster warranty")
        assert decision.route == wiki_router.ROUTE_ABSTAIN
        assert decision.reason == "unresolved_conflict"
    finally:
        _disable_route(monkeypatch)


async def test_partial_conflict_routes_wiki_plus_rag(client_with_db, db_session, monkeypatch):
    _enable_route(monkeypatch)
    try:
        await _seed(db_session, page_notes={
            "contradictions": [{"slug_a": "doc.txt", "slug_b": "x", "summary": "m"}],
        })
        # a second, clean page on the same topic (valid provenance!)
        db_session.add(
            WikiPage(
                id="wp-clean", rag_id="r1", kind=wiki_service.KIND_DOC_SUMMARY,
                slug="doc2.txt", title="Summary: doc2.txt",
                content=PAGE_BODY + " clean copy",
                source_refs=[{"document_id": "d1", "content_hash": "h1",
                              "filename": "doc2.txt"}],
                stale=False, notes=None,
            )
        )
        await db_session.commit()
        decision = await _route(db_session, "zephyr thruster warranty")
        assert decision.route == wiki_router.ROUTE_WIKI_PLUS_RAG
        assert decision.reason == "partial_conflict"
    finally:
        _disable_route(monkeypatch)


@pytest.mark.parametrize("query", [
    "请逐字引用 zephyr thruster warranty 保修条款",
    "zephyr thruster warranty 在第几页？给出页码",
    "Quote the exact wording about zephyr thruster warranty",
    "对比 zephyr thruster warranty 与其他推进器",
])
async def test_exactness_routes_wiki_plus_rag(
    client_with_db, db_session, monkeypatch, query
):
    _enable_route(monkeypatch)
    try:
        await _seed(db_session)
        decision = await _route(db_session, query)
        assert decision.route == wiki_router.ROUTE_WIKI_PLUS_RAG
        assert decision.reason.startswith("exact_verification:")
    finally:
        _disable_route(monkeypatch)


async def test_declared_not_cover_routes_wiki_plus_rag(client_with_db, db_session, monkeypatch):
    _enable_route(monkeypatch)
    try:
        await _seed(db_session, page_meta={
            "version": 1, "covers": ["thruster warranty"],
            "does_not_cover": ["probation exceptions"],
        })
        # the question touches a declared gap
        decision = await _route(db_session, "zephyr thruster probation exceptions")
        assert decision.route == wiki_router.ROUTE_WIKI_PLUS_RAG
        assert decision.reason.startswith("declared_not_covered:")
    finally:
        _disable_route(monkeypatch)


async def test_full_boundary_pass_routes_wiki_only(client_with_db, db_session, monkeypatch):
    _enable_route(monkeypatch)
    try:
        await _seed(db_session, page_meta={
            "version": 3, "covers": ["thruster warranty"],
            "does_not_cover": ["probation exceptions"],
        })
        decision = await _route(db_session, "zephyr thruster warranty")
        assert decision.route == wiki_router.ROUTE_WIKI_ONLY
        assert decision.reason == "boundary_pass"
        assert decision.slugs == ["doc.txt"]
    finally:
        _disable_route(monkeypatch)


def test_status_derivation_matrix():
    valid = {"d1": "h1"}
    fresh = {"source_refs": [{"document_id": "d1", "content_hash": "h1"}],
             "notes": None}
    diverged = {"source_refs": [{"document_id": "d1", "content_hash": "old"}],
                "notes": None}
    ghost = {"source_refs": [{"document_id": "gone", "content_hash": "x"}],
             "notes": None}
    no_refs = {"source_refs": [], "notes": None}
    conflict = {"source_refs": [{"document_id": "d1", "content_hash": "h1"}],
                "notes": {"contradictions": [{"summary": "m"}]}}
    conflict_and_diverged = {
        "source_refs": [{"document_id": "d1", "content_hash": "old"}],
        "notes": {"contradictions": [{"summary": "m"}]},
    }
    assert wiki_router._derive_status(fresh, valid) == "VERIFIED"
    assert wiki_router._derive_status(diverged, valid) == "STALE"
    assert wiki_router._derive_status(ghost, valid) == "STALE"
    assert wiki_router._derive_status(no_refs, valid) == "STALE"
    assert wiki_router._derive_status(conflict, valid) == "CONFLICTED"
    # stale wins over conflict — freshness is checked first
    assert wiki_router._derive_status(conflict_and_diverged, valid) == "STALE"


# ---------------------------------------------------------------------------
# meta profile (boundary declaration)
# ---------------------------------------------------------------------------


class _MetaLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield (
            "## Key Points\n- thrust 310 mN\n\n"
            "```wiki-meta\n"
            '{"covers": ["thruster warranty"], "does_not_cover": ["probation exceptions"]}\n'
            "```\n"
        )


class _BrokenMetaLLM(HashLLMProvider):
    async def stream_chat(self, messages):
        yield "## Key Points\n- x\n\n```wiki-meta\n{not json\n```\n"


def _doc(doc_id="d1", rag_id="r1", filename="doc.txt"):
    return Document(
        id=doc_id, rag_id=rag_id, filename=filename,
        file_type="txt", file_size=10, storage_key="k",
        status="READY", content_hash="h1",
    )


async def test_summary_strips_meta_fence_and_stores_profile(client_with_db, db_session):
    doc = _doc()
    db_session.add(doc)
    db_session.add(
        Chunk(document_id="d1", rag_id="r1", page_number=1, chunk_index=0,
              text="body")
    )
    await db_session.commit()

    page = await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_MetaLLM()
    )
    assert page is not None
    assert "wiki-meta" not in page.content, "fence must not reach the prompt"
    assert page.content.startswith("## Key Points")
    assert page.meta["version"] == 1
    assert page.meta["covers"] == ["thruster warranty"]
    assert page.meta["does_not_cover"] == ["probation exceptions"]
    assert "generated_at" in page.meta

    # regeneration bumps the version
    page2 = await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_MetaLLM()
    )
    assert page2.meta["version"] == 2


async def test_summary_malformed_meta_stripped_profile_none(client_with_db, db_session):
    doc = _doc()
    db_session.add(doc)
    db_session.add(
        Chunk(document_id="d1", rag_id="r1", page_number=1, chunk_index=0,
              text="body")
    )
    await db_session.commit()

    page = await wiki_service.generate_doc_summary_page(
        db_session, doc=doc, llm=_BrokenMetaLLM()
    )
    assert page is not None
    assert "wiki-meta" not in page.content, "garbage fence must still be stripped"
    assert page.meta is not None and "covers" not in page.meta


def test_split_meta_block_unit():
    clean, meta = wiki_service._split_meta_block(
        'body\n```wiki-meta\n{"covers": ["a"]}\n```'
    )
    assert clean == "body"
    assert meta == {"covers": ["a"]}

    clean, meta = wiki_service._split_meta_block(
        "body\n```wiki-meta\n{oops\n```"
    )
    assert clean == "body"
    assert meta is None

    clean, meta = wiki_service._split_meta_block("plain body")
    assert clean == "plain body"
    assert meta is None


# ---------------------------------------------------------------------------
# abstain flow over HTTP — deterministic, no LLM in the loop
# ---------------------------------------------------------------------------


async def test_abstain_flow_never_calls_llm(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    _enable_route(monkeypatch)

    async def _explode(self, messages):
        raise AssertionError("ABSTAIN must not reach the LLM")

    monkeypatch.setattr(HashLLMProvider, "stream_chat", _explode)

    headers = await auth_headers_factory("abstain@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "abstain"}
        )
    ).json()["id"]
    await _seed(db_session, page_notes={
        "contradictions": [{"slug_a": "doc.txt", "slug_b": "other", "summary": "m"}],
    })
    # the seeded rag is "r1" but the chat targets the user's rag — move
    # the page over
    page = (
        await db_session.execute(select(WikiPage).where(WikiPage.rag_id == "r1"))
    ).scalars().first()
    page.rag_id = rid
    await db_session.commit()

    events = []
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "zephyr thruster warranty"},
    ) as resp:
        assert resp.status_code == 200
        async for raw in resp.aiter_lines():
            if raw:
                events.append(json.loads(raw))

    types = [e["type"] for e in events]
    assert types.count("route") == 1
    assert types[0] == "route"
    assert events[0]["route"] == "abstain"
    assert events[0]["reason"] == "unresolved_conflict"
    assert types.count("token") > 0
    answer = "".join(e["delta"] for e in events if e["type"] == "token")
    assert "contradictions" in answer, "refusal text must explain the conflict"
    done = next(e for e in events if e["type"] == "done")
    assert done["route"] == "abstain"
    assert done["citations"] == []
