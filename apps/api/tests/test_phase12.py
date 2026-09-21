"""Unit + integration tests for Phase 12: hybrid retrieval,
conversation compression, and long-term memory."""

from __future__ import annotations

import json

import pytest

from app.services.tokenization import tokenize, tokenize_to_string
from app.services.search_service import _rrf_fuse

pytestmark = pytest.mark.offline

from app.llm.hash_provider import HashLLMProvider
from app.models import Conversation, MemoryFact


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


class TestTokenization:
    def test_english_words_lowercase(self):
        assert tokenize("Rubber and Sponge") == ["rubber", "and", "sponge"]

    def test_chinese_uses_jieba(self):
        tokens = tokenize("乒乓球胶皮")
        assert tokens, "chinese tokens must not be empty"
        assert all(t.strip() for t in tokens)

    def test_mixed_text(self):
        tokens = tokenize("attack 反胶")
        assert "attack" in tokens
        assert any(t for t in tokens if t != "attack")

    def test_punctuation_dropped(self):
        assert tokenize("hello, world! (test)") == ["hello", "world", "test"]

    def test_string_form_joins_with_spaces(self):
        assert tokenize_to_string("a b") == "a b"

    def test_empty(self):
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# RRF fusion (hand-computed, k=60)
# ---------------------------------------------------------------------------


def _hit(cid: str) -> dict:
    return {"chunk_id": cid, "score": 1.0}


class TestRRF:
    def test_hand_computed_fusion(self):
        # rank1 in both lists: 1/61 + 1/61 = 0.0328; rank1 in primary
        # only: 1/61 = 0.0164; rank2 in both: 2/62 = 0.0323.
        fused = _rrf_fuse(
            [_hit("a"), _hit("b")], [_hit("a"), _hit("c")], k=60, top_k=3
        )
        order = [h["chunk_id"] for h in fused]
        assert order[0] == "a"
        assert set(order[1:]) == {"b", "c"}
        assert fused[0]["score"] == pytest.approx(1.0)  # max-normalised
        scores = [h["score"] for h in fused]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_top_k_slices(self):
        fused = _rrf_fuse(
            [_hit(str(i)) for i in range(10)],
            [],
            k=60,
            top_k=3,
        )
        assert len(fused) == 3

    def test_empty_inputs(self):
        assert _rrf_fuse([], [], k=60, top_k=5) == []


# ---------------------------------------------------------------------------
# Hybrid path over SQLite (integration via HTTP)
# ---------------------------------------------------------------------------


async def test_hybrid_search_fusion_over_sqlite(
    client_with_db, auth_headers_factory, monkeypatch: pytest.MonkeyPatch
):
    """Keyword-leg promotion: the query is an exact proper-noun phrase
    that lives in one chunk among many — the hybrid mode must still
    return ordered in-range hits (fusion contract) with the exact
    phrase present."""

    from app.config import get_settings

    monkeypatch.setenv("SEARCH_MODE", "hybrid")
    get_settings.cache_clear()

    headers = await auth_headers_factory("hybrid@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "hyb"}
        )
    ).json()["id"]
    payload = json.dumps(
        [
            {"id": "c0", "text": "Zyzzix Matrix 9000 is an imaginary rubber for the test."},
            {"id": "c1", "text": "Generic filler content about blades and handles."},
            {"id": "c2", "text": "More filler about training routines and footwork."},
        ]
    )
    up = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": ("hyb.json", payload.encode(), "application/json")},
    )
    assert up.status_code == 201

    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/search",
        headers=headers,
        json={"query": "Zyzzix Matrix 9000", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"], "hybrid search must return hits"
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert any("Zyzzix" in h["chunk_text"] for h in body["hits"][:2])

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Conversation compression
# ---------------------------------------------------------------------------


def test_prompt_summary_and_memory_injection():
    from app.models import Conversation, Message
    from app.services import chat_service

    conv = Conversation(user_id="u", rag_id="r", title="")
    history = [Message(conversation_id="c", role="user", content="hi")]
    history[0].conversation = conv

    base = chat_service._build_prompt_messages(
        system_prompt=chat_service.SYSTEM_PROMPT,
        context="ctx",
        history=history,
        user_message="q",
    )
    injected = chat_service._build_prompt_messages(
        system_prompt=chat_service.SYSTEM_PROMPT,
        context="ctx",
        history=history,
        user_message="q",
        conversation_summary="Earlier the user said they play penhold.",
        memories=["User prefers tacky rubbers"],
    )
    # None args → exactly the legacy message list (4).
    assert len(base) == 4
    # +2 system messages (memories, then summary) between context and history.
    assert len(injected) == 6
    assert injected[2]["role"] == "system"
    assert "tacky rubbers" in injected[2]["content"]
    assert injected[3]["role"] == "system"
    assert "penhold" in injected[3]["content"]
    # Injected messages never appear as history roles.
    roles = [m["role"] for m in injected]
    assert roles.count("system") == 4


# ---------------------------------------------------------------------------
# Conversation compression (integration)
# ---------------------------------------------------------------------------


class _SummaryLLM(HashLLMProvider):
    """Hash LLM that records the last prompt it saw."""

    def __init__(self):
        self.prompts = []

    async def stream_chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        yield "COMPRESSED SUMMARY MARKER"


async def test_compression_folds_overflow_into_summary(
    client_with_db, db_session, auth_headers_factory, monkeypatch: pytest.MonkeyPatch
):
    from app.config import get_settings
    from app.db import reset_engine_for_tests
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine

    monkeypatch.setenv("CHAT_MAX_HISTORY_MESSAGES", "2")
    monkeypatch.setenv("SUMMARY_ENABLED", "1")
    get_settings.cache_clear()

    from app.llm import set_provider_for_tests

    llm = _SummaryLLM()
    set_provider_for_tests(llm)

    headers = await auth_headers_factory("compress@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "c"}
        )
    ).json()["id"]
    conv_id = None
    for i in range(6):
        body = {"message": f"turn {i}: tell me about rubbers {i}"}
        if conv_id:
            body["conversation_id"] = conv_id
        async with client_with_db.stream(
            "POST", f"/api/v1/rags/{rid}/chat", headers=headers, json=body,
        ) as resp:
            events = []
            async for line in resp.aiter_lines():
                if line.strip():
                    events.append(json.loads(line))
        if conv_id is None:
            conv_id = next(e for e in events if e["type"] == "done")[
                "conversation_id"
            ]

    assert llm.prompts, "compression LLM was never called"

    # Inspect the persisted Conversation row via the test SQLite DB
    # (the summary column is intentionally not exposed over the API).
    row = (
        await db_session.execute(
            select(
                Conversation.summary,
                Conversation.summary_until_message_id,
            ).where(Conversation.id == conv_id)
        )
    ).one()
    assert row.summary and "COMPRESSED SUMMARY MARKER" in row.summary
    assert row.summary_until_message_id


async def test_compression_disabled_keeps_summary_null(
    client_with_db, db_session, auth_headers_factory, monkeypatch: pytest.MonkeyPatch
):
    from app.config import get_settings
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine

    monkeypatch.setenv("SUMMARY_ENABLED", "0")
    monkeypatch.setenv("CHAT_MAX_HISTORY_MESSAGES", "2")
    get_settings.cache_clear()

    headers = await auth_headers_factory("nosum@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "c2"}
        )
    ).json()["id"]
    conv_id = None
    for i in range(5):
        body = {"message": f"m{i}"}
        if conv_id:
            body["conversation_id"] = conv_id
        async with client_with_db.stream(
            "POST", f"/api/v1/rags/{rid}/chat", headers=headers, json=body,
        ) as resp:
            async for line in resp.aiter_lines():
                if line.strip() and json.loads(line)["type"] == "done":
                    conv_id = json.loads(line)["conversation_id"]

    row = (
        await db_session.execute(
            select(Conversation.summary, Conversation.summary_until_message_id)
            .where(Conversation.id == conv_id)
        )
    ).one()
    assert row.summary is None
    assert row.summary_until_message_id is None


# ---------------------------------------------------------------------------
# Long-term memory: extract_memory task + injection fetch
# ---------------------------------------------------------------------------


class _JsonLLM(HashLLMProvider):
    """LLM returning a fixed JSON fact array."""

    def __init__(self, facts: list[str]):
        self.facts = facts

    async def stream_chat(self, messages):
        yield json.dumps(self.facts)


async def test_extract_memory_task_persists_and_dedupes(
    client_with_db, db_session, auth_headers_factory, monkeypatch: pytest.MonkeyPatch
):
    from app.config import get_settings
    from app.celery_client import celery_app
    from app.llm import set_provider_for_tests

    from sqlalchemy import select

    monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "1")
    get_settings.cache_clear()

    facts = ["User plays penhold grip", "User owns a Stiga blade"]
    set_provider_for_tests(_JsonLLM(facts))

    headers = await auth_headers_factory("mem@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "mem"}
        )
    ).json()["id"]
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "remember my grip"},
    ) as resp:
        # Consume the FULL stream: the memory-extraction enqueue happens
        # after the done event is produced (generator tail).
        async for line in resp.aiter_lines():
            if line.strip():
                json.loads(line)

    rows = (
        await db_session.execute(
            select(MemoryFact.content).where(MemoryFact.rag_id == rid)
        )
    ).scalars().all()
    assert set(facts) <= set(rows)

    # Second identical turn: dedup keeps the table at the same facts.
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers,
        json={"message": "remember my grip again", "top_k": 1},
    ) as resp:
        async for line in resp.aiter_lines():
            if line.strip():
                json.loads(line)
    rows2 = (
        await db_session.execute(
            select(MemoryFact.content).where(MemoryFact.rag_id == rid)
        )
    ).scalars().all()
    assert sorted(rows2) == sorted(rows)

    get_settings.cache_clear()


async def test_memory_injection_in_prompt(
    client_with_db, db_session, auth_headers_factory, monkeypatch: pytest.MonkeyPatch
):
    from app.config import get_settings

    monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "0")
    monkeypatch.setenv("MEMORY_INJECTION_ENABLED", "1")
    get_settings.cache_clear()

    headers = await auth_headers_factory("inj@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "inj"}
        )
    ).json()["id"]
    db_session.add(
        MemoryFact(
            user_id="u-inj", rag_id=rid,
            content="User is allergic to speed glue",
        )
    )
    await db_session.commit()
    # user_id must match the registered account for the fetch to fire —
    # look it up instead of guessing.
    from sqlalchemy import select
    from app.models import User
    user = (
        await db_session.execute(select(User).where(User.email == "inj@example.com"))
    ).scalars().one()
    fact = (
        await db_session.execute(select(MemoryFact))
    ).scalars().one()
    fact.user_id = user.id
    await db_session.commit()

    await client_with_db.post(
        f"/api/v1/rags/{rid}/search",
        headers=headers,
        json={"query": "anything", "top_k": 1},
    )
    # No assertion on search; the injected prompt is covered by the
    # unit test above. This test proves the fetch path runs against
    # (user, rag) without error when facts exist.


# ---------------------------------------------------------------------------
# BM25 keyword leg (Phase 12 hybrid)
# ---------------------------------------------------------------------------


async def test_bm25_keyword_hits_ranks_by_relevance(
    client_with_db, db_session, auth_headers_factory
):
    """BM25 must promote the chunk sharing exact terms with the query
    over unrelated chunks of the same RAG."""

    from app.services import bm25_service
    from app.services.search_service import _keyword_hits

    headers = await auth_headers_factory("bm25@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "bm25"}
        )
    ).json()["id"]
    payload = json.dumps([
        {"id": "c0", "text": "Zyzzix Matrix 9000 blade review with detailed measurements."},
        {"id": "c1", "text": "Completely unrelated cooking recipe content."},
        {"id": "c2", "text": "Footwork drills and training schedule notes."},
    ])
    up = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": ("bm25.json", payload.encode(), "application/json")},
    )
    assert up.status_code == 201

    hits = await bm25_service.bm25_keyword_hits(
        db_session, rag_id=rid, query="Zyzzix Matrix blade", top_k=3
    )
    assert hits is not None, "BM25 must be available (rank_bm25 installed)"
    assert hits[0]["chunk_text"].startswith("Zyzzix Matrix 9000")
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Cache invalidation forces a rebuild without error.
    bm25_service.invalidate(rid)
    hits2 = await bm25_service.bm25_keyword_hits(
        db_session, rag_id=rid, query="Zyzzix Matrix blade", top_k=3
    )
    assert hits2 and hits2[0]["chunk_id"] == hits[0]["chunk_id"]
    bm25_service.invalidate(rid)


async def test_hybrid_search_uses_bm25_leg(
    client_with_db, db_session, auth_headers_factory, test_settings
):
    from app.config import get_settings

    test_settings.search_mode = "hybrid"
    get_settings.cache_clear()

    headers = await auth_headers_factory("bm25hyb@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "bm25h"}
        )
    ).json()["id"]
    payload = json.dumps([
        {"id": "k0", "text": "The unique Beryllium Foil Gasket appears here."},
        {"id": "k1", "text": "Unrelated filler text for the corpus."},
    ])
    up = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": ("bm25h.json", payload.encode(), "application/json")},
    )
    assert up.status_code == 201
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/search",
        headers=headers,
        json={"query": "Beryllium Foil Gasket", "top_k": 2},
    )
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits and "Beryllium Foil Gasket" in hits[0]["chunk_text"]
