"""Phase 16: wiki-first routing — route decision, skip-search, metrics.

Routing semantics under test: with ``WIKI_ROUTE_ENABLED`` on, a chat
turn whose query lexically matches a non-stale wiki page at or above
``WIKI_ROUTE_MIN_SCORE`` is answered from the distilled pages (chunk
retrieval SKIPPED, citations synthesised from page provenance); a miss
falls back to the regular search with wiki pages as an extra prompt
block (Phase 15 behaviour, also the behaviour when the flag is off).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.models import Document, WikiPage
from app.services import search_service, wiki_service

pytestmark = pytest.mark.offline

PAGE_CONTENT = "zephyr thruster warranty 14000 hours millinewton rating"
HIT_QUERY = "zephyr thruster warranty"  # every token overlaps the page
MISS_QUERY = "what is the capital of France"  # zero overlap


async def _setup_rag_with_page(
    client_with_db, auth_headers_factory, db_session, *, stale: bool = False
) -> Dict[str, str]:
    """Create a user/rag/doc and seed one matching wiki page directly.

    The page is seeded (not LLM-generated) so the lexical overlap with
    ``HIT_QUERY`` is deterministic regardless of the hash LLM's output.
    """
    headers = await auth_headers_factory("wiki-route@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "route"}
        )
    ).json()["id"]
    up = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": ("alpha.txt", b"unrelated chunk body bytes", "text/plain")},
    )
    assert up.status_code == 201, up.text
    doc_id = up.json()["id"]

    # the router live-validates provenance — the seeded page must cite
    # the document's REAL content hash or it degrades to WIKI_PLUS_RAG
    real_hash = (
        await db_session.execute(
            select(Document.content_hash).where(Document.id == doc_id)
        )
    ).scalar()

    db_session.add(
        WikiPage(
            id=f"wp-{rid[:8]}",
            rag_id=rid,
            kind=wiki_service.KIND_DOC_SUMMARY,
            slug="alpha-txt",
            title="Summary: alpha.txt",
            content=PAGE_CONTENT,
            source_refs=[
                {
                    "document_id": doc_id,
                    "content_hash": real_hash,
                    "filename": "alpha.txt",
                }
            ],
            stale=stale,
        )
    )
    await db_session.commit()
    return {"headers": headers, "rag_id": rid, "doc_id": doc_id,
            "content_hash": real_hash}


async def _chat_events(client_with_db, headers, rag_id, message) -> List[Dict[str, Any]]:
    async with client_with_db.stream(
        "POST",
        f"/api/v1/rags/{rag_id}/chat",
        headers=headers,
        json={"message": message},
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        events = []
        async for raw in resp.aiter_lines():
            if raw:
                events.append(json.loads(raw))
        return events


def _enable_route(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    monkeypatch.setenv("WIKI_ROUTE_ENABLED", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # chat_service reads the lru_cached get_settings(), not the
    # router-level test_settings override — reset the cache so the
    # env vars above are picked up.
    get_settings.cache_clear()


def _spy_search(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    calls: List[int] = []

    async def _spy(*args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(search_service, "search_rag", _spy)
    return calls


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


async def test_wiki_hit_skips_chunk_retrieval(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    _enable_route(monkeypatch)
    calls = _spy_search(monkeypatch)
    env = await _setup_rag_with_page(
        client_with_db, auth_headers_factory, db_session
    )

    events = await _chat_events(
        client_with_db, env["headers"], env["rag_id"], HIT_QUERY
    )
    types = [e["type"] for e in events]

    route_ev = next(e for e in events if e["type"] == "route")
    assert route_ev["route"] == "wiki_only"
    assert route_ev["score"] >= 0.3
    assert route_ev["wiki_slugs"] == ["alpha-txt"]

    # exactly one route + one citation event, both before any token
    assert types.count("route") == 1
    assert types.count("citation") == 1
    assert types.index("route") < types.index("citation") < types.index("token")

    cite_ev = next(e for e in events if e["type"] == "citation")
    assert len(cite_ev["citations"]) == 1
    c = cite_ev["citations"][0]
    assert c["chunk_id"].startswith("wiki:alpha-txt:")
    assert c["document_id"] == env["doc_id"]
    assert c["filename"] == "alpha.txt"
    assert c["page_number"] == 0
    assert c["metadata"]["source"] == "wiki"
    assert c["metadata"]["content_hash"] == env["content_hash"]
    assert 0.0 <= c["retrieval_score"] <= 1.0

    done = next(e for e in events if e["type"] == "done")
    assert done["route"] == "wiki_only"
    assert done["wiki_slugs"] == ["alpha-txt"]

    # the whole point of the route: search never ran
    assert calls == []


async def test_wiki_miss_falls_back_to_chunks(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    _enable_route(monkeypatch)
    calls = _spy_search(monkeypatch)
    env = await _setup_rag_with_page(
        client_with_db, auth_headers_factory, db_session
    )

    events = await _chat_events(
        client_with_db, env["headers"], env["rag_id"], MISS_QUERY
    )

    route_ev = next(e for e in events if e["type"] == "route")
    assert route_ev["route"] == "rag_only"
    # below the routing threshold — the hash LLM's own pages may share
    # a stopword or two with the query, so assert "below threshold",
    # not exactly zero
    assert route_ev["score"] < 0.3
    done = next(e for e in events if e["type"] == "done")
    assert done["route"] == "rag_only"
    assert calls, "fallback must call the regular search"


async def test_route_disabled_keeps_legacy_behaviour(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    # flag absent → Phase 15 behaviour even when the page matches
    calls = _spy_search(monkeypatch)
    env = await _setup_rag_with_page(
        client_with_db, auth_headers_factory, db_session
    )

    events = await _chat_events(
        client_with_db, env["headers"], env["rag_id"], HIT_QUERY
    )

    route_ev = next(e for e in events if e["type"] == "route")
    assert route_ev["route"] == "rag_only"
    assert calls, "legacy behaviour must search chunks"


async def test_threshold_boundary_forces_fallback(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    # a min score above the theoretical max (1.0) never routes
    _enable_route(monkeypatch, WIKI_ROUTE_MIN_SCORE="1.1")
    calls = _spy_search(monkeypatch)
    env = await _setup_rag_with_page(
        client_with_db, auth_headers_factory, db_session
    )

    events = await _chat_events(
        client_with_db, env["headers"], env["rag_id"], HIT_QUERY
    )
    route_ev = next(e for e in events if e["type"] == "route")
    assert route_ev["route"] == "rag_only"
    assert calls


async def test_stale_pages_never_route(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    _enable_route(monkeypatch)
    calls = _spy_search(monkeypatch)
    env = await _setup_rag_with_page(
        client_with_db, auth_headers_factory, db_session, stale=True
    )

    events = await _chat_events(
        client_with_db, env["headers"], env["rag_id"], HIT_QUERY
    )
    route_ev = next(e for e in events if e["type"] == "route")
    assert route_ev["route"] == "rag_only"
    assert calls


async def test_no_context_refusal_path_intact(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    """Route on + unrelated query + no chunks → the hash provider's
    no-context contract (searched, found nothing, says so) survives."""
    _enable_route(monkeypatch)
    env = await _setup_rag_with_page(
        client_with_db, auth_headers_factory, db_session
    )

    events = await _chat_events(
        client_with_db, env["headers"], env["rag_id"], MISS_QUERY
    )
    assert next(e for e in events if e["type"] == "route")["route"] == "rag_only"
    done = next(e for e in events if e["type"] == "done")
    assert done["message_id"]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


async def test_route_and_token_metrics_recorded(
    client_with_db, auth_headers_factory, db_session, monkeypatch
):
    from app.metrics import render, reset

    _enable_route(monkeypatch)
    env = await _setup_rag_with_page(
        client_with_db, auth_headers_factory, db_session
    )
    reset()
    try:
        await _chat_events(
            client_with_db, env["headers"], env["rag_id"], HIT_QUERY
        )
        await _chat_events(
            client_with_db, env["headers"], env["rag_id"], MISS_QUERY
        )

        out = render()
        assert 'chat_route_total{route="wiki_only"} 1' in out
        assert 'chat_route_total{route="rag_only"} 1' in out
        assert 'llm_tokens_estimated_total{kind="prompt"}' in out
        assert 'llm_tokens_estimated_total{kind="completion"}' in out
        assert "chat_search_seconds_count" in out  # only the fallback searched
        assert "chat_llm_seconds_count" in out
    finally:
        reset()
        get_settings.cache_clear()
