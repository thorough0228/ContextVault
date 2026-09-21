"""End-to-end cross-tenant security tests.

User A creates resources, User B attempts to read / write / search /
chat into them. Every cross-tenant URL must 404. Every cross-tenant
search / chat must not leak content. User A's own URLs continue to
work.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import httpx
import pytest


async def _events(resp) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    async for raw in resp.aiter_lines():
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            raise AssertionError(f"non-JSON event line: {raw!r}")
    return out


async def _seed_alice_with_rag_and_doc(
    client: httpx.AsyncClient, auth_headers_factory
) -> tuple:
    """Create Alice + her rag + one processed document. Returns
    ``(rag_id, document_id)``."""
    headers = await auth_headers_factory("e2e-alice@example.com")
    rag_resp = await client.post(
        "/api/v1/rags", headers=headers,
        json={"name": "alice-rag", "description": ""},
    )
    rag_id = rag_resp.json()["id"]
    upload = await client.post(
        f"/api/v1/rags/{rag_id}/documents",
        headers=headers,
        files={"file": ("a.txt", io.BytesIO(b"alice private content\n" * 6), "text/plain")},
    )
    document_id = upload.json()["id"]
    # Wait for READY
    import asyncio
    for _ in range(20):
        check = await client.get(
            f"/api/v1/documents/{document_id}", headers=headers
        )
        if check.json()["status"] == "READY":
            break
        await asyncio.sleep(0.1)
    return headers, rag_id, document_id


# ----- 1. get_rag 404 --------------------------------------------------


async def test_cross_tenant_get_rag_404(
    client_with_db, auth_headers_factory
) -> None:
    alice_h, alice_rag, _ = await _seed_alice_with_rag_and_doc(
        client_with_db, auth_headers_factory
    )
    bob_h = await auth_headers_factory("e2e-bob@example.com")

    resp = await client_with_db.get(
        f"/api/v1/rags/{alice_rag}", headers=bob_h
    )
    assert resp.status_code == 404
    # Response must not contain alice's rag name.
    assert "alice-rag" not in resp.text.lower()


# ----- 2. get_document 404 ---------------------------------------------


async def test_cross_tenant_get_document_404(
    client_with_db, auth_headers_factory
) -> None:
    _, alice_rag, alice_doc = await _seed_alice_with_rag_and_doc(
        client_with_db, auth_headers_factory
    )
    bob_h = await auth_headers_factory("e2e-bob2@example.com")

    resp = await client_with_db.get(
        f"/api/v1/documents/{alice_doc}", headers=bob_h
    )
    assert resp.status_code == 404


# ----- 3. delete_document 404 ------------------------------------------


async def test_cross_tenant_delete_document_404(
    client_with_db, db_sessionmaker, auth_headers_factory
) -> None:
    alice_h, alice_rag, alice_doc = await _seed_alice_with_rag_and_doc(
        client_with_db, auth_headers_factory
    )
    bob_h = await auth_headers_factory("e2e-bob3@example.com")

    resp = await client_with_db.delete(
        f"/api/v1/documents/{alice_doc}", headers=bob_h
    )
    assert resp.status_code == 404

    # Alice's doc still exists.
    check = await client_with_db.get(
        f"/api/v1/documents/{alice_doc}", headers=alice_h
    )
    assert check.status_code == 200
    assert check.json()["status"] == "READY"


# ----- 4. search 404 ----------------------------------------------------


async def test_cross_tenant_search_404(
    client_with_db, auth_headers_factory
) -> None:
    alice_h, alice_rag, _ = await _seed_alice_with_rag_and_doc(
        client_with_db, auth_headers_factory
    )
    bob_h = await auth_headers_factory("e2e-bob4@example.com")

    resp = await client_with_db.post(
        f"/api/v1/rags/{alice_rag}/search",
        headers=bob_h,
        json={"query": "private", "top_k": 5},
    )
    assert resp.status_code == 404
    # Must not leak Alice's content in the body.
    body = resp.text.lower()
    assert "alice" not in body
    assert "private" not in body


# ----- 5. chat 404 (pre-stream router check) ---------------------------


async def test_cross_tenant_chat_404_pre_stream(
    client_with_db, auth_headers_factory
) -> None:
    alice_h, alice_rag, _ = await _seed_alice_with_rag_and_doc(
        client_with_db, auth_headers_factory
    )
    bob_h = await auth_headers_factory("e2e-bob5@example.com")

    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{alice_rag}/chat",
        headers=bob_h,
        json={"message": "private?"},
    ) as resp:
        # The router-level ownership check fires BEFORE the stream
        # opens, so the response status is 404, not 200 with an
        # in-stream error event.
        assert resp.status_code == 404
        body = await resp.aread()
        # No leak of Alice's content.
        assert b"alice" not in body.lower()
        assert b"private" not in body.lower()


# ----- 6. get_conversation 404 -----------------------------------------


async def test_cross_tenant_get_conversation_404(
    client_with_db, auth_headers_factory
) -> None:
    """A conversation Alice owns must 404 for Bob, even if Bob knows
    the conversation_id. The Phase 6 router-level re-check on
    ``get_conversation`` (``chat.py``) raises NotFoundError before
    any messages are loaded."""
    alice_h, alice_rag, _ = await _seed_alice_with_rag_and_doc(
        client_with_db, auth_headers_factory
    )
    bob_h = await auth_headers_factory("e2e-bob6@example.com")

    # First, Alice creates a conversation.
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{alice_rag}/chat",
        headers=alice_h, json={"message": "hi"},
    ) as resp:
        events = await _events(resp)
    conv_id = next(e for e in events if e["type"] == "done")["conversation_id"]

    # Bob asks for it.
    resp = await client_with_db.get(
        f"/api/v1/conversations/{conv_id}", headers=bob_h
    )
    assert resp.status_code == 404
