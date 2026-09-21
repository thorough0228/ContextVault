"""ACL leakage suite — cross-tenant isolation over the eval harness.

Every test builds TWO users in the same app instance (SqliteEvalEnv,
deterministic local bge-small + hash LLM) and asserts that tenant A's
knowledge — documents, chunks, memories, bluegreen chains — is
invisible to tenant B through every access path: search (vector +
BM25 keyword legs), chat, document update/rollback, and the memory
store.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.offline

TEXT_A = "Tenant alpha secret: the retro encabulator lives in building seven."
TEXT_B = "Tenant beta secret: the turbo encabulator lives in building nine."


async def _register_second_user(client, email: str) -> dict:
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "acl-pass-2026"},
    )
    assert resp.status_code in (200, 201), resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def _make_rag_with_doc(client, headers, rag_name, filename, text):
    rid = (
        await client.post(
            "/api/v1/rags", headers=headers, json={"name": rag_name}
        )
    ).json()["id"]
    up = await client.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": (filename, text, "text/plain")},
    )
    assert up.status_code == 201, up.text
    doc_id = up.json()["id"]
    import asyncio

    for _ in range(100):
        detail = await client.get(f"/api/v1/documents/{doc_id}", headers=headers)
        if detail.json()["status"] == "READY":
            return rid, doc_id
        await asyncio.sleep(0.05)
    raise AssertionError("document never reached READY")


async def _two_tenant_fixture(env):
    headers_a = env.auth_headers  # the env's own eval account
    headers_b = await _register_second_user(env.client, "acl-b@example.com")
    rid_a, doc_a = await _make_rag_with_doc(
        env.client, headers_a, "acl-alpha", "acl-a.txt", TEXT_A
    )
    rid_b, doc_b = await _make_rag_with_doc(
        env.client, headers_b, "acl-beta", "acl-b.txt", TEXT_B
    )
    return headers_a, headers_b, rid_a, rid_b, doc_a, doc_b


class TestSearchIsolationDetailed:
    async def test_cross_tenant_search_returns_404(self, tmp_path):
        from evals.harness.environment import SqliteEvalEnv

        async with SqliteEvalEnv(
            corpora=[], embedding="local", llm="hash",
        ) as env:
            headers_a, headers_b, rid_a, _, _, _ = await _two_tenant_fixture(env)
            resp = await env.client.post(
                f"/api/v1/rags/{rid_a}/search",
                headers=headers_b,
                json={"query": "retro encabulator", "top_k": 5},
            )
            assert resp.status_code == 404


async def test_hybrid_bm25_leg_does_not_leak_across_tenants():
    from evals.harness.environment import SqliteEvalEnv

    async with SqliteEvalEnv(
        corpora=["marvel"], embedding="local", llm="hash",
        corpus_size=8, seed=3, search_mode="hybrid",
    ) as env:
        headers_a, headers_b, rid_a, rid_b, doc_a, doc_b = (
            await _two_tenant_fixture(env)
        )
        query = "secret lives building"
        hits_b = (
            await env.client.post(
                f"/api/v1/rags/{rid_b}/search",
                headers=headers_b,
                json={"query": query, "top_k": 10},
            )
        ).json()["hits"]
        assert hits_b, "B's own corpus must be retrievable"
        for h in hits_b:
            assert "alpha" not in h["chunk_text"]
            assert h["document_id"] == doc_b

        hits_a = (
            await env.client.post(
                f"/api/v1/rags/{rid_a}/search",
                headers=headers_a,
                json={"query": query, "top_k": 10},
            )
        ).json()["hits"]
        assert hits_a
        for h in hits_a:
            assert "beta" not in h["chunk_text"]
            assert h["document_id"] == doc_a


class TestDocumentMutationIsolation:
    async def test_cross_tenant_update_rollback_delete_404(self):
        from evals.harness.environment import SqliteEvalEnv

        async with SqliteEvalEnv(
            corpora=[], embedding="local", llm="hash", document_update_strategy="replace",
        ) as env:
            headers_a, headers_b, _, _, doc_a, _ = await _two_tenant_fixture(env)
            new_bytes = b"attacker controlled replacement content"
            for method, url in [
                ("PUT", f"/api/v1/documents/{doc_a}/content"),
                ("POST", f"/api/v1/documents/{doc_a}/rollback"),
                ("DELETE", f"/api/v1/documents/{doc_a}"),
                ("GET", f"/api/v1/documents/{doc_a}"),
            ]:
                kwargs = {"headers": headers_b}
                if method == "PUT":
                    kwargs["files"] = {
                        "file": ("x.txt", new_bytes, "text/plain")
                    }
                resp = await env.client.request(method, url, **kwargs)
                assert resp.status_code == 404, (method, url, resp.status_code)

    async def test_cross_tenant_chat_is_rag_not_found(self):
        from evals.harness.environment import SqliteEvalEnv

        async with SqliteEvalEnv(
            corpora=[], embedding="local", llm="hash",
        ) as env:
            headers_a, headers_b, rid_a, _, _, _ = await _two_tenant_fixture(env)
            async with env.client.stream(
                "POST",
                f"/api/v1/rags/{rid_a}/chat",
                headers=headers_b,
                json={"message": "tell me the secret"},
            ) as resp:
                events = []
                async for line in resp.aiter_lines():
                    if line.strip():
                        events.append(json.loads(line))
            # Either the NDJSON stream error (code=rag_not_found) or a
            # plain {"detail": ...} 404 — both deny the foreign tenant,
            # and no assistant tokens may ever stream.
            assert not any(e.get("type") == "token" for e in events)
            assert events, "denial must produce a response body"
            first = events[0]
            if first.get("type") == "error":
                assert first["code"] == "rag_not_found"
            elif "error" in first:  # global HTTP error envelope
                assert first["error"]["code"] == "not_found"
            else:
                assert first.get("detail"), events


class TestMemoryIsolation:
    async def test_memories_never_cross_user_or_rag(self):
        from evals.harness.environment import SqliteEvalEnv
        from sqlalchemy import select

        from app.db import get_sessionmaker
        from app.models import MemoryFact, User
        from app.services.chat_service import _fetch_user_memories

        async with SqliteEvalEnv(
            corpora=[], embedding="local", llm="hash",
        ) as env:
            headers_a, headers_b, rid_a, rid_b, _, _ = await _two_tenant_fixture(env)
            session = get_sessionmaker()()
            try:
                user_a = (
                    await session.execute(
                        select(User.id).where(
                            User.email == env.auth_headers and True or None
                        )
                    )
                )
                # resolve by the known eval email pattern: the env's own
                # account is the only user before B registered.
                users = (
                    await session.execute(select(User.email, User.id))
                ).all()
                emails = {e: i for e, i in users}
                user_a = emails[[e for e in emails if e != "acl-b@example.com"][0]]
                user_b = emails["acl-b@example.com"]

                session.add(
                    MemoryFact(
                        user_id=user_a,
                        rag_id=rid_a,
                        content="Tenant ALPHA private fact: vault code 7777.",
                    )
                )
                await session.commit()

                assert await _fetch_user_memories(
                    session, user_id=user_a, rag_id=rid_a
                ) == ["Tenant ALPHA private fact: vault code 7777."]
                assert (
                    await _fetch_user_memories(
                        session, user_id=user_b, rag_id=rid_b
                    )
                ) == []
                # Same user, different rag — still isolated (scope is AND).
                assert (
                    await _fetch_user_memories(
                        session, user_id=user_a, rag_id=rid_b
                    )
                ) == []
            finally:
                await session.close()


class TestBluegreenChainACL:
    async def test_bluegreen_chain_stays_within_tenant(self):
        from evals.harness.environment import SqliteEvalEnv

        async with SqliteEvalEnv(
            corpora=[], embedding="local", llm="hash", document_update_strategy="bluegreen",
        ) as env:
            headers_a, headers_b, rid_a, _, doc_a, _ = await _two_tenant_fixture(env)

            resp_b = await env.client.put(
                f"/api/v1/documents/{doc_a}/content",
                headers=headers_b,
                files={"file": ("x.txt", b"attacker content", "text/plain")},
            )
            assert resp_b.status_code == 404

            resp = await env.client.put(
                f"/api/v1/documents/{doc_a}/content",
                headers=headers_a,
                files={
                    "file": ("acl-a.txt", TEXT_A + " UPDATED v2", "text/plain")
                },
            )
            assert resp.status_code == 202, resp.text
            new_id = resp.json()["new_document_id"]

            detail_b = await env.client.get(
                f"/api/v1/documents/{new_id}", headers=headers_b
            )
            assert detail_b.status_code == 404

            rb_b = await env.client.post(
                f"/api/v1/documents/{new_id}/rollback", headers=headers_b
            )
            assert rb_b.status_code == 404

            rb_a = await env.client.post(
                f"/api/v1/documents/{new_id}/rollback", headers=headers_a
            )
            assert rb_a.status_code == 200
            hits = (
                await env.client.post(
                    f"/api/v1/rags/{rid_a}/search",
                    headers=headers_a,
                    json={"query": "retro encabulator", "top_k": 5},
                )
            ).json()["hits"]
            assert hits and all("UPDATED v2" not in h["chunk_text"] for h in hits)
