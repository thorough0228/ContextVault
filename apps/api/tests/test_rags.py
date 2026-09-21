"""Tests for /api/v1/rags/* — CRUD + cross-user security boundaries."""

from __future__ import annotations

import httpx


# ----- 5. 创建 RAG ----------------------------------------------------------


async def test_create_rag_returns_owned_record(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    resp = await client_with_db.post(
        "/api/v1/rags",
        headers=headers,
        json={"name": "Research Notes", "description": "papers I'm reading"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "Research Notes"
    assert body["description"] == "papers I'm reading"
    assert body["status"] == "ACTIVE"
    assert "id" in body
    assert "user_id" in body


async def test_create_rag_requires_auth(client_with_db: httpx.AsyncClient) -> None:
    resp = await client_with_db.post(
        "/api/v1/rags", json={"name": "anon", "description": ""}
    )
    assert resp.status_code == 401


async def test_create_rag_validates_name(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers = await auth_headers_factory("bob@example.com")
    resp = await client_with_db.post(
        "/api/v1/rags",
        headers=headers,
        json={"name": "", "description": ""},
    )
    assert resp.status_code == 422


# ----- 6. 获取自己的 RAG ---------------------------------------------------


async def test_list_rags_returns_only_caller_owned(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")

    await client_with_db.post(
        "/api/v1/rags", headers=headers_alice, json={"name": "alice-1"}
    )
    await client_with_db.post(
        "/api/v1/rags", headers=headers_alice, json={"name": "alice-2"}
    )
    await client_with_db.post(
        "/api/v1/rags", headers=headers_bob, json={"name": "bob-1"}
    )

    alice_list = await client_with_db.get("/api/v1/rags", headers=headers_alice)
    assert alice_list.status_code == 200
    alice_items = alice_list.json()["items"]
    assert {r["name"] for r in alice_items} == {"alice-1", "alice-2"}

    bob_list = await client_with_db.get("/api/v1/rags", headers=headers_bob)
    bob_items = bob_list.json()["items"]
    assert {r["name"] for r in bob_items} == {"bob-1"}


async def test_get_rag_returns_owned_rag(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    created = await client_with_db.post(
        "/api/v1/rags",
        headers=headers,
        json={"name": "solo"},
    )
    rag_id = created.json()["id"]

    resp = await client_with_db.get(f"/api/v1/rags/{rag_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == rag_id


# ----- 7. 修改自己的 RAG ---------------------------------------------------


async def test_patch_rag_updates_fields(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    created = await client_with_db.post(
        "/api/v1/rags",
        headers=headers,
        json={"name": "before", "description": "old"},
    )
    rag_id = created.json()["id"]

    resp = await client_with_db.patch(
        f"/api/v1/rags/{rag_id}",
        headers=headers,
        json={"name": "after", "status": "PROCESSING"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "after"
    assert body["description"] == "old"  # unchanged
    assert body["status"] == "PROCESSING"


# ----- 8. 删除自己的 RAG ---------------------------------------------------


async def test_delete_rag_removes_it(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    created = await client_with_db.post(
        "/api/v1/rags",
        headers=headers,
        json={"name": "bye"},
    )
    rag_id = created.json()["id"]

    delete_resp = await client_with_db.delete(
        f"/api/v1/rags/{rag_id}", headers=headers
    )
    assert delete_resp.status_code == 204

    follow_up = await client_with_db.get(f"/api/v1/rags/{rag_id}", headers=headers)
    assert follow_up.status_code == 404


# ----- 9. User A 无法读取 User B 的 RAG ----------------------------------


async def test_user_a_cannot_read_user_b_rag(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")

    created = await client_with_db.post(
        "/api/v1/rags",
        headers=headers_alice,
        json={"name": "alice-private"},
    )
    rag_id = created.json()["id"]

    # Bob tries to read Alice's rag — must look like a missing resource.
    resp = await client_with_db.get(f"/api/v1/rags/{rag_id}", headers=headers_bob)
    assert resp.status_code == 404
    # And the response must not leak that this id exists at all.
    assert "alice" not in resp.text.lower()


# ----- 10. User A 无法修改 User B 的 RAG ----------------------------------


async def test_user_a_cannot_modify_user_b_rag(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")

    created = await client_with_db.post(
        "/api/v1/rags",
        headers=headers_alice,
        json={"name": "original"},
    )
    rag_id = created.json()["id"]

    resp = await client_with_db.patch(
        f"/api/v1/rags/{rag_id}",
        headers=headers_bob,
        json={"name": "hijacked"},
    )
    assert resp.status_code == 404

    # Confirm the original rag is unchanged from Alice's perspective.
    after = await client_with_db.get(f"/api/v1/rags/{rag_id}", headers=headers_alice)
    assert after.status_code == 200
    assert after.json()["name"] == "original"


# ----- 11. User A 无法删除 User B 的 RAG ----------------------------------


async def test_user_a_cannot_delete_user_b_rag(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")

    created = await client_with_db.post(
        "/api/v1/rags",
        headers=headers_alice,
        json={"name": "do-not-delete"},
    )
    rag_id = created.json()["id"]

    resp = await client_with_db.delete(f"/api/v1/rags/{rag_id}", headers=headers_bob)
    assert resp.status_code == 404

    # Rag still exists for Alice.
    after = await client_with_db.get(f"/api/v1/rags/{rag_id}", headers=headers_alice)
    assert after.status_code == 200


# ----- extra: user_id in body is ignored -----------------------------------


async def test_user_id_in_payload_is_ignored(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    """Even if a malicious client tries to inject ``user_id`` in the body,
    the rag is always attributed to the authenticated caller."""
    headers = await auth_headers_factory("alice@example.com")

    resp = await client_with_db.post(
        "/api/v1/rags",
        headers=headers,
        json={"name": "x", "description": "y", "user_id": "someone-else"},
    )
    assert resp.status_code == 201, resp.text
    # Find Alice's user id from /me.
    me = await client_with_db.get("/api/v1/auth/me", headers=headers)
    alice_id = me.json()["id"]
    assert resp.json()["user_id"] == alice_id


async def test_rag_endpoints_require_auth(
    client_with_db: httpx.AsyncClient,
) -> None:
    # No bearer token at all.
    assert (await client_with_db.get("/api/v1/rags")).status_code == 401
    assert (
        await client_with_db.post("/api/v1/rags", json={"name": "x"})
    ).status_code == 401
    assert (
        await client_with_db.patch("/api/v1/rags/abc", json={"name": "x"})
    ).status_code == 401
    assert (
        await client_with_db.delete("/api/v1/rags/abc")
    ).status_code == 401