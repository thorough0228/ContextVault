"""Tests for /api/v1/auth/* — register / login / logout / me."""

from __future__ import annotations

import httpx


# ----- 1. 注册成功 ----------------------------------------------------------


async def test_register_success_returns_token_and_user(
    client_with_db: httpx.AsyncClient,
) -> None:
    resp = await client_with_db.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "password123"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["expires_in"] > 0
    assert body["user"]["email"] == "alice@example.com"
    assert body["user"]["is_active"] is True
    assert "id" in body["user"]


# ----- 2. 重复邮箱注册失败 ---------------------------------------------------


async def test_register_duplicate_email_fails(
    client_with_db: httpx.AsyncClient,
) -> None:
    payload = {"email": "dup@example.com", "password": "password123"}
    first = await client_with_db.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 201, first.text

    second = await client_with_db.post("/api/v1/auth/register", json=payload)
    assert second.status_code == 400, second.text
    body = second.json()
    assert body["error"]["code"] == "bad_request"
    assert "email" in str(body["error"]["details"])


async def test_register_duplicate_email_case_insensitive(
    client_with_db: httpx.AsyncClient,
) -> None:
    first = await client_with_db.post(
        "/api/v1/auth/register",
        json={"email": "Foo@Example.com", "password": "password123"},
    )
    assert first.status_code == 201

    second = await client_with_db.post(
        "/api/v1/auth/register",
        json={"email": "foo@example.com", "password": "password123"},
    )
    assert second.status_code == 400


async def test_register_short_password_rejected(
    client_with_db: httpx.AsyncClient,
) -> None:
    resp = await client_with_db.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "password": "short"},
    )
    assert resp.status_code == 422


# ----- 3. 登录成功 ----------------------------------------------------------


async def test_login_success_returns_token(
    client_with_db: httpx.AsyncClient,
) -> None:
    await client_with_db.post(
        "/api/v1/auth/register",
        json={"email": "login@example.com", "password": "password123"},
    )
    resp = await client_with_db.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "password123"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["access_token"]
    assert body["user"]["email"] == "login@example.com"


# ----- 4. 登录失败 ----------------------------------------------------------


async def test_login_wrong_password_fails(
    client_with_db: httpx.AsyncClient,
) -> None:
    await client_with_db.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "password123"},
    )
    resp = await client_with_db.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "WRONG-password"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "unauthorized"


async def test_login_unknown_email_fails(
    client_with_db: httpx.AsyncClient,
) -> None:
    resp = await client_with_db.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "password123"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


# ----- /me behaviour --------------------------------------------------------


async def test_me_requires_token(client_with_db: httpx.AsyncClient) -> None:
    resp = await client_with_db.get("/api/v1/auth/me")
    assert resp.status_code == 401


async def test_me_rejects_bogus_token(client_with_db: httpx.AsyncClient) -> None:
    resp = await client_with_db.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer this.is.not.a.real.token"},
    )
    assert resp.status_code == 401


async def test_me_returns_authenticated_user(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers = await auth_headers_factory("me@example.com")
    resp = await client_with_db.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "me@example.com"


# ----- logout --------------------------------------------------------------


async def test_logout_returns_ok_for_authenticated(
    client_with_db: httpx.AsyncClient,
    auth_headers_factory,
) -> None:
    headers = await auth_headers_factory("bye@example.com")
    resp = await client_with_db.post("/api/v1/auth/logout", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_logout_requires_auth(client_with_db: httpx.AsyncClient) -> None:
    resp = await client_with_db.post("/api/v1/auth/logout")
    assert resp.status_code == 401

async def test_register_reserved_tld_email_rejected(
    client_with_db: httpx.AsyncClient,
) -> None:
    """Input email must be validated as EmailStr — a bad address must be
    rejected with 422 at the door, not stored and only then blown up as a
    500 when serializing ``UserPublic`` (regression: ``@test.local``)."""
    resp = await client_with_db.post(
        "/api/v1/auth/register",
        json={"email": "bad@test.local", "password": "good-password-123"},
    )
    assert resp.status_code == 422
