"""Tests for the request-id middleware + log filter."""

from __future__ import annotations

import json
import logging
import re

import httpx
import pytest

from app.main import create_app
from app.db import get_db
from app.middleware.request_id import (
    REQUEST_ID_HEADER,
    get_request_id,
)


# ----- O.1 RequestIdMiddleware behaviour ------------------------------


async def test_request_id_propagated_from_header(
    app_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_with_db),
        base_url="http://testserver",
    ) as client:
        resp = await client.get(
            "/api/v1/rags",
            headers={**headers, "X-Request-Id": "test-id-123"},
        )
    assert resp.status_code == 200
    assert resp.headers.get(REQUEST_ID_HEADER) == "test-id-123"


async def test_request_id_generated_when_absent(
    app_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_with_db),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/api/v1/rags", headers=headers)
    assert resp.status_code == 200
    rid = resp.headers.get(REQUEST_ID_HEADER)
    assert rid is not None
    # uuid4 format
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", rid
    )


async def test_request_id_present_in_413_envelope(test_settings, app_with_db) -> None:
    """413 from SizeLimitMiddleware should still carry the X-Request-Id
    header — SizeLimitMiddleware is layered INSIDE the request-id
    middleware but its response goes out through it."""
    from app.main import create_app
    from app.db import get_db

    tiny_settings = test_settings.model_copy(update={"upload_max_bytes": 64})
    small_app = create_app(tiny_settings)
    small_app.dependency_overrides[get_db] = app_with_db.dependency_overrides[get_db]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=small_app),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/v1/rags",
            headers={
                "Authorization": "Bearer x",
                "Content-Type": "application/json",
                "X-Request-Id": "abc-413",
            },
            content=b"x" * 200,
        )
    assert resp.status_code == 413
    assert resp.headers.get(REQUEST_ID_HEADER) == "abc-413"


def test_get_request_id_default_none_outside_request() -> None:
    """Outside a request, the ContextVar is None — code that reads it
    gets a safe default rather than a leaked value from a prior request."""
    assert get_request_id() is None
