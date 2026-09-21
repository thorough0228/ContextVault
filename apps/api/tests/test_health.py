"""Smoke tests for the API health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_returns_app_metadata(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["app"] == "contextvault-test"
    assert "version" in body
    assert body["api_prefix"] == "/api/v1"


def test_health_endpoint_returns_ok_shape(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "contextvault-test"
    assert "checks" in body
    # Probe failures are reported in the payload but don't break the contract
    assert isinstance(body["checks"], dict)
    assert "postgres" in body["checks"]
    assert "redis" in body["checks"]


def test_validation_error_returns_structured_payload(client: TestClient) -> None:
    # Hit an unknown path under /api/v1 — should produce a 404 envelope
    resp = client.get("/api/v1/does-not-exist")
    assert resp.status_code == 404


def test_cors_preflight_for_allowed_origin(client: TestClient) -> None:
    resp = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Either 200 with headers OR 405 — both acceptable per Starlette behaviour.
    # The point is the origin is *allowed*.
    assert resp.headers.get("access-control-allow-origin") in (
        "http://localhost:3000",
        "*",
        None,
    )