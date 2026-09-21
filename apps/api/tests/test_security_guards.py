"""Phase 6 hardening — settings / CORS / size-limit guard tests."""

from __future__ import annotations

import httpx
import pytest

from app.config import Settings


# ----- P0.1 default-secret boot guard ----------------------------------


def test_default_jwt_secret_rejected_in_production_env() -> None:
    with pytest.raises(ValueError) as exc:
        Settings(app_env="production", jwt_secret="change-me-locally")
    assert "JWT_SECRET" in str(exc.value)


def test_default_jwt_secret_rejected_in_staging_env() -> None:
    with pytest.raises(ValueError) as exc:
        Settings(app_env="staging", jwt_secret="change-me-locally")
    assert "JWT_SECRET" in str(exc.value)


def test_default_jwt_secret_allowed_in_development() -> None:
    s = Settings(app_env="development", jwt_secret="change-me-locally")
    assert s.jwt_secret == "change-me-locally"


def test_default_jwt_secret_allowed_in_test_env() -> None:
    s = Settings(app_env="test", jwt_secret="change-me-locally")
    assert s.jwt_secret == "change-me-locally"


def test_default_postgres_password_rejected_in_production_env() -> None:
    with pytest.raises(ValueError) as exc:
        Settings(
            app_env="production",
            jwt_secret="real-jwt-secret",
            postgres_password="postgres",
        )
    assert "POSTGRES_PASSWORD" in str(exc.value)


def test_default_s3_secret_rejected_in_production_env() -> None:
    with pytest.raises(ValueError) as exc:
        Settings(
            app_env="production",
            jwt_secret="real-jwt-secret",
            postgres_password="real-pg",
            s3_secret_key="change-me-locally",
        )
    assert "S3_SECRET_KEY" in str(exc.value)


def test_real_secrets_pass_in_production_env() -> None:
    s = Settings(
        app_env="production",
        jwt_secret="real-jwt-secret",
        postgres_password="real-pg",
        s3_secret_key="real-s3",
    )
    assert s.jwt_secret == "real-jwt-secret"


# ----- P0.2 CORS wildcard origin guard ----------------------------------


def test_cors_rejects_wildcard_origin() -> None:
    with pytest.raises(ValueError) as exc:
        Settings(cors_allow_origins=["*"])
    assert "CORS_ALLOW_ORIGINS" in str(exc.value)


def test_cors_rejects_wildcard_in_csv() -> None:
    # The before-mode validator splits the CSV; the after-mode
    # validator must still catch the resulting ``*``.
    with pytest.raises(ValueError):
        Settings(cors_allow_origins="http://a.example,*,http://b.example")


def test_cors_accepts_explicit_origins() -> None:
    s = Settings(
        cors_allow_origins=["http://a.example", "http://b.example"]
    )
    assert s.cors_allow_origins == ["http://a.example", "http://b.example"]


# ----- P0.3 Content-Length pre-check middleware --------------------------


async def test_oversized_content_length_rejected_before_handler(
    test_settings, app_with_db, auth_headers_factory
) -> None:
    """A POST with Content-Length > upload_max_bytes should be
    rejected with 413 by the middleware, before the body is read."""
    # The test_settings fixture uses a 10 MiB cap; the body below is
    # 200 bytes, so the cap has to come down to make the test
    # exercise the reject branch. Easiest path: build a fresh app
    # with a tiny cap and reuse the test DB.
    from app.main import create_app
    from app.db import get_db

    tiny_settings = test_settings.model_copy(update={"upload_max_bytes": 64})
    small_app = create_app(tiny_settings)
    small_app.dependency_overrides[get_db] = app_with_db.dependency_overrides[get_db]

    headers = await auth_headers_factory("alice@example.com")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=small_app),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/v1/rags",
            headers={**headers, "Content-Type": "application/json"},
            content=b"x" * 200,
        )
    assert resp.status_code == 413, resp.text
    body = resp.json()
    assert body["error"]["code"] == "payload_too_large"
    assert "exceeds" in body["error"]["message"]


async def test_normal_post_passes_size_limit(
    app_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("bob@example.com")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_with_db),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/v1/rags",
            headers=headers,
            json={"name": "ok", "description": ""},
        )
    assert resp.status_code == 201, resp.text
