"""Pytest fixtures shared by API tests.

Phase 2 adds SQLite-backed fixtures so the auth + RAG tests can run
against a real (in-memory) database without spinning up Postgres.
Tests run async (pytest-asyncio auto mode) and use ``httpx.AsyncClient``
against the ASGI app directly.

Phase 3 also stubs MinIO/S3 with an in-memory storage and points the
Celery tasks at the SQLite URL so the upload → worker → READY pipeline
can be exercised without external services.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Awaitable, Callable

# Phase 3 needs the worker task to share the same database the API
# uses, and the worker constructs its own engine — so we use a
# file-backed SQLite (the API fixture overrides the path per test).
os.environ.setdefault("STORAGE_IN_MEMORY", "true")
os.environ.setdefault(
    "CELERY_BROKER_URL", "memory://",
)
os.environ.setdefault(
    "CELERY_RESULT_BACKEND", "cache+memory://",
)
# Tests must be deterministic regardless of the developer's .env —
# real providers would make network calls from inside unit tests, and
# model/dimension assertions assume the offline defaults.
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("EMBEDDING_MODEL", "hash-256")
os.environ.setdefault("EMBEDDING_DIMENSION", "256")
os.environ.setdefault("LLM_PROVIDER", "hash")
os.environ.setdefault("LLM_MODEL", "hash-llm")
# Phase 12: keep the offline test path deterministic — memory
# extraction calls the LLM provider after each turn.
os.environ.setdefault("MEMORY_EXTRACTION_ENABLED", "0")
os.environ.setdefault("MEMORY_INJECTION_ENABLED", "0")

import httpx
import pytest
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.db import Base, get_db, reset_engine_for_tests
from app.embedding import (
    reset_provider_for_tests,
    set_provider_for_tests,
)
from app.embedding.hash_provider import HashEmbeddingProvider
from app.llm import (
    reset_provider_for_tests as reset_llm_for_tests,
    set_provider_for_tests as set_llm_for_tests,
)
from app.llm.hash_provider import HashLLMProvider
from app.main import create_app
from app.storage import (
    get_storage,
    reset_storage_for_tests,
    set_storage_for_tests,
)
from app.storage.memory import InMemoryStorage
from app.celery_client import celery_app


# ---------------------------------------------------------------------------
# Settings + app
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Settings with safe local defaults — never touches real infra.

    Each test gets a fresh on-disk SQLite file so the API engine and
    the worker engine (which the task spawns internally) both see the
    same schema without colliding across tests.
    """
    db_path = tmp_path / "test.sqlite"
    return Settings(
        app_env="test",
        app_name="contextvault-test",
        postgres_host="localhost",
        postgres_port=5432,
        redis_host="localhost",
        redis_port=6379,
        cors_allow_origins=["http://localhost:3000"],
        database_url=f"sqlite+aiosqlite:///{db_path}",
        jwt_secret="test-secret-do-not-use-in-prod",
        jwt_algorithm="HS256",
        jwt_expires_minutes=60,
        # Phase 3: keep uploads off MinIO during tests.
        storage_in_memory=True,
        upload_max_bytes=10 * 1024 * 1024,
        chunk_size_chars=200,
        chunk_overlap_chars=40,
        # Phase 4: keep the embedding small so the tests' matrix
        # checks stay cheap. Production should set this to whatever
        # the real model actually returns.
        embedding_provider="hash",
        embedding_model="hash-test",
        embedding_dimension=32,
        embedding_batch_size=8,
    )


@pytest.fixture
def app(test_settings: Settings):
    """Build a fresh app instance per test — no shared state."""
    application = create_app(test_settings)
    application.dependency_overrides[get_settings] = lambda: test_settings
    yield application
    application.dependency_overrides.clear()
    reset_engine_for_tests()


# ---------------------------------------------------------------------------
# Stub-DB client (Phase 1 health probe — no real DB required)
# ---------------------------------------------------------------------------


class _FakeSession:
    """Minimal async session stub for health probes."""

    async def execute(self, _stmt):
        from sqlalchemy.exc import OperationalError

        raise OperationalError("select 1", {}, Exception("no live db in test"))

    async def close(self):
        return None


@pytest.fixture
def client(app):
    """Synchronous TestClient with a stub DB session.

    Phase 1 health probes don't need a real DB — they use the fake
    session so the endpoint reports ``ok=false`` for the probe map
    without spinning up infrastructure.
    """
    from fastapi.testclient import TestClient

    async def _fake_db():
        yield _FakeSession()

    app.dependency_overrides[get_db] = _fake_db
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Ensure tests don't leak cached settings across modules."""
    get_settings.cache_clear()
    # Reset the embedding provider factory too — its instance is built
    # from settings at first access and would otherwise outlive a
    # settings swap.
    reset_provider_for_tests()
    reset_llm_for_tests()
    yield
    get_settings.cache_clear()
    reset_provider_for_tests()
    reset_llm_for_tests()


# ---------------------------------------------------------------------------
# DB-backed fixtures (Phase 2)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine(test_settings: Settings):
    """Per-test SQLite engine that owns its own in-memory database.

    Each test gets a fresh schema; the engine is disposed at teardown
    so the next test starts clean.
    """
    reset_engine_for_tests()
    engine = create_async_engine(
        test_settings.database_url,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()
    reset_engine_for_tests()


@pytest.fixture
async def db_sessionmaker(db_engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
async def db_session(db_sessionmaker) -> AsyncGenerator[AsyncSession, None]:
    async with db_sessionmaker() as session:
        yield session


@pytest.fixture
def app_with_db(app, db_sessionmaker):
    """App whose ``get_db`` yields sessions backed by the test SQLite DB."""

    async def _override_db() -> AsyncGenerator[AsyncSession, None]:
        async with db_sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_db
    yield app
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def client_with_db(
    app_with_db, test_settings: Settings
) -> AsyncGenerator[httpx.AsyncClient, None]:
    # Swap in a per-test in-memory storage so document upload +
    # download paths work without MinIO.
    storage = InMemoryStorage()
    set_storage_for_tests(storage)

    # Pre-warm the embedding provider with this test's settings so
    # the embedding dimension matches what the test asserts on.
    set_provider_for_tests(
        HashEmbeddingProvider(
            dimension=test_settings.embedding_dimension,
            model=test_settings.embedding_model,
        )
    )

    # Run tasks eagerly — no Redis broker required for tests.
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False
    # Tasks read this URL — point it at the same SQLite the test uses.
    celery_app.conf.database_url = test_settings.async_database_url

    # send_task bypasses the eager flag (it always goes through the
    # broker). For tests, route it through apply() so the upload path
    # doesn't need Redis.
    def _send_task(name, args=None, kwargs=None, **other):  # noqa: ARG001
        task = celery_app.tasks.get(name)
        if task is None:
            raise RuntimeError(f"task {name} not registered")
        return task.apply(args=args or (), kwargs=kwargs or {})

    celery_app.send_task = _send_task  # type: ignore[assignment]

    try:
        transport = ASGITransport(app=app_with_db)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client
    finally:
        celery_app.conf.task_always_eager = False
        celery_app.conf.task_eager_propagates = False
        reset_storage_for_tests()


# ---------------------------------------------------------------------------
# Auth header factory
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_headers_factory(
    client_with_db: httpx.AsyncClient,
) -> Callable[[str, str], Awaitable[dict]]:
    """Factory: ``await auth_headers_factory("alice@example.com")`` -> dict.

    Registers a user via the public API and returns the Authorization
    header dict subsequent calls should send.
    """

    async def _factory(email: str, password: str = "password123") -> dict:
        resp = await client_with_db.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        assert resp.status_code == 201, resp.text
        token = resp.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    return _factory