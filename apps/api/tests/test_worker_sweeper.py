"""Phase 6 — worker sweeper tests.

Sweep marks ``PROCESSING`` documents older than the threshold as
``FAILED`` so a worker that died between ``mark_processing`` and
``mark_ready`` / ``mark_failed`` doesn't leave rows stuck forever.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the Worker's ``app.lifespan`` module importable from this
# API test directory. We exercise the sweeper against the API's
# in-memory SQLite (no live Postgres), then test the SQL the
# sweeper actually runs.
_WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(_WORKER_DIR) in sys.path:
    sys.path.remove(str(_WORKER_DIR))
sys.path.insert(0, str(_WORKER_DIR))

from datetime import datetime, timedelta, timezone  # noqa: E402

from sqlalchemy import select, update  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db import Base  # noqa: E402
from app.models import Document, Rag, User  # noqa: E402


async def _make_db():
    """In-memory SQLite with the same schema as the API.

    Returns ``(sessionmaker, engine)``. The caller is responsible
    for disposing the engine after the test.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, autoflush=False
    )
    return factory, engine


async def _seed_factory(
    factory,
    *,
    n_old: int = 0,
    n_fresh: int = 0,
    n_ready: int = 0,
) -> None:
    async with factory() as session:  # type: AsyncSession
        user = User(
            email="sweeper@example.com",
            password_hash="x",
            is_active=True,
        )
        session.add(user)
        await session.flush()
        rag = Rag(user_id=user.id, name="r", description="", status="ACTIVE")
        session.add(rag)
        await session.flush()
        old_threshold = datetime.now(timezone.utc) - timedelta(minutes=10)
        for i in range(n_old):
            session.add(
                Document(
                    rag_id=rag.id,
                    filename=f"old-{i}.txt",
                    file_type="txt",
                    file_size=1,
                    storage_key=f"old-{i}",
                    status="PROCESSING",
                    updated_at=old_threshold,
                )
            )
        for i in range(n_fresh):
            session.add(
                Document(
                    rag_id=rag.id,
                    filename=f"fresh-{i}.txt",
                    file_type="txt",
                    file_size=1,
                    storage_key=f"fresh-{i}",
                    status="PROCESSING",
                )
            )
        for i in range(n_ready):
            session.add(
                Document(
                    rag_id=rag.id,
                    filename=f"ready-{i}.txt",
                    file_type="txt",
                    file_size=1,
                    storage_key=f"ready-{i}",
                    status="READY",
                    updated_at=old_threshold,
                )
            )
        await session.commit()


async def _count_processing(factory) -> int:
    async with factory() as session:  # type: AsyncSession
        rows = (
            await session.execute(
                select(Document).where(Document.status == "PROCESSING")
            )
        ).scalars().all()
        return len(rows)


async def _count_failed(factory) -> int:
    async with factory() as session:  # type: AsyncSession
        rows = (
            await session.execute(
                select(Document).where(Document.status == "FAILED")
            )
        ).scalars().all()
        return len(rows)


async def _apply_sweeper_sql(factory, *, threshold_seconds: int) -> int:
    """The same UPDATE the sweeper runs, isolated from the
    ``create_async_engine`` round-trip the real sweeper does.

    SQLAlchemy's ``Update`` does not support ``.limit()`` cross-
    dialect; the production sweeper relies on the engine's batch
    behavior (PostgreSQL respects it, SQLite ignores it). For the
    test we drop the limit because the seeded row counts are small.
    """
    threshold = datetime.now(timezone.utc) - timedelta(seconds=threshold_seconds)
    async with factory() as session:  # type: AsyncSession
        result = await session.execute(
            update(Document)
            .where(
                Document.status == "PROCESSING",
                Document.updated_at < threshold,
            )
            .values(
                status="FAILED",
                error_message="sweeper: worker died mid-processing",
            )
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    return result.rowcount or 0


async def test_sweeper_marks_old_processing_rows_failed() -> None:
    factory, engine = await _make_db()
    try:
        await _seed_factory(factory, n_old=3, n_fresh=2, n_ready=1)
        moved = await _apply_sweeper_sql(factory, threshold_seconds=60)
        assert moved == 3
        # 3 PROCESSING → FAILED, 2 fresh PROCESSING stay.
        assert await _count_processing(factory) == 2
        assert await _count_failed(factory) == 3
    finally:
        await engine.dispose()


async def test_sweeper_does_not_touch_fresh_processing() -> None:
    factory, engine = await _make_db()
    try:
        await _seed_factory(factory, n_old=1, n_fresh=4)
        moved = await _apply_sweeper_sql(factory, threshold_seconds=60)
        assert moved == 1
        # The 4 fresh PROCESSING rows are untouched.
        assert await _count_processing(factory) == 4
    finally:
        await engine.dispose()


async def test_sweeper_is_idempotent() -> None:
    """A second pass touches no extra rows."""
    factory, engine = await _make_db()
    try:
        await _seed_factory(factory, n_old=2)
        first = await _apply_sweeper_sql(factory, threshold_seconds=60)
        second = await _apply_sweeper_sql(factory, threshold_seconds=60)
        assert first == 2
        assert second == 0
    finally:
        await engine.dispose()


async def test_sweeper_does_not_touch_old_ready_rows() -> None:
    """A READY row older than the threshold must stay READY."""
    factory, engine = await _make_db()
    try:
        await _seed_factory(factory, n_old=2, n_ready=3)
        moved = await _apply_sweeper_sql(factory, threshold_seconds=60)
        assert moved == 2
        # The 3 READY rows are still READY.
        async with factory() as session:  # type: AsyncSession
            ready = (
                await session.execute(
                    select(Document).where(Document.status == "READY")
                )
            ).scalars().all()
            assert len(ready) == 3
    finally:
        await engine.dispose()
