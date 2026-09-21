"""SQLAlchemy async engine + session factory.

Session lifecycle is bound to a request via the FastAPI dependency
:func:`get_db`. The engine itself is module-level and created lazily so
that ``Settings`` can be patched in tests before first use.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base. Models inherit from this (Phase 2: User, Rag)."""


_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    """Lazily build the async engine from current settings."""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.async_database_url
        engine_kwargs: dict
        if url.startswith("sqlite"):
            # SQLite (used in tests) needs StaticPool + cross-thread allowance.
            from sqlalchemy.pool import StaticPool

            engine_kwargs = {
                "echo": False,
                "connect_args": {"check_same_thread": False},
                "poolclass": StaticPool,
            }
        else:
            engine_kwargs = {
                "echo": False,
                "pool_pre_ping": True,
                "pool_size": 5,
                "max_overflow": 10,
            }
        _engine = create_async_engine(url, **engine_kwargs)
        logger.info(
            "db.engine.ready host=%s db=%s",
            settings.postgres_host,
            settings.postgres_db,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a transactional async session.

    Phase 2 wraps each request in a transaction: the session is
    committed on clean exit and rolled back on exception. Routers do
    **not** call ``session.commit()`` themselves — the boundary is
    here.
    """
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def dispose_engine() -> None:
    """Tear down the engine on shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        logger.info("db.engine.disposed")
    _engine = None
    _sessionmaker = None


def reset_engine_for_tests() -> None:
    """Clear cached engine/sessionmaker. Tests call this after
    overriding settings or between SQLite in-memory sessions."""
    global _engine, _sessionmaker
    _engine = None
    _sessionmaker = None