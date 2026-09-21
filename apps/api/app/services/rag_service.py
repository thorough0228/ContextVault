"""RAG service: CRUD + ownership enforcement.

The cardinal rule: every read/write operation is scoped by ``user_id``.
``rag_id`` alone is never enough — a stray request with another user's
``rag_id`` must be indistinguishable from "not found".
"""
from __future__ import annotations

import logging
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import ForbiddenError, NotFoundError
from app.models import Rag

logger = logging.getLogger(__name__)


async def create_rag(
    db: AsyncSession, *, user_id: str, name: str, description: str
) -> Rag:
    rag = Rag(user_id=user_id, name=name, description=description, status="ACTIVE")
    db.add(rag)
    await db.flush()
    await db.refresh(rag)
    logger.info("rag.create.ok user_id=%s rag_id=%s name=%s", user_id, rag.id, name)
    return rag


async def list_user_rags(db: AsyncSession, *, user_id: str) -> List[Rag]:
    result = await db.scalars(
        select(Rag)
        .where(Rag.user_id == user_id)
        .order_by(Rag.created_at.desc())
    )
    return list(result.all())


async def get_user_rag(
    db: AsyncSession, *, user_id: str, rag_id: str
) -> Rag:
    """Return a RAG only if it belongs to ``user_id``. Otherwise 404.

    Why 404 (not 403): leaking that ``rag_id`` exists under another
    account is itself a tenant-boundary violation. Phase 3 will revisit
    if telemetry shows we need to distinguish "doesn't exist" vs
    "wrong owner" for diagnostics.
    """
    rag = await db.get(Rag, rag_id)
    if rag is None or rag.user_id != user_id:
        raise NotFoundError(f"rag {rag_id} not found")
    return rag


async def update_user_rag(
    db: AsyncSession,
    *,
    user_id: str,
    rag_id: str,
    name: str | None,
    description: str | None,
    status: str | None,
) -> Rag:
    rag = await get_user_rag(db, user_id=user_id, rag_id=rag_id)
    if name is not None:
        rag.name = name
    if description is not None:
        rag.description = description
    if status is not None:
        rag.status = status
    await db.flush()
    # Pull server-side defaults (e.g. ``updated_at``) into the in-memory
    # object so the router's ``model_validate`` doesn't trigger a
    # synchronous lazy-load that the greenlet runtime can't service.
    await db.refresh(rag)
    logger.info("rag.update.ok user_id=%s rag_id=%s", user_id, rag_id)
    return rag


async def delete_user_rag(
    db: AsyncSession, *, user_id: str, rag_id: str
) -> None:
    rag = await get_user_rag(db, user_id=user_id, rag_id=rag_id)
    await db.delete(rag)
    await db.flush()
    logger.info("rag.delete.ok user_id=%s rag_id=%s", user_id, rag_id)