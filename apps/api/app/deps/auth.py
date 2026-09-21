"""Auth dependencies: ``get_current_user`` and ``get_current_user_id``.

These are the **only** place where a request learns who the caller is.
Every RAG endpoint pulls ``user_id`` from one of these — never from
query params, body, or path components supplied by the client.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_db
from app.exceptions import UnauthorizedError
from app.models import User
from app.security import InvalidTokenError, decode_access_token

logger = logging.getLogger(__name__)

_BEARER = "bearer"


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise UnauthorizedError("missing Authorization header")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != _BEARER or not parts[1].strip():
        raise UnauthorizedError("Authorization header must be 'Bearer <token>'")
    return parts[1].strip()


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve the calling :class:`User` from the bearer token."""
    token = _extract_token(authorization)
    try:
        payload = decode_access_token(token)
    except InvalidTokenError as exc:
        raise UnauthorizedError(str(exc)) from exc

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedError("token missing subject")

    user = await db.scalar(select(User).where(User.id == user_id))
    if user is None:
        # Token signed correctly but user gone. Don't leak whether the
        # user existed historically — just say unauthorized.
        raise UnauthorizedError("user no longer exists")
    if not user.is_active:
        raise UnauthorizedError("user is inactive")
    return user


async def get_current_user_id(
    current: User = Depends(get_current_user),
) -> str:
    """Cheap variant that returns just the user id. Use this when a
    router doesn't need the full :class:`User` object."""
    return current.id