"""Auth service: registration + login flows."""

from __future__ import annotations

import logging
from typing import Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import BadRequestError, UnauthorizedError
from app.models import User
from app.security import (
    create_access_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)


async def register_user(
    db: AsyncSession, *, email: str, password: str
) -> User:
    """Create a new user. Raises :class:`BadRequestError` on duplicate email."""
    # Case-insensitive uniqueness: normalize at the boundary.
    email_normalized = email.strip().lower()

    # Cheap pre-check so we don't waste a bcrypt round on a guaranteed collision.
    existing = await db.scalar(
        select(User).where(User.email == email_normalized)
    )
    if existing is not None:
        raise BadRequestError("email already registered", details={"field": "email"})

    user = User(
        email=email_normalized,
        password_hash=hash_password(password),
        is_active=True,
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError as exc:
        # Race: another concurrent register claimed the same email.
        await db.rollback()
        raise BadRequestError(
            "email already registered", details={"field": "email"}
        ) from exc
    logger.info("auth.register.ok email=%s user_id=%s", user.email, user.id)
    return user


async def authenticate(
    db: AsyncSession, *, email: str, password: str
) -> User:
    """Return the user iff (active) credentials match. Otherwise
    raise :class:`UnauthorizedError` — never leak which side failed."""
    email_normalized = email.strip().lower()
    user = await db.scalar(select(User).where(User.email == email_normalized))
    if user is None:
        # Run a dummy bcrypt to keep timing roughly constant.
        verify_password(password, "$2b$12$" + "0" * 53)
        raise UnauthorizedError("invalid email or password")
    if not user.is_active:
        raise UnauthorizedError("invalid email or password")
    if not verify_password(password, user.password_hash):
        raise UnauthorizedError("invalid email or password")
    return user


def issue_token(user: User) -> Tuple[str, int]:
    """Mint a JWT for ``user`` and return (token, expires_in_seconds)."""
    from app.config import get_settings

    settings = get_settings()
    expires_seconds = settings.jwt_expires_minutes * 60
    token = create_access_token(subject=user.id)
    return token, expires_seconds