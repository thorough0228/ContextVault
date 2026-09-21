"""Security primitives: password hashing + JWT signing/verification.

Passwords use bcrypt directly (not passlib) — passlib 1.7.x doesn't
recognise bcrypt 5.x and pulling in the older 4.x line is a smaller
dependency than the workaround. JWTs are HS256 with ``sub`` = user id;
``exp`` is set from ``JWT_EXPIRES_MINUTES``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bcrypt
import jwt

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---- Password hashing -----------------------------------------------------

# bcrypt has a 72-byte input limit. We pre-hash with SHA-256 so that
# arbitrarily long passphrases still work and so that the bcrypt salt
# is the only thing that varies. The SHA-256 step is constant-time per
# message and is *not* the security bottleneck.
import hashlib


def _prepare(password: str) -> bytes:
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    return hashlib.sha256(password.encode("utf-8")).digest()


def hash_password(password: str) -> str:
    """Return a bcrypt hash string for ``password``."""
    digest = _prepare(password)
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(digest, salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time bcrypt check. Returns False on any error."""
    try:
        digest = _prepare(password)
        return bcrypt.checkpw(digest, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---- JWT ------------------------------------------------------------------


class InvalidTokenError(Exception):
    """Raised when a JWT cannot be decoded or fails validation."""


def create_access_token(
    subject: str,
    extra_claims: Dict[str, Any] | None = None,
    expires_minutes: int | None = None,
) -> str:
    """Sign a JWT with ``sub=subject`` and the configured expiry."""
    settings = get_settings()
    minutes = expires_minutes if expires_minutes is not None else settings.jwt_expires_minutes
    now = datetime.now(tz=timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
        "iss": settings.app_name,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Verify signature + expiry. Raises :class:`InvalidTokenError` on failure."""
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError(f"invalid token: {exc}") from exc