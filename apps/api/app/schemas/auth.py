"""Auth-related request/response schemas."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, EmailStr, Field, StringConstraints

from app.schemas.user import UserPublic


_EmailStr = Annotated[
    EmailStr, StringConstraints(strip_whitespace=True, to_lower=True)
]


class RegisterRequest(BaseModel):
    """POST /api/v1/auth/register payload."""

    email: _EmailStr = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """POST /api/v1/auth/login payload."""

    email: _EmailStr = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """POST /auth/register + POST /auth/login response."""

    access_token: str
    token_type: str = Field(default="bearer")
    expires_in: int  # seconds
    user: UserPublic


class LogoutResponse(BaseModel):
    """POST /api/v1/auth/logout response.

    Phase 2 uses stateless JWT, so ``logout`` is effectively a
    client-side discard; the endpoint exists so the front-end has a
    canonical URL to call and so future revocation lists have a
    single hook.
    """

    status: str = "ok"
    message: str = "logged out"