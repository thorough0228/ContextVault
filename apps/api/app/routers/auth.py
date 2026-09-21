"""Auth router: register, login, logout, me."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps.auth import get_current_user
from app.models import User
from app.schemas.auth import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    TokenResponse,
)
from app.schemas.user import UserPublic
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user and return an access token",
)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user = await auth_service.register_user(
        db, email=payload.email, password=payload.password
    )
    token, expires_in = auth_service.issue_token(user)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserPublic.model_validate(user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Exchange email + password for an access token",
)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user = await auth_service.authenticate(
        db, email=payload.email, password=payload.password
    )
    token, expires_in = auth_service.issue_token(user)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserPublic.model_validate(user),
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    summary="Drop the current session",
)
async def logout(_: User = Depends(get_current_user)) -> LogoutResponse:
    # JWT is stateless in Phase 2; the client discards the token. A
    # future phase may add a server-side revocation list keyed on
    # ``jti``; the contract here is stable so the front-end won't have
    # to change.
    return LogoutResponse()


@router.get(
    "/me",
    response_model=UserPublic,
    status_code=status.HTTP_200_OK,
    summary="Return the currently authenticated user",
)
async def me(current: User = Depends(get_current_user)) -> UserPublic:
    return UserPublic.model_validate(current)