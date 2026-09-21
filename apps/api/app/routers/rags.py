"""RAG router: CRUD with strict user-id ownership.

Security boundary:

1. ``get_current_user`` resolves the caller from the JWT.
2. ``user_id = current.id`` — never read from URL, body, or query.
3. ``rag_id`` from the path is validated *together with* ``user_id``
   inside :mod:`app.services.rag_service`, which raises 404 if the
   rag doesn't belong to the caller.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps.auth import get_current_user
from app.models import User
from app.schemas.rag import (
    RagCreateRequest,
    RagListResponse,
    RagPatchRequest,
    RagResponse,
)
from app.services import rag_service

router = APIRouter(prefix="/rags", tags=["rags"])
logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=RagResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new RAG owned by the caller",
)
async def create_rag(
    payload: RagCreateRequest,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RagResponse:
    rag = await rag_service.create_rag(
        db,
        user_id=current.id,
        name=payload.name,
        description=payload.description,
    )
    return RagResponse.model_validate(rag)


@router.get(
    "",
    response_model=RagListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all RAGs owned by the caller",
)
async def list_rags(
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RagListResponse:
    rags = await rag_service.list_user_rags(db, user_id=current.id)
    return RagListResponse(
        items=[RagResponse.model_validate(r) for r in rags],
        total=len(rags),
    )


@router.get(
    "/{rag_id}",
    response_model=RagResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch one of the caller's RAGs by id",
)
async def get_rag(
    rag_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RagResponse:
    rag = await rag_service.get_user_rag(db, user_id=current.id, rag_id=rag_id)
    return RagResponse.model_validate(rag)


@router.patch(
    "/{rag_id}",
    response_model=RagResponse,
    status_code=status.HTTP_200_OK,
    summary="Update one of the caller's RAGs",
)
async def patch_rag(
    rag_id: str,
    payload: RagPatchRequest,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RagResponse:
    rag = await rag_service.update_user_rag(
        db,
        user_id=current.id,
        rag_id=rag_id,
        name=payload.name,
        description=payload.description,
        status=payload.status,
    )
    return RagResponse.model_validate(rag)


@router.delete(
    "/{rag_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete one of the caller's RAGs",
)
async def delete_rag(
    rag_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    await rag_service.delete_user_rag(db, user_id=current.id, rag_id=rag_id)
    return None