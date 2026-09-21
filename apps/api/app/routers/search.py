"""Search router: POST /api/v1/rags/{rag_id}/search.

Security:
* JWT required (current_user dependency).
* rag ownership verified inside search_service — cross-tenant URLs
  look identical to "not found".
* query embedding happens server-side; the client never sees vectors.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps.auth import get_current_user
from app.embedding import get_embedding_provider
from app.exceptions import (
    BadRequestError,
    NotFoundError,
    ServiceUnavailableError,
)
from app.models import User
from app.schemas.search import SearchRequest, SearchResponse
from app.services import search_service

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)


@router.post(
    "/rags/{rag_id}/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic search inside one of the caller's RAGs",
)
async def search(
    rag_id: str,
    payload: SearchRequest,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    try:
        hits = await search_service.search_rag(
            db,
            user_id=current.id,
            rag_id=rag_id,
            query=payload.query,
            top_k=payload.top_k,
            provider=get_embedding_provider(),
        )
    except search_service.SearchError as exc:
        if exc.code == "empty_query":
            raise BadRequestError("query must be non-empty") from exc
        if exc.code == "embedding_transient":
            raise ServiceUnavailableError(
                f"embedding provider temporarily unavailable: {exc.message}"
            ) from exc
        if exc.code == "embedding_failed":
            raise BadRequestError(f"embedding failed: {exc.message}") from exc
        if exc.code == "rag_not_found":
            raise NotFoundError("rag not found") from exc
        raise BadRequestError(f"search failed: {exc.message}") from exc

    provider = get_embedding_provider()
    return SearchResponse(
        hits=hits,
        query=payload.query,
        top_k=payload.top_k or 0,
        embedding_model=provider.model,
        embedding_dimension=provider.dimension,
    )