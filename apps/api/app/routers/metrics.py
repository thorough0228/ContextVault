"""Phase 6 — operational metrics.

``GET /api/v1/metrics`` returns the in-memory metrics registry
rendered as Prometheus text format v0.0.4. Admin-only (current
``User.is_admin`` must be true).

The endpoint deliberately sits behind auth so we don't leak
operational data (queue depth, latency histograms) to unauthenticated
callers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps.auth import get_current_user
from app.exceptions import ForbiddenError
from app.metrics import render as render_metrics
from app.models import User

router = APIRouter(tags=["metrics"])
logger = logging.getLogger(__name__)


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Operational metrics (admin only, Prometheus text format v0.0.4)",
)
async def metrics(
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    if not current.is_admin:
        raise ForbiddenError("admin only")
    body = render_metrics()
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
