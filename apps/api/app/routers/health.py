"""Health probe for the API and its critical dependencies.

Per spec, ``GET /api/v1/health`` returns 200 when the service and its
probed dependencies are healthy, and a structured payload describing
each probe's status. Redis and Postgres are probed best-effort: a probe
failure is reported in the payload but does NOT flip the overall status
to unhealthy — Phase 1 just needs to *observe* dependency reachability.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_db
from app.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)


async def _probe_postgres(session: AsyncSession) -> Dict[str, Any]:
    try:
        await asyncio.wait_for(session.execute(text("SELECT 1")), timeout=2.0)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        logger.warning("health.probe.postgres failed: %s", exc)
        return {"ok": False, "error": str(exc)}


async def _probe_redis(settings: Settings) -> Dict[str, Any]:
    try:
        import redis.asyncio as redis_asyncio

        client = redis_asyncio.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            socket_timeout=1.5,
            socket_connect_timeout=1.5,
        )
        try:
            pong = await client.ping()
            return {"ok": bool(pong)}
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001
        logger.warning("health.probe.redis failed: %s", exc)
        return {"ok": False, "error": str(exc)}


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health probe",
)
async def health(
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Liveness + dependency probes. Always returns 200 in Phase 1."""

    pg, rd = await asyncio.gather(
        _probe_postgres(db),
        _probe_redis(settings),
    )

    checks = {
        "postgres": pg,
        "redis": rd,
    }

    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        checks=checks,
    )