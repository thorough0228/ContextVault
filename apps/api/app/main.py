"""FastAPI application entrypoint.

This module wires:

* logging configuration
* CORS middleware
* global exception handlers
* the ``/api/v1`` router
* lifecycle hooks for the async DB engine

``create_app`` is the factory — uvicorn imports ``app.main:app`` which is
created at import time for ergonomics, but the factory is also exposed
for tests that need to override settings.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.db import dispose_engine
from app.exceptions import register_exception_handlers
from app.logging_config import configure_logging
from app.routers import api_v1_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    logger.info(
        "api.startup app=%s env=%s version=%s",
        settings.app_name,
        settings.app_env,
        settings.app_version,
    )
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("api.shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="ContextVault Phase 1 — base API skeleton.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = settings

    # Middleware order note: FastAPI/Starlette processes middleware
    # in REVERSE-add order. The LAST ``add_middleware`` call wraps
    # everything else — its request preprocessing fires FIRST and its
    # response postprocessing fires LAST. So:
    #   1. add_middleware(CORSMiddleware, ...)     -> innermost
    #   2. add_middleware(SizeLimitMiddleware)     -> middle (catches
    #         big bodies, returns 413 with standard envelope)
    #   3. add_middleware(RequestIdMiddleware)     -> outermost — every
    #         downstream response, including the 413, gets the
    #         X-Request-Id header echoed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.middleware.size_limit import SizeLimitMiddleware
    app.add_middleware(SizeLimitMiddleware, max_bytes=settings.upload_max_bytes)

    from app.middleware.request_id import RequestIdMiddleware
    app.add_middleware(RequestIdMiddleware)

    register_exception_handlers(app)
    app.include_router(api_v1_router)

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "env": settings.app_env,
            "api_prefix": "/api/v1",
            "docs": "/docs",
        }

    return app


app = create_app()