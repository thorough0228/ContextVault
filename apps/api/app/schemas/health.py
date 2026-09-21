"""Pydantic schemas shared across API responses.

Phase 2 split this module into a package; ``HealthResponse`` now lives
in ``app.schemas.health`` to keep room for the per-resource modules
(``auth``, ``user``, ``rag``) without an import-name collision.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Overall service health")
    app: str = Field(description="Application name")
    version: str = Field(description="Application version")
    environment: str = Field(description="Deployment environment")
    checks: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-dependency probe results",
    )