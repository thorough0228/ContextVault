"""RAG schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

RagStatusLiteral = Literal["ACTIVE", "PROCESSING", "ERROR"]

RagName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
RagDescription = Annotated[str, StringConstraints(strip_whitespace=True, max_length=4096)]


class RagCreateRequest(BaseModel):
    """POST /api/v1/rags payload."""

    name: RagName
    description: RagDescription = ""


class RagPatchRequest(BaseModel):
    """PATCH /api/v1/rags/{rag_id} payload — every field optional."""

    name: RagName | None = None
    description: RagDescription | None = None
    status: RagStatusLiteral | None = None


class RagResponse(BaseModel):
    """Single RAG as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    name: str
    description: str
    status: RagStatusLiteral
    created_at: datetime
    updated_at: datetime


class RagListResponse(BaseModel):
    items: list[RagResponse]
    total: int = Field(description="Total items returned in this page (==len(items))")