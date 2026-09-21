"""Pydantic schemas for the API.

The schemas are split per resource (auth, user, rag, health,
document, search). ``HealthResponse`` migrated from the legacy
``app.schemas`` module (now a package).
"""

from app.schemas.auth import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    TokenResponse,
)
from app.schemas.document import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.schemas.health import HealthResponse
from app.schemas.rag import (
    RagCreateRequest,
    RagListResponse,
    RagPatchRequest,
    RagResponse,
)
from app.schemas.search import (
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.schemas.user import UserPublic

__all__ = [
    "HealthResponse",
    "LoginRequest",
    "LogoutResponse",
    "RegisterRequest",
    "TokenResponse",
    "RagCreateRequest",
    "RagListResponse",
    "RagPatchRequest",
    "RagResponse",
    "UserPublic",
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentDetailResponse",
    "ChunkResponse",
    "SearchRequest",
    "SearchHit",
    "SearchResponse",
]