"""Service layer.

Business logic lives here, not in the routers. Routers handle HTTP
wiring + schema validation; services own the rules.
"""

from app.services import (
    auth_service,
    chat_service,
    document_chunk_service,
    document_service,
    rag_service,
    search_service,
)

__all__ = [
    "auth_service",
    "rag_service",
    "document_service",
    "document_chunk_service",
    "search_service",
    "chat_service",
]