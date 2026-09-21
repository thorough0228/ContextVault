"""Cross-app constants — mirror in packages/shared/ts/index.ts."""

from __future__ import annotations

APP_NAME = "contextvault"
API_VERSION = "0.4.0"
API_PREFIX = "/api/v1"
HEALTH_PATH = f"{API_PREFIX}/health"

SERVICE_API = "api"
SERVICE_WORKER = "worker"
SERVICE_WEB = "web"

# Auth routes
AUTH_PATH_REGISTER = f"{API_PREFIX}/auth/register"
AUTH_PATH_LOGIN = f"{API_PREFIX}/auth/login"
AUTH_PATH_LOGOUT = f"{API_PREFIX}/auth/logout"
AUTH_PATH_ME = f"{API_PREFIX}/auth/me"

# RAG routes
RAG_PATH_LIST = f"{API_PREFIX}/rags"
RAG_PATH_CREATE = f"{API_PREFIX}/rags"


def rag_path_detail(rag_id: str) -> str:
    return f"{API_PREFIX}/rags/{rag_id}"


# Document routes (Phase 3)
def rag_documents_path(rag_id: str) -> str:
    return f"{API_PREFIX}/rags/{rag_id}/documents"


def document_path_detail(document_id: str) -> str:
    return f"{API_PREFIX}/documents/{document_id}"


# Search route (Phase 4)
def rag_search_path(rag_id: str) -> str:
    return f"{API_PREFIX}/rags/{rag_id}/search"