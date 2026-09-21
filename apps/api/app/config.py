"""Application settings loaded from environment variables.

Precedence (highest first):
1. Shell environment
2. .env file (auto-discovered walking up to the repo root)
3. Defaults defined here

Use ``get_settings()`` to obtain a cached instance — keeps tests cheap.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _find_env_file(start: Path | None = None) -> str | None:
    """Nearest ``.env`` walking up from ``start`` (default: this file's dir).

    A bare ``env_file=".env"`` resolves against the process CWD — which is
    ``apps/api`` under uvicorn/alembic — and silently misses the repo-root
    ``.env``. Anchoring to ``__file__`` makes loading CWD-independent.
    """
    current = start or Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            return str(candidate)
    return None


# Placeholder defaults that *must* be replaced before running in any
# non-development / non-test environment. Used by the boot guard.
_DEFAULT_JWT_SECRET = "change-me-locally"
_DEFAULT_S3_SECRET_KEY = "change-me-locally"
_DEFAULT_POSTGRES_PASSWORD = "postgres"
# Origins we treat as acceptable in development. If `app_env` is
# anything else, these are still allowed because they're local — but
# a `*` is never allowed because the API serves `allow_credentials=True`.
_LOCAL_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- General ----
    app_name: str = "contextvault"
    app_env: str = "development"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_public_url: str = "http://localhost:8000"

    cors_allow_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # ---- Database ----
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "contextvault"
    postgres_user: str = "contextvault"
    postgres_password: str = _DEFAULT_POSTGRES_PASSWORD

    database_url: str | None = None

    # ---- Redis ----
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # ---- JWT (Phase 2) ----
    # Real values come from .env / environment. The default is only a
    # placeholder so tests can run without an env file.
    jwt_secret: str = _DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60

    # ---- Alembic ----
    # Synchronous URL used by `alembic upgrade head`. Alembic is sync.
    sync_database_url: str | None = None

    # ---- Phase 3: uploads + chunks + storage + Celery ----
    # Hard upload size cap, in bytes. Phase 3 first cut: 20 MiB.
    upload_max_bytes: int = 20 * 1024 * 1024

    # Allowed upload MIME types / extensions. Kept narrow on purpose:
    # only what the parsers actually understand.
    allowed_file_types: List[str] = Field(
        default_factory=lambda: ["pdf", "txt", "csv", "json", "jsonl"]
    )

    # Chunker knobs — read by the worker at task-time so changes here
    # take effect without redeploying the API.
    chunk_size_chars: int = 1500
    chunk_overlap_chars: int = 200

    # Object storage (MinIO / S3). The API uses these for upload
    # (`PUT object`) and the worker uses them for download.
    s3_endpoint_url: str = "http://localhost:9000"
    s3_public_endpoint_url: str = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_bucket: str = "contextvault"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = _DEFAULT_S3_SECRET_KEY

    # Celery — used by the API to enqueue ingestion jobs.
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_default_queue: str = "default"

    # When True, uploads go to a process-local in-memory dict instead of
    # MinIO/S3 — useful for tests + offline dev. The API never sets
    # this in production; tests do.
    storage_in_memory: bool = False

    # ---- Phase 4: Embedding + retrieval ----
    # All values are env-driven so production can swap providers
    # without code changes. Dimension is read at runtime; never
    # hard-code it inside the embedding/search code.
    embedding_provider: str = "hash"
    embedding_model: str = "hash-256"
    embedding_dimension: int = 256
    embedding_batch_size: int = 32
    embedding_timeout_seconds: float = 30.0
    embedding_max_chars_per_input: int = 8000
    # OpenAI-compatible endpoint (Phase 4 ships a stub provider).
    embedding_openai_api_key: str = ""
    embedding_openai_base_url: str = "https://api.openai.com/v1"
    # Some OpenAI-compatible endpoints reject the ``dimensions`` body
    # parameter (e.g. SiliconFlow's BAAI/bge-m3, which only outputs its
    # native size). Off = never send it; the model must then natively
    # output EMBEDDING_DIMENSION.
    embedding_openai_send_dimensions: bool = True
    # Minimum seconds between embedding requests to this endpoint.
    # Free-tier endpoints sometimes answer rate-limit excesses with a
    # 400 (code 20015) instead of a 429 — spacing the sequential batch
    # calls avoids that entirely.
    embedding_openai_request_interval: float = 0.0
    # Published provider quotas (requests / tokens per minute). When
    # either is > 0 the provider saturates the budgets with a sliding
    # window instead of a fixed sleep — e.g. SiliconFlow 2000 / 500000.
    embedding_openai_rpm_limit: int = 0
    embedding_openai_tpm_limit: int = 0
    # Default top_k when the client doesn't specify.
    search_default_top_k: int = 5
    search_max_top_k: int = 50

    # ---- Hybrid retrieval + rerank (Phase 12) ----
    # "vector" = pure semantic search (original behaviour);
    # "hybrid" = vector + keyword (tsvector/RRF) fusion, then optional
    # reranker precision pass.
    search_mode: str = "vector"
    # BM25 keyword leg (rank_bm25 over the persisted tokenized column,
    # per-RAG cached index). Falls back to tsvector recall when off or
    # when rank_bm25 is unavailable.
    search_bm25_enabled: bool = True
    search_hybrid_rrf_k: int = 60
    search_hybrid_candidate_multiplier: int = 2
    search_rerank_enabled: bool = False
    search_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    search_rerank_candidates: int = 20

    # ---- Document update (Phase 13) ----
    # replace: PUT rewrites the same document (delete+re-ingest).
    # bluegreen: PUT creates a new document and supersedes the old one
    # (old stays rollbackable, excluded from retrieval).
    document_update_strategy: str = "replace"

    # ---- Conversation memory (Phase 12) ----
    # Rolling summary of turns that fell out of the prompt window.
    summary_enabled: bool = True
    summary_max_chars: int = 800
    # Long-term (user, rag)-scoped facts extracted asynchronously after
    # each turn and injected into prompts. Extraction costs LLM tokens.
    memory_extraction_enabled: bool = False
    memory_injection_enabled: bool = True
    memory_top_k: int = 20
    memory_extract_max_facts: int = 5

    # ---- LLM (Phase 5) ----
    # All values come from .env / environment. The default provider
    # is the deterministic hash implementation, which works without
    # a network round trip — used for local dev and tests.
    llm_provider: str = "hash"
    llm_model: str = "hash-llm"
    llm_timeout_seconds: float = 30.0
    llm_max_context_tokens: int = 8000
    llm_openai_api_key: str = ""
    llm_openai_base_url: str = "https://api.openai.com/v1"
    # Reasoning models (MiniMax-M3, DeepSeek-R1, ...) stream <think>
    # blocks inline — strip them so the chat UI shows the answer only.
    llm_strip_think: bool = True

    # Bounded multi-turn memory — Phase 5 caps the prompt at this
    # many prior turns to prevent silent context blow-ups.
    chat_max_history_messages: int = 8

    # ---- MiniMax embeddings (EMBEDDING_PROVIDER=minimax) ----
    # MiniMax's /v1/embeddings is not OpenAI-compatible: it needs the
    # account GroupId as a query parameter and uses its own body shape.
    embedding_minimax_api_key: str = ""
    embedding_minimax_group_id: str = ""
    embedding_minimax_base_url: str = "https://api.minimax.chat"

    # ---- Local embeddings (EMBEDDING_PROVIDER=local) ----
    # In-process sentence-transformers model (no network, no key).
    # Requires: pip install sentence-transformers
    # BGE-family only: instruction prefix applied to queries, not to
    # indexed passages.
    embedding_local_query_instruction: str = ""
    # Half-precision weights on fp16-capable CUDA GPUs (Turing+).
    # Halves VRAM and ~doubles throughput for large local models such
    # as bge-m3; skipped automatically on CPU / older GPUs.
    embedding_local_fp16: bool = False

    @field_validator("cors_allow_origins", "allowed_file_types", mode="before")
    @classmethod
    def _split_csv(cls, value):
        # NoDecode skips pydantic-settings' strict-JSON env parsing, so
        # raw strings land here: accept both JSON arrays and the comma
        # form that .env files naturally use.
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    return json.loads(text)
                except ValueError:
                    pass
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator("cors_allow_origins", mode="after")
    @classmethod
    def _reject_wildcard_origin(cls, value):
        # `allow_credentials=True` (set in `app/main.py`) combined with
        # a wildcard origin is a credentials-with-`*` CORS policy. CORS
        # spec / browsers reject it, but server-to-server callers
        # (httpx, curl, internal services) would silently bypass the
        # Same-Origin Policy. Refuse at boot instead of debugging in
        # production.
        if any(origin == "*" for origin in value):
            raise ValueError(
                "CORS_ALLOW_ORIGINS may not contain '*' because "
                "allow_credentials=True; list each allowed origin explicitly"
            )
        return value

    @model_validator(mode="after")
    def _reject_default_secrets_outside_dev(self):
        """Boot guard: refuse to start in production-shaped envs when
        any of the secret defaults is still in place.

        Local dev + tests explicitly use these defaults — that's
        fine. ``staging`` / ``production`` must override them in the
        environment, otherwise the application refuses to start.
        """
        env = (self.app_env or "").lower()
        dev_environments = {"development", "dev", "local", "test", "testing"}
        if env in dev_environments:
            return self

        problems = []
        if self.jwt_secret == _DEFAULT_JWT_SECRET:
            problems.append("JWT_SECRET still equals the placeholder default")
        if self.postgres_password == _DEFAULT_POSTGRES_PASSWORD:
            problems.append("POSTGRES_PASSWORD still equals the placeholder default")
        if self.s3_secret_key == _DEFAULT_S3_SECRET_KEY:
            problems.append("S3_SECRET_KEY still equals the placeholder default")
        if problems:
            raise ValueError(
                "refusing to start with placeholder secrets: " + "; ".join(problems)
            )
        return self

    @property
    def async_database_url(self) -> str:
        """Build async SQLAlchemy URL from components if DATABASE_URL is absent."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def resolved_sync_database_url(self) -> str:
        """Synchronous URL for Alembic. Prefer explicit ``sync_database_url``,
        otherwise derive from components using ``psycopg2``."""
        if self.sync_database_url:
            return self.sync_database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() in {"development", "dev", "local"}

    @property
    def is_test(self) -> bool:
        return self.app_env.lower() in {"test", "testing"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — overridden in tests via ``Settings.__init__``."""
    return Settings()