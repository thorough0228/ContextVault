"""Eval environments: isolated app stacks the suites run against.

Two profiles:

* **sqlite** — in-process ASGI app on a throwaway SQLite file, in-memory
  storage, eager Celery worker. Fully offline when configured with the
  local bge embedding + hash LLM. The environment is an async context
  manager so every test/CLI entry point gets identical setup/teardown.
* **live** — plain HTTP client against the running docker stack
  (postgres + pgvector + redis + real worker). Registers a dedicated
  eval account and cleans up its own RAGs; never touches user data.

Why env vars instead of FastAPI ``dependency_overrides``: the service
layer (search, chat, the eager worker task) resolves settings via the
cached ``get_settings()`` — only environment-level configuration reaches
every consumer identically, which is exactly what an eval must
guarantee (router-level overrides would silently diverge).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from httpx import ASGITransport

from evals.config import EvalSettings, get_eval_settings
from evals.harness.corpus import CorpusSpec, build_corpus

# Env vars the sqlite profile owns while active; restored on exit.
_MANAGED_ENV_KEYS = (
    "APP_ENV", "APP_NAME", "DATABASE_URL", "STORAGE_IN_MEMORY",
    "JWT_SECRET", "JWT_ALGORITHM", "JWT_EXPIRES_MINUTES",
    "UPLOAD_MAX_BYTES", "CHUNK_SIZE_CHARS", "CHUNK_OVERLAP_CHARS",
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION",
    "EMBEDDING_BATCH_SIZE", "EMBEDDING_LOCAL_FP16",
    "EMBEDDING_LOCAL_QUERY_INSTRUCTION",
    "SEARCH_MODE", "SEARCH_RERANK_ENABLED",
    "MEMORY_EXTRACTION_ENABLED", "MEMORY_INJECTION_ENABLED",
    "LLM_PROVIDER", "LLM_MODEL",
    "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND",
)


@dataclass
class EvalRunMeta:
    """Snapshot of what the environment ran with — goes into reports."""

    profile: str
    corpora: Dict[str, dict] = field(default_factory=dict)
    embedding: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "profile": self.profile,
            "corpora": self.corpora,
            "embedding": self.embedding,
            "llm": self.llm,
        }


# ---------------------------------------------------------------------------
# Shared API helpers (used by both profiles)
# ---------------------------------------------------------------------------


async def register_or_login(
    client: httpx.AsyncClient, *, email: str, password: str
) -> dict:
    """Return Authorization headers for the eval account."""
    payload = {"email": email, "password": password}
    resp = await client.post("/api/v1/auth/register", json=payload)
    if resp.status_code not in (200, 201):
        resp = await client.post("/api/v1/auth/login", json=payload)
        resp.raise_for_status()
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def upload_corpus(
    client: httpx.AsyncClient,
    headers: dict,
    *,
    rag_id: str,
    spec: CorpusSpec,
) -> str:
    """Upload one corpus file, return the document id."""
    data = spec.jsonl_path.read_bytes()
    resp = await client.post(
        f"/api/v1/rags/{rag_id}/documents",
        headers=headers,
        files={"file": (spec.jsonl_path.name, data, "application/json")},
    )
    assert resp.status_code == 201, f"upload failed: {resp.status_code} {resp.text}"
    return resp.json()["id"]


async def wait_documents_ready(
    client: httpx.AsyncClient,
    headers: dict,
    *,
    rag_id: str,
    expected: int,
    timeout_s: float,
    interval_s: float = 1.0,
) -> Dict[str, str]:
    """Poll until every document is READY; return {document_id: status}."""

    deadline = time.monotonic() + timeout_s
    last: Dict[str, str] = {}
    while time.monotonic() < deadline:
        resp = await client.get(
            f"/api/v1/rags/{rag_id}/documents",
            headers=headers,
            params={"page": 1, "page_size": 100},
        )
        assert resp.status_code == 200, f"document list failed: {resp.text}"
        body = resp.json()
        last = {d["id"]: d["status"] for d in body["items"]}
        statuses = list(last.values())
        if len(statuses) >= expected and all(s == "READY" for s in statuses):
            return last
        if any(s == "FAILED" for s in statuses):
            failed = [
                (d["filename"], d.get("error_message"))
                for d in body["items"] if d["status"] == "FAILED"
            ]
            raise AssertionError(f"ingestion FAILED during eval setup: {failed}")
        await asyncio.sleep(interval_s)
    raise TimeoutError(
        f"documents not READY after {timeout_s}s (last statuses: {last})"
    )


async def run_search(
    client: httpx.AsyncClient,
    headers: dict,
    *,
    rag_id: str,
    query: str,
    top_k: int,
    retries: int = 3,
) -> List[dict]:
    """POST /search with transient-failure retries.

    Free-tier embedding endpoints occasionally time out mid-run; the
    app maps that to 503, so retry with a short backoff instead of
    aborting a long evaluation.
    """

    last_error = ""
    for attempt in range(retries + 1):
        resp = await client.post(
            f"/api/v1/rags/{rag_id}/search",
            headers=headers,
            json={"query": query, "top_k": top_k},
        )
        if resp.status_code == 200:
            return resp.json()["hits"]
        last_error = f"{resp.status_code} {resp.text[:200]}"
        if resp.status_code >= 500 and attempt < retries:
            await asyncio.sleep(2 * (attempt + 1))
            continue
        break
    raise AssertionError(f"search failed after retries: {last_error}")


async def chat_turn(
    client: httpx.AsyncClient,
    headers: dict,
    *,
    rag_id: str,
    message: str,
    top_k: int = 0,
    conversation_id: Optional[str] = None,
    timeout_s: float = 180.0,
) -> dict:
    """Run one streaming chat turn; collect + classify all NDJSON events.

    Returns ``{events, answer, citations, terminal, error}`` where
    ``terminal`` is ``"done"`` or ``"error"``.
    """

    payload = {"message": message, "top_k": top_k}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    events: List[dict] = []
    answer_parts: List[str] = []
    citations: List[dict] = []
    terminal: Optional[str] = None
    error: Optional[dict] = None

    async with client.stream(
        "POST", f"/api/v1/rags/{rag_id}/chat", headers=headers, json=payload,
    ) as resp:
        assert resp.status_code == 200, (
            f"chat HTTP {resp.status_code}: {await resp.aread()!r}"
        )
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            etype = event.get("type")
            events.append(event)
            if etype == "citation":
                citations = event.get("citations", [])
            elif etype == "token":
                answer_parts.append(event.get("delta", ""))
            elif etype == "done":
                terminal = "done"
                citations = event.get("citations", citations)
            elif etype == "error":
                terminal = "error"
                error = {"code": event.get("code"), "message": event.get("message")}

    return {
        "events": events,
        "answer": "".join(answer_parts),
        "citations": citations,
        "terminal": terminal,
        "error": error,
    }


# ---------------------------------------------------------------------------
# sqlite profile
# ---------------------------------------------------------------------------


class SqliteEvalEnv:
    """In-process app + eager worker on a throwaway SQLite database."""

    def __init__(
        self,
        *,
        corpora: List[str],
        embedding: str = "production",  # "production" (.env API) | "hash" (instant)
        llm: str = "hash",  # "hash" (offline) | "real" (MiniMax via .env)
        corpus_size: Optional[int] = None,
        seed: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunk_guard: Optional[int] = None,
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        query_instruction: Optional[str] = None,
        search_mode: str = "vector",
        memory_extraction: bool = False,
        memory_injection: bool = False,
        document_update_strategy: str = "replace",
        ingest_timeout_s: float = 120.0,
        tmp_root: Optional[Path] = None,
    ) -> None:
        _s = get_eval_settings()
        self._corpora_names = corpora
        self._embedding = embedding
        self._llm = llm
        self._search_mode = search_mode
        self._memory_extraction = memory_extraction
        self._memory_injection = memory_injection
        self._document_update_strategy = document_update_strategy
        self._corpus_size = corpus_size if corpus_size is not None else _s.corpus_size
        self._seed = seed if seed is not None else _s.seed
        self._chunk_size = chunk_size if chunk_size is not None else _s.chunk_size
        self._chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else _s.chunk_overlap
        )
        self._chunk_guard = chunk_guard if chunk_guard is not None else _s.sqlite_chunk_guard
        self._embedding_model = (
            embedding_model
            if embedding_model is not None
            else str(_s.embedding_model_path)
        )
        self._embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else _s.embedding_dimension
        )
        self._query_instruction = (
            query_instruction if query_instruction is not None else _s.query_instruction
        )
        self._ingest_timeout_s = ingest_timeout_s
        self._tmp_root = Path(tmp_root) if tmp_root else None

        self.client: Optional[httpx.AsyncClient] = None
        self.auth_headers: dict = {}
        self.rag_ids: Dict[str, str] = {}
        self.corpora: Dict[str, CorpusSpec] = {}
        self.meta = EvalRunMeta(profile="sqlite")

        self._tmpdir: Optional[str] = None
        self._saved_env: Dict[str, Optional[str]] = {}
        self._app = None

    async def __aenter__(self) -> "SqliteEvalEnv":
        self._tmpdir = tempfile.mkdtemp(prefix="cv-eval-")
        tmp = Path(self._tmpdir)
        self._saved_env = {k: os.environ.get(k) for k in _MANAGED_ENV_KEYS}

        db_path = tmp / "eval.sqlite"
        env: Dict[str, str] = {
            "APP_ENV": "test",
            "APP_NAME": "contextvault-eval",
            "DATABASE_URL": f"sqlite+aiosqlite:///{db_path.as_posix()}",
            "STORAGE_IN_MEMORY": "true",
            "JWT_SECRET": "eval-secret-do-not-use-in-prod",
            "JWT_ALGORITHM": "HS256",
            "JWT_EXPIRES_MINUTES": "120",
            "UPLOAD_MAX_BYTES": str(64 * 1024 * 1024),
            "CHUNK_SIZE_CHARS": str(self._chunk_size),
            "CHUNK_OVERLAP_CHARS": str(self._chunk_overlap),
            "CELERY_BROKER_URL": "memory://",
            "CELERY_RESULT_BACKEND": "cache+memory://",
        }
        if self._embedding == "production":
            # Inherit the API's own embedding provider from .env (the
            # SiliconFlow/OpenAI-compatible endpoint) — the eval then
            # measures exactly what production runs. No local model:
            # bge-m3 fp32 segfaults on 4GB GPUs inside the pytest
            # process when the worker holds CUDA memory.
            # Search/memory switches are PINNED (not inherited) so a
            # .env flip cannot silently move the baselines.
            env.update(
                SEARCH_MODE=self._search_mode,
                MEMORY_EXTRACTION_ENABLED=(
                    "1" if self._memory_extraction else "0"
                ),
                MEMORY_INJECTION_ENABLED=(
                    "1" if self._memory_injection else "0"
                ),
                DOCUMENT_UPDATE_STRATEGY=self._document_update_strategy,
            )
        elif self._embedding == "local":
            env.update(
                EMBEDDING_PROVIDER="local",
                EMBEDDING_MODEL=self._embedding_model,
                EMBEDDING_DIMENSION=str(self._embedding_dimension),
                EMBEDDING_LOCAL_QUERY_INSTRUCTION=self._query_instruction,
                EMBEDDING_LOCAL_FP16="1",
                EMBEDDING_BATCH_SIZE="8",
                SEARCH_MODE=self._search_mode,
                DOCUMENT_UPDATE_STRATEGY=self._document_update_strategy,
            )
        elif self._embedding == "hash":
            env.update(
                EMBEDDING_PROVIDER="hash",
                SEARCH_MODE="vector",
                MEMORY_EXTRACTION_ENABLED="0",
                MEMORY_INJECTION_ENABLED="0",
                DOCUMENT_UPDATE_STRATEGY="replace",
                EMBEDDING_MODEL="hash-eval",
                EMBEDDING_DIMENSION="32",
            )
        else:
            raise ValueError(f"unknown embedding mode {self._embedding!r}")
        if self._llm == "hash":
            env.update(LLM_PROVIDER="hash", LLM_MODEL="hash-llm")
        elif self._llm != "real":
            raise ValueError(f"unknown llm mode {self._llm!r}")
        os.environ.update(env)

        # Everything downstream must see the eval settings: clear every
        # cached singleton the app owns, in dependency order.
        from app.config import get_settings
        from app.db import Base, reset_engine_for_tests
        from app.embedding import reset_provider_for_tests
        from app.llm import reset_provider_for_tests as reset_llm
        from app.storage import reset_storage_for_tests

        get_settings.cache_clear()
        reset_engine_for_tests()
        reset_provider_for_tests()
        reset_llm()
        reset_storage_for_tests()

        from sqlalchemy.ext.asyncio import create_async_engine

        import app.models  # noqa: F401 — registers tables on Base.metadata

        engine = create_async_engine(env["DATABASE_URL"], echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

        from app.celery_client import celery_app
        from app.main import create_app

        settings = get_settings()
        self._app = create_app(settings)
        self._celery_state = _force_eager_celery(
            celery_app, database_url=env["DATABASE_URL"]
        )

        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self._app),
            base_url="http://evalserver",
        )

        self.meta.embedding = {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
        }
        self.meta.llm = {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }

        email = f"eval-{os.getpid()}-{int(time.time())}@example.com"
        self.auth_headers = await register_or_login(
            self.client, email=email, password="eval-password-2026"
        )
        await self._ingest_corpora(tmp)
        return self

    async def __aexit__(self, *exc) -> None:
        try:
            if self.client is not None:
                await self.client.aclose()
            state = getattr(self, "_celery_state", None)
            if state is not None:
                _restore_celery(state)
            from app.db import dispose_engine, reset_engine_for_tests
            from app.config import get_settings

            await dispose_engine()
            reset_engine_for_tests()
            get_settings.cache_clear()
        finally:
            for key, value in self._saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if self._tmpdir:
                shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def _ingest_corpora(self, tmp: Path) -> None:
        for name in self._corpora_names:
            spec = build_corpus(
                name,
                size=self._corpus_size,
                seed=self._seed,
                out_dir=tmp / "corpora",
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            if spec.meta["estimated_chunks"] > self._chunk_guard:
                raise AssertionError(
                    f"corpus {name!r} would produce ~{spec.meta['estimated_chunks']} "
                    f"vector chunks — above the sqlite guard {self._chunk_guard}. "
                    "The SQLite search path caps candidates at top_k*50, so lower "
                    "EVAL_CORPUS_SIZE or run the live profile."
                )
            resp = await self.client.post(
                "/api/v1/rags",
                headers=self.auth_headers,
                json={"name": f"eval-{name}", "description": "auto eval rag"},
            )
            assert resp.status_code == 201, resp.text
            rag_id = resp.json()["id"]
            await upload_corpus(
                self.client, self.auth_headers, rag_id=rag_id, spec=spec
            )
            await wait_documents_ready(
                self.client,
                self.auth_headers,
                rag_id=rag_id,
                expected=1,
                timeout_s=self._ingest_timeout_s,
            )
            self.rag_ids[name] = rag_id
            self.corpora[name] = spec
            self.meta.corpora[name] = dict(spec.meta)


class _CeleryState:
    """Saved celery config + send_task so eager mode restores cleanly."""

    def __init__(self) -> None:
        self.always_eager: Optional[bool] = None
        self.eager_propagates: Optional[bool] = None
        self.send_task = None
        self.database_url: Optional[str] = None


def _force_eager_celery(celery_app, database_url: str) -> _CeleryState:
    """Run tasks in-process; bypass the broker entirely (conftest pattern).

    ``conf.database_url`` MUST be set explicitly: celery_client.py pins
    it at module-import time, so in a multi-env process the eager task
    would otherwise keep opening the first env's (already deleted)
    SQLite file. Same seam the API's own conftest uses.
    """

    state = _CeleryState()
    state.always_eager = celery_app.conf.task_always_eager
    state.eager_propagates = celery_app.conf.task_eager_propagates
    state.send_task = celery_app.send_task
    state.database_url = getattr(celery_app.conf, "database_url", None)
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False
    celery_app.conf.database_url = database_url

    def _send_task(name, args=None, kwargs=None, **other):  # noqa: ARG001
        task = celery_app.tasks.get(name)
        if task is None:
            raise RuntimeError(f"task {name} not registered")
        return task.apply(args=args or (), kwargs=kwargs or {})

    celery_app.send_task = _send_task  # type: ignore[assignment]
    return state


def _restore_celery(state: _CeleryState) -> None:
    if state.always_eager is None:
        return
    from app.celery_client import celery_app

    celery_app.conf.task_always_eager = state.always_eager
    celery_app.conf.task_eager_propagates = state.eager_propagates
    celery_app.conf.database_url = state.database_url
    celery_app.send_task = state.send_task  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# live profile
# ---------------------------------------------------------------------------


class LiveEvalEnv:
    """HTTP-only environment against the running docker stack."""

    def __init__(
        self,
        *,
        api_url: str,
        email: str,
        password: str,
        corpora: List[str],
        corpus_size: int = 800,
        seed: Optional[int] = None,
        ingest_timeout_s: float = 300.0,
        keep_rags: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        _s = get_eval_settings()
        self._api_url = api_url
        self._email = email
        self._password = password
        self._corpora_names = corpora
        self._corpus_size = corpus_size
        self._seed = seed if seed is not None else _s.seed
        self._ingest_timeout_s = ingest_timeout_s
        self._keep_rags = keep_rags
        self._chunk_size = chunk_size if chunk_size is not None else _s.chunk_size
        self._chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else _s.chunk_overlap
        )

        self.client: Optional[httpx.AsyncClient] = None
        self.auth_headers: dict = {}
        self.rag_ids: Dict[str, str] = {}
        self.corpora: Dict[str, CorpusSpec] = {}
        self.meta = EvalRunMeta(profile="live")

    async def __aenter__(self) -> "LiveEvalEnv":
        self.client = httpx.AsyncClient(base_url=self._api_url, timeout=300.0)
        health = await self.client.get("/api/v1/health")
        if health.status_code != 200:
            raise ConnectionError(
                f"live stack not reachable at {self._api_url} "
                f"(HTTP {health.status_code}); start docker compose + API first"
            )
        self.auth_headers = await register_or_login(
            self.client, email=self._email, password=self._password
        )
        tmp = Path(tempfile.mkdtemp(prefix="cv-eval-live-"))
        try:
            await self._ingest_corpora(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return self

    async def __aexit__(self, *exc) -> None:
        try:
            if not self._keep_rags:
                for rag_id in self.rag_ids.values():
                    resp = await self.client.delete(
                        f"/api/v1/rags/{rag_id}", headers=self.auth_headers
                    )
                    if resp.status_code not in (200, 204):
                        # Cleanup is best-effort; a stale eval RAG is
                        # visible in the UI and deletable by hand.
                        pass
        finally:
            if self.client is not None:
                await self.client.aclose()

    async def _ingest_corpora(self, tmp: Path) -> None:
        for name in self._corpora_names:
            spec = build_corpus(
                name,
                size=self._corpus_size,
                seed=self._seed,
                out_dir=tmp / "corpora",
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            resp = await self.client.post(
                "/api/v1/rags",
                headers=self.auth_headers,
                json={
                    "name": f"eval-{name}-{int(time.time())}",
                    "description": "auto eval rag (safe to delete)",
                },
            )
            assert resp.status_code == 201, resp.text
            rag_id = resp.json()["id"]
            await upload_corpus(
                self.client, self.auth_headers, rag_id=rag_id, spec=spec
            )
            await wait_documents_ready(
                self.client,
                self.auth_headers,
                rag_id=rag_id,
                expected=1,
                timeout_s=self._ingest_timeout_s,
            )
            self.rag_ids[name] = rag_id
            self.corpora[name] = spec
            self.meta.corpora[name] = dict(spec.meta)
