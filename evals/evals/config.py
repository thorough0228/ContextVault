"""Eval harness settings.

Deliberately separate from the API's ``app.config.Settings``: these
knobs only steer the harness (corpus sizes, k grid, thresholds
tolerance, live-stack coordinates). Provider/model configuration still
flows through the API's own Settings so the eval measures exactly what
production would run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# evals/evals/config.py -> evals/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = REPO_ROOT / "evals" / "datasets"
RESULTS_DIR = REPO_ROOT / "evals" / "results"
BASELINE_PATH = REPO_ROOT / "evals" / "baseline.json"

# Small legacy model kept for deterministic pipeline self-checks
# (mechanics suite): CPU-fast, zero-network, never used for scoring.
LOCAL_SMALL_MODEL_PATH = REPO_ROOT / "models" / "bge-small-zh-v1.5"
LOCAL_SMALL_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


@dataclass(frozen=True)
class EvalSettings:
    """Harness knobs. Every field reads ``EVAL_*`` env vars so runs can
    be tuned without code edits (documented in evals/README.md)."""

    # sqlite (offline, default) | live (running docker stack + pgvector)
    profile: str = os.environ.get("EVAL_PROFILE", "sqlite")

    # Local embedding model used by the sqlite profile's ingestion and
    # search. Must match EMBEDDING_MODEL/EMBEDDING_DIMENSION in .env
    # when comparing against live runs. bge-m3 uses no query prefix;
    # bge-small-zh wants the Chinese retrieval instruction instead.
    embedding_model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "EVAL_EMBEDDING_MODEL",
                str(REPO_ROOT / "models" / "bge-m3"),
            )
        )
    )
    embedding_dimension: int = int(
        os.environ.get("EVAL_EMBEDDING_DIMENSION", "1024")
    )
    query_instruction: str = os.environ.get("EVAL_QUERY_INSTRUCTION", "")

    # Retrieval metric cut-offs. One search call per query at
    # ``retrieval_top_k`` serves the whole grid. Keep top_k high
    # enough that top_k * 50 (the SQLite candidate cap) exceeds the
    # largest corpus' chunk count — with the default 500/100 chunking
    # and corpus_size 300 that means top_k >= 20.
    k_grid: tuple[int, ...] = field(
        default_factory=lambda: _csv_ints(
            os.environ.get("EVAL_K_GRID", "1,3,5,10")
        )
    )
    retrieval_top_k: int = int(os.environ.get("EVAL_RETRIEVAL_TOP_K", "20"))

    # Chunking for eval ingestion — keep in sync with the API's
    # CHUNK_SIZE_CHARS / CHUNK_OVERLAP_CHARS.
    chunk_size: int = int(os.environ.get("EVAL_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.environ.get("EVAL_CHUNK_OVERLAP", "100"))

    # Source records sampled per corpus. The sqlite profile must keep
    # the re-chunked vector count under ``sqlite_chunk_guard`` because
    # the SQLite search path caps candidates at top_k * 50 (see
    # search_service._search_python) — beyond that, ranking is silently
    # truncated and the metrics would lie.
    # live_corpus_size MUST match corpus_size (300): golden labels are
    # synthesized against the 300-sample, and different sample sizes
    # produce disjoint subsets (same seed does NOT nest).
    # Search mode for production-embedding runs: "vector" or "hybrid".
    search_mode: str = os.environ.get("EVAL_SEARCH_MODE", "vector")

    corpus_size: int = int(os.environ.get("EVAL_CORPUS_SIZE", "300"))
    live_corpus_size: int = int(os.environ.get("EVAL_LIVE_CORPUS_SIZE", "300"))
    sqlite_chunk_guard: int = int(os.environ.get("EVAL_SQLITE_CHUNK_GUARD", "950"))

    # Generation suite sampling — bounds MiniMax token spend.
    qa_sample_per_corpus: int = int(os.environ.get("EVAL_QA_SAMPLE", "10"))

    # Regression gate: fail when a metric drops below
    # baseline - tolerance (absolute). Also the sanity floor applied
    # when no baseline entry exists yet.
    tolerance: float = float(os.environ.get("EVAL_TOLERANCE", "0.03"))

    # Deterministic corpus sampling / shuffling.
    seed: int = int(os.environ.get("EVAL_SEED", "20260918"))

    # Live profile coordinates (docker compose stack).
    api_url: str = os.environ.get("EVAL_API_URL", "http://localhost:8000")
    live_email: str = os.environ.get(
        "EVAL_LIVE_EMAIL", "contextvault.eval@example.com"
    )
    live_password: str = os.environ.get(
        "EVAL_LIVE_PASSWORD", "contextvault-eval-2026"
    )
    # Live worker ingestion is real (not eager) — poll READY this long.
    live_ingest_timeout_s: float = float(
        os.environ.get("EVAL_LIVE_INGEST_TIMEOUT_S", "300")
    )
    # Leave the eval RAGs behind for manual inspection instead of
    # deleting them at the end of a live run.
    live_keep_rags: bool = os.environ.get("EVAL_LIVE_KEEP_RAGS", "") not in (
        "", "0", "false",
    )

    @property
    def is_live(self) -> bool:
        return self.profile == "live"


def get_eval_settings() -> EvalSettings:
    return EvalSettings()
