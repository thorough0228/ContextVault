"""Wiki routing suite (Phase 16) — informational, NOT gated.

Measures wiki-first routing behaviour over real golden queries:

* ``wiki_hit_rate``      — retrieval golden queries against the source
  document's wiki summary page (queries were generated FROM the record
  text, so the distilled page should lexically match)
* ``wiki_fallback_rate`` — same queries, plus out_of_kb.jsonl questions
  (domain-external, should miss the wiki layer and fall back)
* ``wiki_route_overhead_ms`` — wall time of the lexical route scorer
  itself (p50/p95); the thing a wiki hit saves is a full search

Wiki pages are the REAL LLM summaries (``WIKI_LLM_MODEL`` inherited
from .env, exactly what production routes on). Ingestion uses hash
embeddings because the routing decision never touches vectors — this
suite measures the distilled layer, not retrieval quality. Like
test_retrieval_metrics, "offline" here still makes bounded provider
calls (one summary-LLM call per ingested corpus).

No baseline.json key on purpose: compare_to_baseline treats a missing
suite as informational-only (notes, no failures). Promote deliberately
via ``update_baseline("wiki", scores)`` once the metric semantics
settle; fallback-rate/overhead keys are pre-registered in
LOWER_IS_BETTER_KEYS so a future gate can't invert direction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

DATASETS = Path(__file__).resolve().parents[2] / "datasets"
CORPORA = ["marvel"]              # smallest corpus → fastest ingest
GOLDEN_SAMPLE = 40                # deterministic head slice per corpus
ROUTE_THRESHOLD = 0.3             # keep in sync with WIKI_ROUTE_MIN_SCORE


def _load_jsonl(path: Path, limit: int | None = None) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _pct(values: list, p: float) -> float:
    s = sorted(values)
    return float(s[min(len(s) - 1, int(len(s) * p))])


async def _score_queries(db, rag_id: str, queries: list) -> tuple:
    """Run the route scorer per query; return (hits, overheads_ms)."""

    from sqlalchemy import select

    from app.models import WikiPage
    from app.services.wiki_service import wiki_context_for_query

    pages = (
        await db.scalars(
            select(WikiPage).where(
                WikiPage.rag_id == rag_id,
                WikiPage.stale.is_(False),
            )
        )
    ).all()
    assert pages, (
        "no wiki pages in the eval rag — the summary-LLM call likely "
        "failed during ingestion; check wiki.summary.llm_failed in the log"
    )

    hits = 0
    overheads: list = []
    for q in queries:
        text = q.get("query") or q.get("question") or ""
        t0 = time.perf_counter()
        matched = await wiki_context_for_query(
            db, rag_id=rag_id, query=text
        )
        overheads.append((time.perf_counter() - t0) * 1000.0)
        if matched and float(matched[0]["score"]) >= ROUTE_THRESHOLD:
            hits += 1
    return hits, overheads


async def test_wiki_routing_metrics():
    from evals.harness.environment import SqliteEvalEnv
    from evals.harness.report import write_run_report

    async with SqliteEvalEnv(
        corpora=CORPORA, embedding="hash", llm="hash",
    ) as env:
        from app.db import get_sessionmaker

        scores: dict = {}
        all_overheads: list = []
        golden_hits = golden_total = 0

        for name in CORPORA:
            queries = _load_jsonl(
                DATASETS / name / "retrieval.jsonl", GOLDEN_SAMPLE
            )
            async with get_sessionmaker()() as db:
                hits, overheads = await _score_queries(
                    db, env.rag_ids[name], queries
                )
            all_overheads.extend(overheads)
            golden_hits += hits
            golden_total += len(queries)
            scores[name] = {
                "wiki_hit_rate": hits / len(queries),
                "wiki_fallback_rate": 1 - hits / len(queries),
            }

        # domain-external questions must fall back to chunk retrieval
        ook = _load_jsonl(DATASETS / "out_of_kb.jsonl")
        async with get_sessionmaker()() as db:
            ook_hits, ook_overheads = await _score_queries(
                db, env.rag_ids[CORPORA[0]], ook
            )
        all_overheads.extend(ook_overheads)
        scores["out_of_kb"] = {
            "queries": float(len(ook)),
            "wiki_fallback_rate": 1 - ook_hits / max(len(ook), 1),
        }

        denom = max(golden_total, 1)
        scores["overall"] = {
            "wiki_hit_rate": golden_hits / denom,
            "wiki_fallback_rate": 1 - golden_hits / denom,
        }
        scores["overhead"] = {
            "wiki_route_overhead_ms_p50": _pct(all_overheads, 0.5),
            "wiki_route_overhead_ms_p95": _pct(all_overheads, 0.95),
        }

    report_path = write_run_report("wiki", scores, run_meta=env.meta.as_dict())
    print(f"\nreport: {report_path}")

    # Structural sanity, not a quality gate — baseline promotion is a
    # deliberate later step (docs: evals/README, update_baseline).
    assert golden_total > 0
    # disaster detector only: the page is a Qwen distillation whose
    # exact coverage varies per run, and Phase 18 stopword-filtered
    # scoring moved the absolute rate — 0.15 catches "pages missing /
    # scorer totally broken" without pinning a content-dependent number
    assert scores["overall"]["wiki_hit_rate"] >= 0.15, (
        "golden queries should mostly hit their source doc's wiki page"
    )
    assert scores["out_of_kb"]["wiki_fallback_rate"] >= 0.8, (
        "out-of-kb questions should mostly fall back to chunk retrieval"
    )
