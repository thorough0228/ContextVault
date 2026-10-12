"""Wiki boundary router suite (Phase 18) — informational, NOT gated.

Scores the four-way router against hand-labelled queries over the
marvel corpus (real LLM-summarised wiki page, hash embeddings — the
router never touches vectors):

* ``route_accuracy``          — exact match between the router and the
  ``expected`` label;
* ``wiki_only_false_rate``    — MUST-RAG questions that slipped into
  WIKI_ONLY (the dangerous error; hard-floored at 0);
* ``necessary_fallback_recall`` — must-RAG questions correctly kept out
  of WIKI_ONLY;
* ``unnecessary_fallback_rate`` — wiki-sufficient questions that fell
  back anyway (the cost metric; reported, not gated).

ABSTAIN is exercised by forcing ``notes.contradictions`` onto the page
(the same state the daily contradiction scan produces).

No baseline key on purpose — promote deliberately once the label set
matures. WIKI_ROUTE_ENABLED is pinned ON for the env so the router is
live; routing itself is pure DB + regex, no LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

DATASETS = Path(__file__).resolve().parents[2] / "datasets"
CORPUS = "marvel"


def _load_queries() -> list:
    rows = []
    with (DATASETS / "wiki_router" / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def test_wiki_router_metrics():
    import os

    from evals.harness.environment import SqliteEvalEnv
    from evals.harness.report import write_run_report

    queries = _load_queries()
    assert queries, "router dataset missing"

    # the router must be live inside the env; the key is managed, so
    # the guard restores it afterwards
    os.environ["WIKI_ROUTE_ENABLED"] = "true"
    try:
        async with SqliteEvalEnv(
            corpora=[CORPUS], embedding="hash", llm="hash",
        ) as env:
            from app.db import get_sessionmaker
            from app.services import wiki_router
            from sqlalchemy import select

            from app.models import WikiPage

            rag_id = env.rag_ids[CORPUS]

            async with get_sessionmaker()() as db:
                page = (
                    await db.scalars(
                        select(WikiPage).where(
                            WikiPage.rag_id == rag_id,
                            WikiPage.kind == "doc_summary",
                        )
                    )
                ).first()
                assert page is not None, (
                    "no wiki page — summary LLM call likely failed during ingest"
                )

                routed: list = []
                details: list = []
                conflict = {
                    "contradictions": [
                        {"slug_a": page.slug, "slug_b": "external",
                         "summary": "synthetic conflict for abstain eval"},
                    ]
                }
                for q in queries:
                    # abstain rows run against the CONFLICTED page (the
                    # state the daily scan produces); every other row
                    # runs against the clean page
                    wanted = conflict if q["expected"] == "abstain" else None
                    if page.notes != wanted:
                        page.notes = wanted
                        await db.commit()
                    decision = await wiki_router.route_for_query(
                        db, rag_id=rag_id, query=q["query"]
                    )
                    routed.append(decision.route)
                    details.append({
                        "id": q["id"], "expected": q["expected"],
                        "routed": decision.route, "reason": decision.reason,
                    })

        expected = [q["expected"] for q in queries]

        # per-label sub-metrics
        must = [d for d, e in zip(details, expected) if e == "must_rag"]
        suff = [d for d, e in zip(details, expected) if e == "wiki_only"]
        oob = [d for d, e in zip(details, expected) if e == "rag_only"]
        abst = [d for d, e in zip(details, expected) if e == "abstain"]

        false_wiki_only = sum(1 for d in must if d["routed"] == "wiki_only")
        necessary_recall = (
            sum(1 for d in must if d["routed"] != "wiki_only") / max(len(must), 1)
        )
        unnecessary_fallback = (
            sum(1 for d in suff if d["routed"] != "wiki_only") / max(len(suff), 1)
        )
        accuracy = sum(
            1 for d in details if d["routed"] == d["expected"]
        ) / len(details)

        scores = {
            "overall": {
                "route_accuracy": accuracy,
                "wiki_only_false_rate": false_wiki_only / max(len(must), 1),
                "necessary_fallback_recall": necessary_recall,
                "unnecessary_fallback_rate": unnecessary_fallback,
            },
            "must_rag": {"queries": float(len(must))},
            "wiki_only": {"queries": float(len(suff))},
            "out_of_kb": {
                "queries": float(len(oob)),
                # the safety property that matters: oob must never be
                # answered from the wiki alone (rag_only OR plus_rag ok)
                "not_wiki_only_rate": (
                    sum(1 for d in oob if d["routed"] != "wiki_only")
                    / max(len(oob), 1)
                ),
            },
            "abstain": {
                "queries": float(len(abst)),
                "abstain_rate": (
                    sum(1 for d in abst if d["routed"] == "abstain")
                    / max(len(abst), 1)
                ),
            },
        }
    finally:
        os.environ.pop("WIKI_ROUTE_ENABLED", None)

    report_path = write_run_report(
        "wiki_router", scores, run_meta=env.meta.as_dict(),
        per_case=details,
    )
    print(f"\nreport: {report_path}")
    for d in details:
        flag = "OK " if d["routed"] == d["expected"] else "MISS"
        print(f"  {flag} {d['id']}: expected={d['expected']} "
              f"routed={d['routed']} ({d['reason']})")

    # hard floors — the dangerous failure mode can never be tolerated
    assert false_wiki_only == 0, (
        f"{false_wiki_only} must-RAG questions were routed WIKI_ONLY: "
        + str([d for d in must if d["routed"] == "wiki_only"])
    )
    assert scores["out_of_kb"]["not_wiki_only_rate"] >= 0.9, (
        "out-of-kb questions must never be answered from the wiki alone"
    )
    assert scores["abstain"]["abstain_rate"] >= 0.9, (
        "conflicted pages must produce a deterministic abstain"
    )
