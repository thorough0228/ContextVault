"""Wiki boundary router suite (Phase 18/19) — informational, NOT gated.

Scores the four-way router against hand-labelled queries. The QA-notes
page is seeded deterministically (Phase 19: the wiki is query-driven,
so ingest creates no pages) with live-valid provenance against the
ingested document; abstain rows force the same ``notes.contradictions``
state the daily scan produces. Fully offline — routing is pure DB +
regex, no LLM calls.

No baseline key on purpose — promote deliberately once the label set
matures. The hard floor: a must-RAG question can NEVER be routed
WIKI_ONLY (that is the dangerous failure mode).
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

            from app.models import Document, WikiPage

            rag_id = env.rag_ids[CORPUS]

            async with get_sessionmaker()() as db:
                # Phase 19: the wiki is query-driven — ingest creates no
                # pages. Seed the QA-notes page the route eval measures
                # against: content-rich enough for the fact queries to
                # lexically match, provenance live-valid against the
                # ingested document.
                doc = (
                    await db.scalars(
                        select(Document).where(Document.rag_id == rag_id)
                    )
                ).first()
                assert doc is not None
                # Phase 19: a QA-notes page literally accumulates the
                # questions that were asked and answered — simulate an
                # accumulated notebook from this dataset's own queries
                # (deterministic, honest to the production shape).
                entries = [
                    q["query"] for q in queries
                    # only answered, chunk-backed questions get absorbed —
                    # out-of-kb refusals never enter the notebook
                    if q["expected"] in ("wiki_only", "must_rag")
                ]
                content = "# Query Notes\n\n" + "\n\n".join(
                    f"### Q: {entry}\nKey facts and numbers from the source"
                    f" chunks are retained in this note."
                    for entry in entries
                )
                page = WikiPage(
                    id=f"qa-notes-eval-{rag_id[:8]}",
                    rag_id=rag_id,
                    kind="qa_notes",
                    slug="qa-notes",
                    title="Query Notes",
                    content=content,
                    source_refs=[
                        {
                            "document_id": doc.id,
                            "chunk_id": "eval-chunk-0",
                            "content_hash": doc.content_hash,
                            "filename": "marvel.jsonl",
                            "page_number": 1,
                            "score": 0.8,
                        }
                    ],
                    stale=False,
                )
                db.add(page)
                await db.commit()

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
