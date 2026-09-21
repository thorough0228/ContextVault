"""Suite ① — retrieval quality (deterministic, offline, regression-gated).

For every corpus: run each golden query through the REAL search API
(this exercises the bge query-side instruction asymmetry exactly as
production does), map hits back to golden record ids via the
deterministic CorpusIndex, then score Recall@k / HitRate@k / MRR@k /
NDCG@k and gate against ``baseline.json``.

``test_retrieval_metrics`` needs the LLM-synthesized golden datasets
(runs after ``python -m evals.datasets.synthesize``); it skips with a
clear message until they exist.
``test_retrieval_pipeline_mechanics`` is a self-sufficient harness
regression: doc-as-query golden labels over a small marvel corpus —
validates env → ingest → search → alignment → metrics → report →
baseline gate with zero network and no synthesized data.
"""

from __future__ import annotations

import pytest

from evals.config import (
    DATASETS_DIR,
    LOCAL_SMALL_MODEL_PATH,
    LOCAL_SMALL_QUERY_INSTRUCTION,
    get_eval_settings,
)
from evals.datasets.loader import (
    audit_against_corpus,
    load_retrieval_cases,
)
from evals.harness.alignment import CorpusIndex
from evals.harness.environment import (
    LiveEvalEnv,
    SqliteEvalEnv,
    run_search,
)
from evals.harness.report import compare_to_baseline, write_run_report
from evals.metrics.retrieval import QueryJudgement, evaluate_queries

pytestmark = [pytest.mark.offline, pytest.mark.live]

CORPORA = ["table_tennis", "indian_law", "marvel"]


def _make_env(corpora: list[str]):
    settings = get_eval_settings()
    if settings.is_live:
        return LiveEvalEnv(
            api_url=settings.api_url,
            email=settings.live_email,
            password=settings.live_password,
            corpora=corpora,
            corpus_size=settings.live_corpus_size,
            seed=settings.seed,
            ingest_timeout_s=settings.live_ingest_timeout_s,
            keep_rags=settings.live_keep_rags,
        )
    return SqliteEvalEnv(
        corpora=corpora,
        embedding="production",
        llm="hash",
        corpus_size=settings.corpus_size,
        seed=settings.seed,
        chunk_guard=settings.sqlite_chunk_guard,
        search_mode=settings.search_mode,
    )


def _rank_records_for_hits(index: CorpusIndex, hits: list[dict]) -> tuple[list[str], int]:
    """Flatten hits into a ranked record-id list; count unmapped hits."""

    ranked: list[str] = []
    unmapped = 0
    for hit in hits:
        record_ids = index.map_chunk(
            hit.get("page_number", 1), hit.get("metadata"), hit.get("chunk_text")
        )
        if not record_ids:
            unmapped += 1
            continue
        # Records under one hit share its rank; page order keeps it stable.
        ranked.extend(sorted(record_ids))
    return ranked, unmapped


async def _run_metrics(env, corpora: list[str], cases_by_corpus: dict) -> tuple[dict, list[str]]:
    """Score every corpus + the pooled overall; return (scores, problems)."""

    settings = get_eval_settings()
    scores: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    all_judgements: list[QueryJudgement] = []
    per_case: list[dict] = []

    for corpus in corpora:
        cases = cases_by_corpus[corpus]
        index = CorpusIndex(env.corpora[corpus].records)

        # Golden labels must reconcile with the sampled corpus — a
        # mismatch means sampling drifted, and every metric after it
        # would be noise.
        warnings = audit_against_corpus(cases, index)
        if warnings:
            problems.extend(f"[{corpus}] {w}" for w in warnings)
            continue

        judgements: list[QueryJudgement] = []
        unmapped_total = 0
        for case in cases:
            hits = await run_search(
                env.client,
                env.auth_headers,
                rag_id=env.rag_ids[corpus],
                query=case.query,
                top_k=settings.retrieval_top_k,
            )
            ranked, unmapped = _rank_records_for_hits(index, hits)
            unmapped_total += unmapped
            judgements.append(
                QueryJudgement(case.case_id, case.query, ranked, set(case.positive_ids))
            )
            per_case.append(
                {
                    "corpus": corpus,
                    "case_id": case.case_id,
                    "query": case.query,
                    "ranked": ranked[:10],
                    "positives": sorted(case.positive_ids),
                }
            )
        all_judgements.extend(judgements)

        corpus_scores = evaluate_queries(judgements, settings.k_grid)
        scores[corpus] = corpus_scores
        hit_total = sum(len(j.ranked_ids) for j in judgements) or 1
        print(
            f"\n[{corpus}] {len(judgements)} queries "
            f"hit_rate@5={corpus_scores.get('hit_rate@5', 0):.3f} "
            f"recall@5={corpus_scores.get('recall@5', 0):.3f} "
            f"mrr@10={corpus_scores.get('mrr@10', 0):.3f} "
            f"ndcg@10={corpus_scores.get('ndcg@10', 0):.3f} "
            f"(unmapped hits: {unmapped_total}/{hit_total})"
        )
        if unmapped_total / hit_total > 0.10:
            problems.append(
                f"[{corpus}] {unmapped_total}/{hit_total} hits failed record "
                "alignment (>10%) — chunker/parser drift suspected"
            )

    if all_judgements:
        scores["overall"] = evaluate_queries(all_judgements, settings.k_grid)
    return scores, problems


async def test_retrieval_metrics():
    """The real retrieval scoreboard over synthesized golden datasets."""

    settings = get_eval_settings()
    missing = [
        c for c in CORPORA if not (DATASETS_DIR / c / "retrieval.jsonl").is_file()
    ]
    if missing:
        pytest.skip(
            f"golden retrieval datasets not synthesized yet for: {missing}; "
            "run `python -m evals.datasets.synthesize --corpus <name>` first"
        )

    cases_by_corpus = {c: load_retrieval_cases(c) for c in CORPORA}
    async with _make_env(CORPORA) as env:
        scores, problems = await _run_metrics(env, CORPORA, cases_by_corpus)
        assert problems == [], "dataset/corpus reconciliation problems:\n" + "\n".join(problems)
        report_path = write_run_report(
            "retrieval", scores, run_meta=env.meta.as_dict(),
        )
        print(f"\nreport: {report_path}")

    failures, notes = compare_to_baseline("retrieval", scores)
    for note in notes:
        print(f"  baseline: {note}")
    assert not failures, "retrieval regression:\n" + "\n".join(failures)


async def test_retrieval_pipeline_mechanics():
    """Harness self-regression: doc-as-query golden labels, small corpus.

    Marvel's records are templated CSV exports (every head reads
    "Title: … Year: … Universe: …"), so with production 1500/200
    windows a head-query is dominated by shared boilerplate and ranks
    poorly — that is a property of the corpus, not the harness. With
    250/0 windows each record head is an exact substring of one chunk,
    which makes the probe a deterministic plumbing check: env → ingest
    → chunk → align → search → metrics. Anything materially below the
    floor means the harness broke, not the model.
    """

    settings = get_eval_settings()
    async with SqliteEvalEnv(
        corpora=["marvel"], embedding="local", llm="hash",
        # Deterministic self-check on the small CPU model: zero network
        # (API endpoints are not deterministic enough for a pass/fail
        # plumbing gate) and no fp32 bge-m3 in-process (segfaults on
        # 4GB GPUs inside pytest when the worker holds CUDA memory).
        embedding_model=str(LOCAL_SMALL_MODEL_PATH),
        embedding_dimension=512,
        query_instruction=LOCAL_SMALL_QUERY_INSTRUCTION,
        corpus_size=30, seed=settings.seed, chunk_guard=420,
        chunk_size=250, chunk_overlap=0,
    ) as env:
        records = env.corpora["marvel"].records
        index = CorpusIndex(records)

        judgements: list[QueryJudgement] = []
        for record in records[:8]:
            query = record["text"][:60]
            hits = await run_search(
                env.client, env.auth_headers,
                rag_id=env.rag_ids["marvel"], query=query,
                top_k=settings.retrieval_top_k,
            )
            ranked, unmapped = _rank_records_for_hits(index, hits)
            assert unmapped == 0, "mechanics run must map every hit"
            judgements.append(
                QueryJudgement(record["id"], query, ranked, {record["id"]})
            )

        scores = {"mechanics": evaluate_queries(judgements, settings.k_grid)}
        report_path = write_run_report(
            "retrieval-mechanics", scores, run_meta=env.meta.as_dict()
        )
        print(f"\nreport: {report_path}")

    assert scores["mechanics"]["hit_rate@5"] >= 0.8, (
        f"doc-as-query hit_rate@5={scores['mechanics']['hit_rate@5']:.3f} — "
        "ingestion/alignment/search pipeline regressed"
    )
    assert scores["mechanics"]["recall@5"] >= 0.8
