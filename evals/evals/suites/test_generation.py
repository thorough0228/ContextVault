"""Suite ② — generation quality (online: MiniMax chat + judge).

Per QA case the REAL chat pipeline runs end-to-end (bge retrieval →
prompt → MiniMax-M3 stream), then DeepEval judges the transcript:

* ``faithfulness``        — is the answer supported by the retrieved
                            context (anti-hallucination)?
* ``answer_relevancy``    — does the answer address the question?
* ``citation_accuracy``   — do the [n] citations point at passages that
                            genuinely support the claims (no invented
                            sources)?  (custom G-Eval)
* ``refusal_correctness`` — for out-of-KB questions, does the answer
                            honestly decline instead of fabricating?
                            (custom G-Eval over out_of_kb.jsonl)

Judge = MiniMax-M3 through the app's LLMProvider (self-preference bias
documented in README; swap the judge by editing .env). Token spend is
bounded by ``EVAL_QA_SAMPLE`` (default 10 per corpus) and counted.
"""

from __future__ import annotations

import asyncio

import pytest

from evals.config import DATASETS_DIR, get_eval_settings
from evals.datasets.loader import load_out_of_kb, load_qa_cases, sample_cases
from evals.harness.alignment import CorpusIndex
from evals.harness.environment import SqliteEvalEnv, chat_turn
from evals.harness.report import compare_to_baseline, write_run_report
from evals.metrics.judge import JUDGE_CALLS, build_judge_llm

pytestmark = [pytest.mark.online]

CORPORA = ["table_tennis", "indian_law", "marvel"]

CITATION_CRITERIA = (
    "You are evaluating a RAG assistant answer that may cite retrieved "
    "passages with [1], [2], ... markers. A score of 1 means: every claim "
    "that comes from the context is cited with an [n] marker, each cited "
    "number refers to a passage that genuinely supports the claim being "
    "made, and the answer does not invent filenames, page numbers, or "
    "sources not present in the context. A score of 0 means citations are "
    "absent where required, point at irrelevant passages, or are "
    "fabricated."
)

REFUSAL_CRITERIA = (
    "The question is NOT answerable from the retrieved context (it comes "
    "from outside the knowledge base). A score of 1 means the answer "
    "honestly communicates that the knowledge base does not contain the "
    "answer, and does not fabricate knowledge-base facts; it may add "
    "clearly-labelled general knowledge. A score of 0 means the answer "
    "pretends the knowledge base contains information it does not have, "
    "or fabricates cited facts."
)


async def _collect_turns(env, corpus: str, cases) -> list[dict]:
    turns = []
    for case in cases:
        turn = await chat_turn(
            env.client,
            env.auth_headers,
            rag_id=env.rag_ids[corpus],
            message=case.question,
            top_k=5,
        )
        assert turn["terminal"] == "done", (
            f"{case.case_id}: chat failed: {turn['error']}"
        )
        turns.append({"case": case, "turn": turn})
    return turns


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def _safe_measure(metric, test_case) -> float | None:
    """Measure, tolerating judge hiccups.

    A judge that returns unparseable JSON on one case must not abort
    the whole run — record None (excluded from the aggregate) and keep
    going; the error count lands in the report meta.
    """

    try:
        await asyncio.to_thread(metric.measure, test_case)
        return float(metric.score) if metric.score is not None else None
    except Exception as exc:  # noqa: BLE001
        JUDGE_CALLS["errors"] = JUDGE_CALLS.get("errors", 0) + 1
        print(f"    judge error: {type(exc).__name__}: {str(exc)[:120]}")
        return None


async def test_generation_quality():
    deepeval = pytest.importorskip("deepeval")
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    settings = get_eval_settings()
    if not (DATASETS_DIR / "table_tennis" / "qa.jsonl").is_file():
        pytest.skip("qa datasets not synthesized yet (run evals.datasets.synthesize)")

    JUDGE_CALLS["count"] = 0

    def citation_metric() -> GEval:
        return GEval(
            name="citation_accuracy",
            criteria=CITATION_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            model=judge,
            threshold=0.7,
            strict_mode=False,
            async_mode=False,
        )

    def refusal_metric() -> GEval:
        return GEval(
            name="refusal_correctness",
            criteria=REFUSAL_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            model=judge,
            threshold=0.7,
            strict_mode=False,
            async_mode=False,
        )

    async with SqliteEvalEnv(
        corpora=CORPORA,
        embedding="production",
        llm="real",  # MiniMax via .env — the system under test
        corpus_size=settings.corpus_size,
        seed=settings.seed,
        chunk_guard=settings.sqlite_chunk_guard,
        ingest_timeout_s=180.0,
    ) as env:
        # Build judge adapters INSIDE the env: the factories then resolve
        # to this env's providers (repo-local bge; MiniMax via .env) —
        # outside an env, .env's stale mind_harbor model path would fail.
        # deepeval 4.x: AnswerRelevancy is LLM-judged only (no embedding
        # model parameter any more).
        judge = build_judge_llm()

        scores: dict[str, dict[str, float]] = {}
        per_case: list[dict] = []

        for corpus in CORPORA:
            all_cases = load_qa_cases(corpus)
            cases = sample_cases(all_cases, settings.qa_sample_per_corpus, settings.seed)
            turns = await _collect_turns(env, corpus, cases)
            index = CorpusIndex(env.corpora[corpus].records)

            faith_values, relev_values, cite_values = [], [], []
            eligible_turns = 0
            for item in turns:
                case, turn = item["case"], item["turn"]
                contexts = [c.get("chunk_text", "") for c in turn["citations"]]
                # Citation accuracy is CONDITIONAL on retrieval actually
                # fetching the golden source: with hit_rate@5 ≈ 0.3 most
                # turns legitimately have nothing to cite, and scoring
                # those as citation failures conflates retrieval quality
                # with citation behaviour. context_hit_rate reports the
                # upstream split separately.
                retrieved_ids: set[str] = set()
                for c in turn["citations"]:
                    retrieved_ids |= index.map_chunk(
                        c.get("page_number", 1), c.get("metadata"),
                        c.get("chunk_text"),
                    )
                eligible = bool(set(case.source_ids) & retrieved_ids)

                test_case = LLMTestCase(
                    input=case.question,
                    actual_output=turn["answer"],
                    expected_output=case.ground_truth,
                    retrieval_context=contexts,
                )
                faith = FaithfulnessMetric(
                    threshold=0.7, model=judge, include_reason=False,
                    async_mode=False,
                )
                relev = AnswerRelevancyMetric(
                    threshold=0.7, model=judge, async_mode=False,
                )
                # DeepEval's sync measure() may manage its own loop —
                # keep it off the suite's loop.
                faith_score = await _safe_measure(faith, test_case)
                relev_score = await _safe_measure(relev, test_case)
                cite_score = None
                if eligible:
                    eligible_turns += 1
                    cite = citation_metric()
                    cite_score = await _safe_measure(cite, test_case)

                row = {
                    "corpus": corpus,
                    "case_id": case.case_id,
                    "question": case.question,
                    "cite_eligible": eligible,
                    "faithfulness": faith_score,
                    "answer_relevancy": relev_score,
                    "citation_accuracy": cite_score,
                    "answer_head": turn["answer"][:160],
                }
                per_case.append(row)
                print(
                    f"  [{corpus}] {case.case_id}: "
                    f"faith={faith_score} relev={relev_score} "
                    f"cite={cite_score if cite_score is not None else 'n/a'}"
                )
                if faith_score is not None:
                    faith_values.append(faith_score)
                if relev_score is not None:
                    relev_values.append(relev_score)
                if cite_score is not None:
                    cite_values.append(cite_score)

            corpus_scores = {
                "faithfulness": round(_mean(faith_values), 4),
                "answer_relevancy": round(_mean(relev_values), 4),
                "context_hit_rate": round(eligible_turns / len(turns), 4) if turns else 0.0,
                "cite_eligible_turns": eligible_turns,
            }
            # Undefined (not zero) when no turn retrieved the golden
            # source — omit so the floor gate doesn't punish a corpus
            # for upstream retrieval misses.
            if cite_values:
                corpus_scores["citation_accuracy"] = round(_mean(cite_values), 4)
            scores[corpus] = corpus_scores

        # Out-of-KB refusal probes — asked against table_tennis.
        oob_rows = load_out_of_kb()
        oob_sample = sample_cases(oob_rows, 8, settings.seed)
        refusal_values = []
        for oob in oob_sample:
            turn = await chat_turn(
                env.client, env.auth_headers,
                rag_id=env.rag_ids["table_tennis"],
                message=oob["question"], top_k=5,
            )
            assert turn["terminal"] == "done", f"{oob['id']}: {turn['error']}"
            test_case = LLMTestCase(
                input=oob["question"],
                actual_output=turn["answer"],
                retrieval_context=[c.get("chunk_text", "") for c in turn["citations"]],
            )
            metric = refusal_metric()
            value = await _safe_measure(metric, test_case)
            if value is None:
                continue
            refusal_values.append(value)
            per_case.append(
                {"corpus": "out_of_kb", "case_id": oob["id"],
                 "question": oob["question"], "refusal_correctness": round(value, 4),
                 "answer_head": turn["answer"][:160]}
            )
            print(f"  [oob] {oob['id']}: refusal={value:.2f}")

        cite_scores = [scores[c]["citation_accuracy"] for c in CORPORA if "citation_accuracy" in scores[c]]
        scores["overall"] = {
            "faithfulness": round(
                _mean([scores[c]["faithfulness"] for c in CORPORA]), 4
            ),
            "answer_relevancy": round(
                _mean([scores[c]["answer_relevancy"] for c in CORPORA]), 4
            ),
            "citation_accuracy": round(_mean(cite_scores), 4) if cite_scores else 0.0,
            "context_hit_rate": round(
                _mean([scores[c]["context_hit_rate"] for c in CORPORA]), 4
            ),
            "refusal_correctness": round(_mean(refusal_values), 4),
        }
        scores["out_of_kb"] = {"refusal_correctness": round(_mean(refusal_values), 4)}

        meta = dict(env.meta.as_dict())
        meta["llm"]["judge_calls"] = JUDGE_CALLS.get("count", 0)
        meta["llm"]["judge_empty_retries"] = JUDGE_CALLS.get("empty", 0)
        meta["llm"]["judge_errors"] = JUDGE_CALLS.get("errors", 0)
        meta["qa_sample_per_corpus"] = settings.qa_sample_per_corpus
        report_path = write_run_report("generation", scores, run_meta=meta, per_case=per_case)
        print(f"\nreport: {report_path}")

    failures, notes = compare_to_baseline("generation", scores)
    for note in notes:
        print(f"  baseline: {note}")
    assert not failures, "generation regression:\n" + "\n".join(failures)
