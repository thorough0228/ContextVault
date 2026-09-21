"""Unit tests for the deterministic retrieval metrics.

Every expected value below is hand-computed from the definition in
``evals.metrics.retrieval`` — these tests ARE the metric spec.

Reference example used throughout:

    ranked   = [a, b, c, d]
    relevant = {b, d}

    hit_rate@1 = 0        hit_rate@3 = 1
    recall@1   = 0        recall@2   = 0.5      recall@4 = 1.0
    mrr@4      = 0.5      (b at rank 2)
    ndcg@2     = DCG(0.6309)/IDCG(1.6309) = 0.3869
    ndcg@4     = DCG(1.0616)/IDCG(1.6309) = 0.6509
"""

from __future__ import annotations

import pytest

from evals.metrics.retrieval import (
    QueryJudgement,
    dedupe_preserving_order,
    evaluate_queries,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

pytestmark = pytest.mark.offline

RANKED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d"}


class TestHitRate:
    def test_hit_at_3_and_beyond(self):
        assert hit_rate_at_k(RANKED, RELEVANT, 3) == 1.0
        assert hit_rate_at_k(RANKED, RELEVANT, 4) == 1.0

    def test_miss_below_first_relevant(self):
        assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_empty_relevant_scores_zero(self):
        assert hit_rate_at_k(RANKED, set(), 4) == 0.0

    def test_k_past_ranked_length_is_safe(self):
        assert hit_rate_at_k(["x"], {"x"}, 99) == 1.0


class TestRecall:
    def test_partial_recall(self):
        assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)

    def test_full_recall(self):
        assert recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(1.0)

    def test_no_relevant_in_top1(self):
        assert recall_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_empty_relevant_scores_zero(self):
        assert recall_at_k(RANKED, set(), 4) == 0.0


class TestMRR:
    def test_first_relevant_at_rank_2(self):
        assert mrr_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)

    def test_no_relevant_in_window(self):
        assert mrr_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_rank1_gives_full_score(self):
        assert mrr_at_k(["b", "a"], RELEVANT, 2) == pytest.approx(1.0)


class TestNDCG:
    def test_ndcg_at_2(self):
        assert ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.3869, abs=1e-4)

    def test_ndcg_at_4(self):
        assert ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.6509, abs=1e-4)

    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k(["b", "d", "a", "c"], RELEVANT, 4) == pytest.approx(1.0)

    def test_more_relevant_than_k_scales_idcg(self):
        # 4 relevant, k=2: DCG = 1/log2(2) = 1.0 (b at rank 1); IDCG
        # covers only the top-2 ideal ranks = 1 + 0.6309.
        assert ndcg_at_k(["b", "x"], {"a", "b", "c", "d"}, 2) == pytest.approx(
            1.0 / 1.6309, abs=1e-4
        )

    def test_empty_relevant_scores_zero(self):
        assert ndcg_at_k(RANKED, set(), 4) == 0.0


class TestDedupe:
    def test_keeps_first_occurrence_order(self):
        assert dedupe_preserving_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert dedupe_preserving_order([]) == []


class TestEvaluateQueries:
    def test_macro_average_over_queries(self):
        judgements = [
            QueryJudgement("q1", "query one", ["b", "a", "c", "d"], {"b", "d"}),
            # Perfect ranking for q2 at every k.
            QueryJudgement("q2", "query two", ["d", "b", "x", "y"], {"b", "d"}),
        ]
        scores = evaluate_queries(judgements, k_grid=[1, 3])

        assert scores["hit_rate@3"] == pytest.approx(1.0)
        # q1 mrr@3 = 1/1 (b at rank 1)... q1 ranked b first -> 1.0; q2 -> 1.0
        assert scores["mrr@3"] == pytest.approx(1.0)
        # recall@3: q1 -> {b} in top3 = 0.5 ; q2 -> {d,b} = 1.0
        assert scores["recall@3"] == pytest.approx(0.75)

    def test_duplicate_ids_dont_double_count(self):
        # Two DB chunks mapping to the same source record must count once.
        j = QueryJudgement("q", "q", ["b", "b", "a"], {"b"})
        scores = evaluate_queries([j], k_grid=[2])
        assert scores["recall@2"] == pytest.approx(1.0)
        assert scores["ndcg@2"] == pytest.approx(1.0)

    def test_no_judgements_returns_zeros(self):
        scores = evaluate_queries([], k_grid=[1, 5])
        assert scores["recall@1"] == 0.0
        assert scores["ndcg@5"] == 0.0
