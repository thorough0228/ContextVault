"""Deterministic retrieval metrics — pure functions, zero dependencies.

These are exact set/rank computations over golden relevance labels.
They must never be LLM-judged: when ground-truth chunk ids exist,
math is strictly better than a judge (free, reproducible, offline).

Conventions
-----------
* ``ranked`` is the retrieved id list in rank order. It is deduplicated
  (preserving first occurrence) before scoring — a re-chunked corpus can
  legitimately map two DB chunks onto the same source record, and the
  second occurrence adds no information.
* ``relevant`` is the golden positive id set. Empty ``relevant`` scores
  0.0 everywhere (the loader rejects such rows; this is a safety net).
* ``k`` may exceed ``len(ranked)`` — slices simply shorten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set


@dataclass(frozen=True)
class QueryJudgement:
    """One scored query: what was retrieved vs. what should have been."""

    query_id: str
    query: str
    ranked_ids: Sequence[str]
    relevant_ids: Set[str] = field(default_factory=set)


def dedupe_preserving_order(ids: Iterable[str]) -> List[str]:
    """Drop repeat ids, keeping the first (highest-ranked) occurrence."""
    seen: Set[str] = set()
    out: List[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def hit_rate_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """1.0 iff any of the top-k retrieved ids is relevant."""
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if any(i in relevant for i in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of golden positives found in the top k."""
    if not relevant or k <= 0:
        return 0.0
    found = sum(1 for i in ranked[:k] if i in relevant)
    return found / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """1 / rank of the first relevant hit within the top k, else 0."""
    if not relevant or k <= 0:
        return 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Binary-relevance NDCG@k.

    DCG sums ``1 / log2(rank + 1)`` over relevant hits in the top k;
    IDCG is the same sum over the best possible placement of
    ``min(len(relevant), k)`` positives. Ties cannot occur with a
    deterministic ranker, so no tie-handling is needed.
    """
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


_METRICS = {
    "hit_rate": hit_rate_at_k,
    "recall": recall_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def evaluate_queries(
    judgements: Sequence[QueryJudgement], k_grid: Iterable[int]
) -> Dict[str, float]:
    """Macro-average every metric over every k — the suite's scoreboard.

    Returns ``{"hit_rate@1": 0.92, "recall@5": 0.74, ...}``. Queries are
    deduped once here so each metric sees the same cleaned ranking.
    """
    grid = sorted({k for k in k_grid if k > 0})
    scores: Dict[str, float] = {}
    if not judgements:
        return {f"{name}@{k}": 0.0 for name in _METRICS for k in grid}

    cleaned = [
        (dedupe_preserving_order(j.ranked_ids), set(j.relevant_ids))
        for j in judgements
    ]
    for name, fn in _METRICS.items():
        for k in grid:
            values = [fn(ranked, relevant, k) for ranked, relevant in cleaned]
            scores[f"{name}@{k}"] = sum(values) / len(values)
    return scores
