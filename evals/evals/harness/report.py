"""Run reports, baselines, and the regression gate.

Every suite run writes a timestamped JSON artifact under ``results/``
plus a Markdown render, and is compared against ``baseline.json``:
a metric regressing more than ``EVAL_TOLERANCE`` below its baseline
fails the run (CI-able). When no baseline entry exists yet, a floor
from ``FLOORS`` applies — sanity only, deliberately generous until the
first real run is promoted to baseline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evals.config import BASELINE_PATH, RESULTS_DIR, get_eval_settings

# Absolute sanity floors for first-run (no baseline yet) asserts.
# These are NOT quality targets — they only catch catastrophic
# breakage (misalignment, broken ingestion, random embeddings).
FLOORS: Dict[str, Dict[str, float]] = {
    "retrieval": {
        "hit_rate@5": 0.15,
        "recall@5": 0.10,
        "mrr@10": 0.10,
        "ndcg@10": 0.08,
    },
    "generation": {
        "faithfulness": 0.50,
        "answer_relevancy": 0.40,
        "citation_accuracy": 0.30,
        "refusal_correctness": 0.50,
    },
    "robustness": {
        "citation_preserved_rate": 0.90,
    },
}


def write_run_report(
    suite: str,
    scores: Dict[str, Dict[str, float]],
    *,
    run_meta: dict,
    per_case: Optional[List[dict]] = None,
) -> Path:
    """Persist one run artifact (JSON) + a Markdown render; return the JSON path."""

    settings = get_eval_settings()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"eval-{stamp}-{suite}-{settings.profile}.json"

    payload = {
        "suite": suite,
        "profile": settings.profile,
        "created": datetime.now().isoformat(timespec="seconds"),
        "meta": run_meta,
        "scores": scores,  # {group: {metric: value}}
        "per_case": per_case or [],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_path = json_path.with_suffix(".md")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return json_path


def _render_markdown(payload: dict) -> str:
    lines = [
        f"# Eval report — {payload['suite']} ({payload['profile']})",
        "",
        f"* created: {payload['created']}",
        "",
    ]
    meta = payload.get("meta", {})
    if meta.get("embedding"):
        emb = meta["embedding"]
        lines.append(
            f"* embedding: `{emb.get('provider')}` / `{emb.get('model')}` "
            f"({emb.get('dimension')}d), chunk {emb.get('chunk_size_chars')}/"
            f"{emb.get('chunk_overlap_chars')}"
        )
    if meta.get("llm"):
        llm = meta["llm"]
        lines.append(f"* llm: `{llm.get('provider')}` / `{llm.get('model')}`")
    for corpus, cmeta in (meta.get("corpora") or {}).items():
        lines.append(
            f"* corpus **{corpus}**: {cmeta.get('records')} records "
            f"→ ~{cmeta.get('estimated_chunks')} chunks"
        )
    lines.append("")
    for group, metrics in payload.get("scores", {}).items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key in sorted(metrics):
            lines.append(f"| `{key}` | {metrics[key]:.4f} |")
        lines.append("")
    return "\n".join(lines)


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_baseline(
    suite: str,
    scores: Dict[str, Dict[str, float]],
    *,
    path: Path = BASELINE_PATH,
) -> None:
    """Promote a run's aggregate scores into the baseline (manual action)."""

    baseline = load_baseline(path)
    suites = baseline.setdefault("suites", {})
    suites[suite] = scores
    baseline["updated"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# Metric keys where a HIGHER value is worse (counts of failures/leaks).
# compare_to_baseline flips the regression direction for these.
LOWER_IS_BETTER_KEYS = {
    "canary_leak_queries",
    "system_prompt_leak_queries",
    "judge_errors",
}


def compare_to_baseline(
    suite: str,
    scores: Dict[str, Dict[str, float]],
    *,
    tolerance: Optional[float] = None,
    path: Path = BASELINE_PATH,
) -> Tuple[List[str], List[str]]:
    """Return (failures, notes) for a finished run.

    * baseline hit → fail when the metric moves below (above, for
      ``LOWER_IS_BETTER_KEYS``) the baseline by more than tolerance
    * no baseline  → fail if value < FLOORS floor
    """

    settings = get_eval_settings()
    tol = settings.tolerance if tolerance is None else tolerance
    baseline_suites = load_baseline(path).get("suites", {})
    baseline_scores = baseline_suites.get(suite) or {}
    floors = FLOORS.get(suite, {})

    failures: List[str] = []
    notes: List[str] = []
    for group, metrics in sorted(scores.items()):
        base_group = baseline_scores.get(group, {}) if baseline_scores else {}
        for key in sorted(metrics):
            value = metrics[key]
            label = f"{group}.{key}"
            if key in base_group:
                target = float(base_group[key])
                if key in LOWER_IS_BETTER_KEYS:
                    if value > target + tol:
                        failures.append(
                            f"{label}={value:.4f} > baseline {target:.4f} "
                            f"+ tol {tol:.2f}"
                        )
                    else:
                        notes.append(
                            f"{label}={value:.4f} ≤ baseline {target:.4f}"
                        )
                else:
                    if value < target - tol:
                        failures.append(
                            f"{label}={value:.4f} < baseline {target:.4f} "
                            f"- tol {tol:.2f}"
                        )
                    else:
                        notes.append(
                            f"{label}={value:.4f} ≥ baseline {target:.4f}"
                        )
            elif key in floors:
                if value < floors[key]:
                    failures.append(f"{label}={value:.4f} < floor {floors[key]:.2f}")
                else:
                    notes.append(f"{label}={value:.4f} ≥ floor {floors[key]:.2f} (no baseline)")
            else:
                notes.append(f"{label}={value:.4f} (no baseline/floor — informational)")
    return failures, notes
