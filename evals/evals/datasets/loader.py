"""Golden dataset loading + schema validation.

Dataset files (all NDJSON, one object per line):

* ``datasets/{corpus}/retrieval.jsonl`` — retrieval golden labels::

      {"id": "...", "query": "...", "positive_ids": ["chunk_42"], "notes": ""}

* ``datasets/{corpus}/qa.jsonl`` — generation golden QA::

      {"id": "...", "question": "...", "ground_truth": "...",
       "source_ids": ["chunk_42"], "category": "factual"}

* ``datasets/out_of_kb.jsonl`` — out-of-KB refusal probes::

      {"id": "...", "question": "...", "why_out_of_kb": "..."}

The loader is strict: malformed rows raise :class:`DatasetError` with
the file + line number instead of silently skewing the metrics.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from evals.config import DATASETS_DIR


class DatasetError(ValueError):
    """Malformed golden dataset row or file."""


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    positive_ids: frozenset
    notes: str = ""


@dataclass(frozen=True)
class QACase:
    case_id: str
    question: str
    ground_truth: str
    source_ids: frozenset
    category: str = "factual"


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        raise DatasetError(f"dataset file missing: {path}")
    rows: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    if not rows:
        raise DatasetError(f"{path}: no rows")
    return rows


def _require(row: dict, key: str, path: Path, lineno: int):
    value = row.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise DatasetError(f"{path}:{lineno}: missing or empty {key!r}")
    return value


def _ids(row: dict, key: str, path: Path, lineno: int, *, required: bool) -> frozenset:
    raw = row.get(key)
    if raw is None:
        if required:
            raise DatasetError(f"{path}:{lineno}: missing {key!r}")
        return frozenset()
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise DatasetError(f"{path}:{lineno}: {key!r} must be a list of strings")
    ids = frozenset(x for x in raw if x.strip())
    if required and not ids:
        raise DatasetError(f"{path}:{lineno}: {key!r} is empty")
    return ids


def load_retrieval_cases(
    corpus: str, datasets_dir: Path = DATASETS_DIR
) -> List[RetrievalCase]:
    path = datasets_dir / corpus / "retrieval.jsonl"
    cases: List[RetrievalCase] = []
    seen: Set[str] = set()
    for lineno, row in enumerate(_read_jsonl(path), start=1):
        case_id = _require(row, "id", path, lineno)
        if case_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate id {case_id!r}")
        seen.add(case_id)
        cases.append(
            RetrievalCase(
                case_id=case_id,
                query=_require(row, "query", path, lineno),
                positive_ids=_ids(row, "positive_ids", path, lineno, required=True),
                notes=str(row.get("notes", "")),
            )
        )
    return cases


def load_qa_cases(corpus: str, datasets_dir: Path = DATASETS_DIR) -> List[QACase]:
    path = datasets_dir / corpus / "qa.jsonl"
    cases: List[QACase] = []
    seen: Set[str] = set()
    for lineno, row in enumerate(_read_jsonl(path), start=1):
        case_id = _require(row, "id", path, lineno)
        if case_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate id {case_id!r}")
        seen.add(case_id)
        cases.append(
            QACase(
                case_id=case_id,
                question=_require(row, "question", path, lineno),
                ground_truth=_require(row, "ground_truth", path, lineno),
                source_ids=_ids(row, "source_ids", path, lineno, required=True),
                category=str(row.get("category", "factual")),
            )
        )
    return cases


def load_out_of_kb(datasets_dir: Path = DATASETS_DIR) -> List[dict]:
    path = datasets_dir / "out_of_kb.jsonl"
    rows: List[dict] = []
    for lineno, row in enumerate(_read_jsonl(path), start=1):
        rows.append(
            {
                "id": _require(row, "id", path, lineno),
                "question": _require(row, "question", path, lineno),
                "why_out_of_kb": str(row.get("why_out_of_kb", "")),
            }
        )
    return rows


def audit_against_corpus(
    cases: Sequence, corpus_index
) -> List[str]:
    """Return human-readable warnings for golden ids the corpus lacks.

    Sampling the corpus without its golden rows makes metrics lie —
    always run this audit when composing a run and fail loudly on any
    result (the caller decides whether warnings are fatal).
    """

    known: Set[str] = corpus_index.record_ids
    warnings: List[str] = []
    for case in cases:
        positives = set(case.positive_ids if hasattr(case, "positive_ids") else case.source_ids)
        missing = positives - known
        if missing:
            warnings.append(
                f"{case.case_id}: {len(missing)} golden ids not in corpus "
                f"(e.g. {sorted(missing)[:3]})"
            )
    return warnings


def sample_cases(cases: List, n: int, seed: int) -> List:
    """Seeded sampling that preserves file order (stable, diff-able)."""

    if n >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    picked = set(rng.sample(range(len(cases)), n))
    return [c for i, c in enumerate(cases) if i in picked]
