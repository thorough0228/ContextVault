"""Eval corpus construction: dataset_rag sources -> ingested .jsonl files.

Each corpus is written as NDJSON with one ``{"id", "text"}`` object per
line. The app's JSON parser recognises the ``text`` field
(``_TEXT_FIELD_CANDIDATES``), so each record renders as exactly its
stripped text — which ``alignment.CorpusIndex`` mirrors to map retrieved
chunks back to record ids.

Sampling is seeded and order-preserving (original corpus order), so the
same seed always yields the same eval corpus.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from evals.config import REPO_ROOT

REPO_DATASETS = REPO_ROOT / "dataset_rag"


@dataclass(frozen=True)
class CorpusSpec:
    """One ready-to-upload corpus plus its alignment index inputs."""

    name: str
    records: List[dict]  # [{"id": str, "text": str}, ...] in file order
    jsonl_path: Path
    meta: dict  # provenance for the report


# ---------------------------------------------------------------------------
# Source loaders — each yields raw {"id", "text"} records
# ---------------------------------------------------------------------------


def _load_table_tennis() -> List[dict]:
    raw = json.loads(
        (REPO_DATASETS / "乒乓球知识库" / "chunks.json").read_text(encoding="utf-8")
    )
    out: List[dict] = []
    for item in raw:
        text = str(item.get("text") or "").strip()
        if text:
            out.append({"id": str(item["id"]), "text": text})
    return out


def _load_indian_law() -> List[dict]:
    out: List[dict] = []
    seen: set[str] = set()
    for path in sorted((REPO_DATASETS / "印度劳动法与公司法").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for section in data.get("sections", []):
            text = str(section.get("text") or "").strip()
            rid = f"{section.get('act', path.stem)}#{section.get('section_number', '')}"
            if text and rid not in seen:
                seen.add(rid)
                out.append({"id": rid, "text": text})
    return out


def _load_marvel() -> List[dict]:
    out: List[dict] = []
    with open(
        REPO_DATASETS / "漫威数据集" / "marvel_rag.csv", encoding="utf-8-sig", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            text = str(row.get("document") or "").strip()
            if text:
                out.append({"id": f"marvel#{row.get('id', len(out))}", "text": text})
    return out


def _load_injection() -> List[dict]:
    """Handcrafted prompt-injection corpus (see datasets/injection/)."""

    path = REPO_ROOT / "evals" / "datasets" / "injection" / "corpus.jsonl"
    out: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                out.append({"id": str(row["id"]), "text": str(row["text"])})
    return out


CORPUS_SOURCES: Dict[str, Callable[[], List[dict]]] = {
    "table_tennis": _load_table_tennis,
    "indian_law": _load_indian_law,
    "marvel": _load_marvel,
    "injection": _load_injection,
}


# ---------------------------------------------------------------------------
# Sampling + serialization
# ---------------------------------------------------------------------------


def sample_records(records: Sequence[dict], size: int, seed: int) -> List[dict]:
    """Seeded sample that keeps the original corpus order.

    Order matters: the ingested page layout (and therefore alignment)
    depends on it, and stable order keeps the corpus diff-able.
    """

    size = min(size, len(records))
    if size >= len(records):
        return list(records)
    rng = random.Random(seed)
    picked = set(rng.sample(range(len(records)), size))
    return [rec for i, rec in enumerate(records) if i in picked]


def estimate_chunk_count(records: Sequence[dict], *, size: int, overlap: int) -> int:
    """Run the production chunker over the corpus' virtual pages.

    Mirrors parser page grouping + FixedWindowChunker so the harness can
    enforce the SQLite candidate-cap guard BEFORE uploading anything.
    """

    from app.ingest.chunker import FixedWindowChunker
    from app.ingest.parser import ParsedPage

    texts = [str(r["text"]).strip() for r in records]
    texts = [t for t in texts if t]
    pages: List[ParsedPage] = []
    per_page = 200
    for start in range(0, len(texts), per_page):
        batch = texts[start : start + per_page]
        pages.append(ParsedPage(page_number=len(pages) + 1, text="\n\n".join(batch)))
    chunker = FixedWindowChunker(size=size, overlap=overlap)
    return len(chunker.chunk_pages(pages, document_id="__eval__", rag_id="__eval__"))


def build_corpus(
    name: str,
    *,
    size: int,
    seed: int,
    out_dir: Path,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> CorpusSpec:
    """Load, sample, serialize, and pre-flight one eval corpus."""

    if name not in CORPUS_SOURCES:
        raise ValueError(f"unknown corpus {name!r}; known: {sorted(CORPUS_SOURCES)}")
    records = sample_records(CORPUS_SOURCES[name](), size, seed)
    if not records:
        raise ValueError(f"corpus {name!r} produced zero records")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return CorpusSpec(
        name=name,
        records=records,
        jsonl_path=jsonl_path,
        meta={
            "source": name,
            "records": len(records),
            "seed": seed,
            "chunk_size_chars": chunk_size,
            "chunk_overlap_chars": chunk_overlap,
            "estimated_chunks": estimate_chunk_count(
                records, size=chunk_size, overlap=chunk_overlap
            ),
        },
    )
