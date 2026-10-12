"""PDF + TXT parsers.

Both yield :class:`ParsedPage` records. TXT is one page (1). PDF
extracts one page per PyMuPDF ``Page`` and preserves the 1-based page
number.
"""

from __future__ import annotations

import csv as csv_module
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Structured-file records (CSV rows / JSON objects) are grouped into
# virtual pages of this many records before chunking.
_RECORDS_PER_PAGE = 200
# When a record has one of these fields (case-insensitive), its value
# is the meaningful text — e.g. marvel_rag.csv's pre-built `document`
# column or chunks.json's `text` chunks.
_TEXT_FIELD_CANDIDATES = ("text", "document", "content")


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1")
        if not isinstance(self.text, str):
            raise TypeError("text must be a str")
        # Postgres TEXT rejects NUL bytes (0x00) and PDF extraction leaks
        # them from embedded-font ligatures/glyphs — strip at this single
        # funnel every parser passes through, before chunking/embedding.
        if "\x00" in self.text:
            object.__setattr__(self, "text", self.text.replace("\x00", ""))


def parse_pdf(stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Extract text from a PDF stream, honouring ``PDF_PARSER``.

    * ``auto`` (default) — try the mineru CLI (highest-quality layout/
      table/OCR extraction), then unstructured, then PyMuPDF.
    * ``mineru`` / ``unstructured`` / ``pymupdf`` — pin one engine
      (missing dependencies degrade to the next in the auto chain).
    """

    settings = None
    try:
        from app.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 — standalone usage without config
        pass
    order = _pdf_engine_order(
        settings.pdf_parser if settings else "auto",
    )
    errors: List[str] = []
    for engine in order:
        try:
            if engine == "mineru":
                pages = _parse_pdf_mineru(
                    stream,
                    timeout_s=(
                        settings.pdf_parser_timeout_seconds
                        if settings
                        else 600.0
                    ),
                )
            elif engine == "unstructured":
                pages = _parse_pdf_unstructured(stream)
            else:
                pages = _parse_pdf_pymupdf(stream)
            logger.info("parser.pdf.ok engine=%s pages=%d", engine, len(pages))
            return pages
        except Exception as exc:  # noqa: BLE001 — degrade to next engine
            errors.append(f"{engine}: {str(exc)[:120]}")
            logger.warning(
                "parser.pdf.engine_failed engine=%s err=%s", engine, str(exc)[:120]
            )
    raise ValueError("all PDF parsers failed — " + " | ".join(errors))


def _pdf_engine_order(configured: str) -> List[str]:
    if configured == "auto":
        return ["mineru", "unstructured", "pymupdf"]
    return [configured]


def _parse_pdf_pymupdf(stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Original engine: PyMuPDF text-layer extraction. Pages without
    extractable text yield an empty string — normal for scans."""

    import fitz  # local so tests that skip parsers don't pay the cost

    doc = fitz.open(stream=stream)
    try:
        pages: List[ParsedPage] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(ParsedPage(page_number=i, text=text))
        return pages
    finally:
        doc.close()


def _parse_pdf_mineru(
    stream: io.BufferedIOBase, *, timeout_s: float = 600.0
) -> List[ParsedPage]:
    """Engine 1: the mineru CLI as an isolated subprocess.

    The layout model never loads inside this process (it would bloat
    the worker and fight for GPU memory). Writes the stream to a temp
    .pdf, runs ``mineru -p ... -o ...``, then converts the produced
    markdown (tables arrive as inline HTML — cleaned by
    ``mineru_to_markdown.clean_mineru_markdown``) into virtual pages
    split at markdown H1/H2 headings.
    """

    import subprocess
    import tempfile

    from app.ingest.mineru_convert import clean_mineru_markdown

    data = stream.read()

    with tempfile.TemporaryDirectory(prefix="cv-mineru-") as tmp:
        tmp_dir = Path(tmp)
        pdf_path = tmp_dir / "input.pdf"
        pdf_path.write_bytes(data)
        out_dir = tmp_dir / "out"

        import os as _os

        source = _os.environ.get("MINERU_SOURCE", "modelscope")
        import sys

        exe = Path(sys.executable).parent / "Scripts" / "mineru.exe"
        if not exe.exists():  # posix venv/bin layout
            exe = Path(sys.executable).parent / "mineru"
        out_md = tmp_dir / "out.md"
        result = subprocess.run(
            [str(exe), "parse", str(pdf_path),
             "--tier", "flash", "--format", "markdown",
             "-o", str(out_md), "--wait", "300"],
            capture_output=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mineru exited {result.returncode}: "
                f"{result.stderr.decode('utf-8', 'ignore')[:200]}"
            )
        if not out_md.exists():
            raise RuntimeError("mineru produced no markdown output")
        clean = clean_mineru_markdown(out_md.read_text(encoding="utf-8"))

    return _markdown_to_pages(clean)


def _parse_pdf_unstructured(stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Engine 2: unstructured.partition_pdf (optional dependency)."""

    from unstructured.partition.pdf import partition_pdf

    stream.seek(0)
    elements = partition_pdf(file=stream)
    by_page: dict[int, List[str]] = {}
    for el in elements:
        page = int(getattr(el.metadata, "page_number", 1) or 1)
        text = str(el).strip()
        if text:
            by_page.setdefault(page, []).append(text)
    pages = [
        ParsedPage(page_number=p, text="\n\n".join(parts))
        for p, parts in sorted(by_page.items())
    ]
    if not pages:
        raise RuntimeError("unstructured produced no text")
    return pages


def _markdown_to_pages(md_text: str) -> List[ParsedPage]:
    """Split cleaned markdown into virtual pages at H1/H2 headings
    (each chapter becomes one page; no headings → single page)."""

    lines = md_text.splitlines()
    pages: List[ParsedPage] = []
    current: List[str] = []
    start = 1

    def flush():
        text = "\n".join(current).strip()
        if text:
            pages.append(ParsedPage(page_number=len(pages) + 1, text=text))

    for line in lines:
        if line.startswith("# ") or line.startswith("## "):
            flush()
            current = [line]
            start = 1
        else:
            current.append(line)
    flush()

    if not pages:
        pages.append(ParsedPage(page_number=1, text=md_text.strip()))
    return pages


def parse_markdown(stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Parse a markdown document (e.g. MinerU output already cleaned by
    ``scripts/mineru_to_markdown.py``). Splits at H1/H2 headings."""

    raw = stream.read()
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else str(raw)
    return _markdown_to_pages(text)


def parse_txt(stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Decode a text stream as UTF-8. Returns one virtual page."""
    raw = stream.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)
    if not text:
        # An empty .txt file is a valid input — produces zero chunks.
        return [ParsedPage(page_number=1, text="")]
    return [ParsedPage(page_number=1, text=text)]


def parse_document(*, file_type: str, stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Dispatch by file type. Returns a list of :class:`ParsedPage`."""
    if file_type == "pdf":
        return parse_pdf(stream)
    if file_type == "txt":
        return parse_txt(stream)
    if file_type == "csv":
        return parse_csv(stream)
    if file_type == "json":
        return parse_json(stream)
    if file_type == "md":
        return parse_markdown(stream)
    raise ValueError(f"unsupported file_type: {file_type!r}")


def iter_text_pages(pages: Iterable[ParsedPage]) -> Iterable[tuple[int, str]]:
    """Convenience iterator over (page_number, text) tuples."""
    for p in pages:
        yield p.page_number, p.text


def _pages_from_records(records: List[str]) -> List[ParsedPage]:
    """Group rendered records into virtual pages for the chunker."""
    pages: List[ParsedPage] = []
    for start in range(0, len(records), _RECORDS_PER_PAGE):
        batch = records[start : start + _RECORDS_PER_PAGE]
        pages.append(
            ParsedPage(page_number=len(pages) + 1, text="\n\n".join(batch))
        )
    if not pages:
        # An empty structured file is valid — behaves like an empty txt.
        pages.append(ParsedPage(page_number=1, text=""))
    return pages


def _pick_text_field(fieldnames: Iterable[str]) -> Optional[str]:
    """Return the first recognised text-bearing field name (original case)."""
    lowered = {name.lower(): name for name in fieldnames if name}
    for candidate in _TEXT_FIELD_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _record_to_text(record: dict, text_field: Optional[str]) -> str:
    if text_field is not None:
        value = record.get(text_field)
        if value is not None and str(value).strip():
            return str(value).strip()
    lines = [
        f"{key}: {value}"
        for key, value in sorted(record.items())
        if value is not None and str(value).strip()
    ]
    return "\n".join(lines)


def _records_to_dicts(data) -> List[dict]:
    """Normalise parsed JSON data to a list of plain records."""
    if not isinstance(data, list):
        data = [data]
    records: List[dict] = []
    for item in data:
        records.append(item if isinstance(item, dict) else {"value": item})
    return records


def parse_csv(stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Parse a UTF-8(-sig) CSV into per-row text records.

    If the header contains a ``text`` / ``document`` / ``content``
    column, each row renders just that value (RAG-ready exports such as
    marvel_rag.csv); otherwise every row renders as ``col: value`` lines.
    """
    raw = stream.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")

    reader = csv_module.DictReader(io.StringIO(raw))
    text_field = _pick_text_field(fieldnames := (reader.fieldnames or []))
    records: List[str] = []
    for row in reader:
        clean = {k: v for k, v in row.items() if k is not None}
        text = _record_to_text(clean, text_field)
        if text:
            records.append(text)
    logger.info("parser.csv.ok records=%d fields=%s", len(records), list(fieldnames))
    return _pages_from_records(records)


def parse_json(stream: io.BufferedIOBase) -> List[ParsedPage]:
    """Parse a UTF-8(-sig) JSON document (array, object, or NDJSON).

    Falls back to NDJSON (one JSON object per line) when the whole-file
    parse fails — several public datasets ship .json files that are
    actually line-delimited.
    """
    raw = stream.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        records_dicts: List[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records_dicts.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON/NDJSON document: {exc}") from exc
        if not records_dicts:
            raise ValueError("empty JSON document")
        data = records_dicts

    dicts = _records_to_dicts(data)
    text_field = _pick_text_field(
        {key for record in dicts[:20] for key in record}
    )
    records: List[str] = []
    for record in dicts:
        text = _record_to_text(record, text_field)
        if text:
            records.append(text)
    logger.info("parser.json.ok records=%d", len(records))
    return _pages_from_records(records)