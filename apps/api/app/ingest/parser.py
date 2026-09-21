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
    """Extract text from a PDF stream.

    Uses PyMuPDF (``fitz``). Pages without extractable text yield an
    empty string — that's normal for image-only scans.
    """
    import fitz  # local so tests that skip parsers don't pay the cost

    doc = fitz.open(stream=stream)
    try:
        pages: List[ParsedPage] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(ParsedPage(page_number=i, text=text))
        logger.info("parser.pdf.ok pages=%d", len(pages))
        return pages
    finally:
        doc.close()


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