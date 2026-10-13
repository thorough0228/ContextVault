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


# Agent 轻量解析 API 每片上限（mineru.net 免 token 档）: 20 页 / 10MB。
# 字节上限留 5% 余量——分片重编码后体积可能不降反升（图重页面）。
_MINERU_MAX_PAGES_PER_PART = 20
_MINERU_MAX_BYTES_PER_PART = 10 * 1024 * 1024
_MINERU_BYTES_SAFETY = 0.95


def _split_pdf_pages(data: bytes, max_pages: int) -> List[bytes]:
    """Split a PDF into parts that each satisfy the Agent API limits:
    ≤max_pages pages AND ≤10 MB of bytes (5% safety margin).

    Pages are packed greedily; when a part re-encodes over the byte
    budget the part shrinks (binary search on page count) — image-heavy
    pages can make fewer-page parts *larger*, so page count alone is
    not enough. Single-page parts that still exceed the byte budget
    are returned anyway: the API rejects them with -30001 and the
    auto-chain degrades (nothing more we can do client-side).
    PDFs already within both limits come back as the original bytes.
    Corrupted/encrypted PDFs raise — the auto-chain degrades.
    """

    import fitz

    byte_budget = int(_MINERU_MAX_BYTES_PER_PART * _MINERU_BYTES_SAFETY)
    src = fitz.open(stream=data, filetype="pdf")
    try:
        if src.page_count <= max_pages and len(data) <= byte_budget:
            return [data]

        parts: List[bytes] = []
        start = 0
        while start < src.page_count:
            # page-count ceiling for this part
            hi = min(start + max_pages, src.page_count)
            part = fitz.open()
            part.insert_pdf(src, from_page=start, to_page=hi - 1)
            blob = part.tobytes()
            # shrink on the byte budget: binary search the largest
            # page count whose re-encode fits
            while len(blob) > byte_budget and hi - start > 1:
                lo, cur = start, hi
                while lo + 1 < cur:
                    mid = (lo + cur) // 2
                    probe = fitz.open()
                    probe.insert_pdf(src, from_page=start, to_page=mid - 1)
                    if len(probe.tobytes()) <= byte_budget:
                        lo = mid
                    else:
                        cur = mid
                    probe.close()
                part.close()
                hi = lo
                part = fitz.open()
                part.insert_pdf(src, from_page=start, to_page=hi - 1)
                blob = part.tobytes()
            part.close()
            parts.append(blob)
            start = hi
        logger.info(
            "parser.pdf.split pages=%d parts=%d", src.page_count, len(parts)
        )
        return parts
    finally:
        src.close()


def _parse_pdf_mineru(
    stream: io.BufferedIOBase, *, timeout_s: float = 600.0
) -> List[ParsedPage]:
    """Engine 1: the mineru.net Agent lightweight API (no token).

    PDFs over the Agent limit (20 pages) are pre-split into ≤20-page
    parts; each part is its own API task and the markdown parts are
    joined before the shared cleaning + pagination pipeline (virtual
    pages renumber continuously across parts). Any part failure fails
    the whole engine — the auto-chain degrades to the next engine.
    """

    from app.config import get_settings
    from app.ingest import mineru_api as mineru_api_mod
    from app.ingest.mineru_convert import clean_mineru_markdown

    settings = get_settings()
    data = stream.read()
    try:
        parts = _split_pdf_pages(data, _MINERU_MAX_PAGES_PER_PART)
        md_parts = [
            # module-attribute access (not a from-import binding) so
            # tests can monkeypatch the wire function per-part
            mineru_api_mod.fetch_markdown_via_api(
                part,
                base_url=settings.mineru_api_base_url,
                timeout_s=timeout_s,
                poll_interval=settings.mineru_api_poll_interval_seconds,
                filename=f"part-{i}.pdf" if len(parts) > 1 else "input.pdf",
            )
            for i, part in enumerate(parts)
        ]
        clean = clean_mineru_markdown("\n\n".join(md_parts))
    except mineru_api_mod.MineruApiError as exc:
        raise RuntimeError(f"mineru api: {exc}") from exc

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