"""Phase 20: >20-page PDF auto-splitting for the mineru Agent API.

The Agent lightweight API caps each file at 20 pages; oversized PDFs
are pre-split into ≤20-page parts (fitz), each part parsed as its own
task, and the markdown parts joined before the shared pagination
pipeline (virtual pages renumber continuously).
"""

from __future__ import annotations

import io
import json

import fitz
import httpx
import pytest

from app.ingest import parser as parser_mod

PART_LIMIT = 20


def _pdf(pages: int) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_split_21_pages_yields_2_parts():
    parts = parser_mod._split_pdf_pages(_pdf(21), PART_LIMIT)
    assert len(parts) == 2
    counts = [fitz.open(stream=p, filetype="pdf").page_count for p in parts]
    assert counts == [20, 1]
    for p in parts:
        assert len(p) <= 10 * 1024 * 1024


def test_split_41_pages_yields_3_parts():
    parts = parser_mod._split_pdf_pages(_pdf(41), PART_LIMIT)
    counts = [fitz.open(stream=p, filetype="pdf").page_count for p in parts]
    assert counts == [20, 20, 1]


def test_short_pdf_single_part_no_reencode():
    raw = _pdf(15)
    parts = parser_mod._split_pdf_pages(raw, PART_LIMIT)
    assert parts == [raw], "within-limit PDFs must pass through as-is"


def test_multi_part_parse_renumbers_pages_continuously(client_with_db, monkeypatch):
    """21-page PDF → 2 API tasks → markdown parts joined → virtual
    pages renumbered 1..N across the part boundary."""
    from app.ingest import mineru_api as mineru_api_mod

    def fetch_markdown_via_api(pdf_bytes, **kwargs):
        n = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
        first_page = 1 if n == 20 else 21  # part 1 = 20 pages, part 2 = 1
        return f"# Part heading {first_page}\n\nContent for page {first_page}.\n"

    monkeypatch.setattr(mineru_api_mod, "fetch_markdown_via_api", fetch_markdown_via_api)

    pages = parser_mod._parse_pdf_mineru(io.BytesIO(_pdf(21)))
    texts = [p.text for p in pages]
    assert any("Part heading 1" in t for t in pages[0].text.splitlines() or [""]), \
        "part 1's H1 must open the first virtual page"
    assert any("Part heading 21" in t for t in texts), \
        "part 2's H1 must survive the join"
    assert [p.page_number for p in pages] == list(range(1, len(pages) + 1)), \
        "virtual page numbers must be continuous across parts"


def test_second_part_api_error_degrades_whole_engine(client_with_db, monkeypatch):
    """Any part failing fails the ENGINE — auto-chain falls to pymupdf
    (no partial successes)."""
    from app.ingest import mineru_api as mineru_api_mod
    from app.ingest.mineru_api import MineruApiError

    calls = {"n": 0}

    def fetch_markdown_via_api(pdf_bytes, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MineruApiError("mineru parse failed: part 2 exploded")
        return "# Part 1\ncontent"

    monkeypatch.setattr(mineru_api_mod, "fetch_markdown_via_api", fetch_markdown_via_api)

    with pytest.raises(RuntimeError, match="part 2 exploded"):
        parser_mod._parse_pdf_mineru(io.BytesIO(_pdf(21)))
    assert calls["n"] == 2

    # ...and the auto-chain still lands on pymupdf:
    pages = parser_mod.parse_pdf(io.BytesIO(_pdf(21)))
    assert pages and any("page 1" in p.text for p in pages)
