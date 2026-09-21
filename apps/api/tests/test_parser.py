"""Tests for PDF + TXT parsers."""

from __future__ import annotations

import io

import fitz  # PyMuPDF
import pytest

from app.ingest.parser import (
    parse_csv,
    parse_document,
    parse_json,
    parse_pdf,
    parse_txt,
)


def _make_pdf_bytes(text_per_page: list[str]) -> bytes:
    """Build a minimal in-memory PDF with one page per text entry.

    We use PyMuPDF itself rather than shipping a static fixture —
    this keeps the test deterministic across versions.
    """
    doc = fitz.open()
    try:
        for text in text_per_page:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        return doc.tobytes()
    finally:
        doc.close()


def test_parse_txt_decodes_utf8() -> None:
    body = "Line 1\nLine 2\n你好".encode("utf-8")
    pages = parse_txt(io.BytesIO(body))
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == "Line 1\nLine 2\n你好"


def test_parse_txt_empty_returns_one_page() -> None:
    pages = parse_txt(io.BytesIO(b""))
    assert len(pages) == 1
    assert pages[0].text == ""


def test_parse_pdf_returns_one_page_per_page() -> None:
    blob = _make_pdf_bytes(["page one text", "page two text", "page three text"])
    pages = parse_pdf(io.BytesIO(blob))
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "page one text" in pages[0].text
    assert "page two text" in pages[1].text
    assert "page three text" in pages[2].text


def test_parse_pdf_handles_garbage_input() -> None:
    # PyMuPDF raises on truly malformed PDFs; surface that as-is so the
    # worker can decide transient vs permanent.
    with pytest.raises(Exception):
        parse_pdf(io.BytesIO(b"not a pdf"))


def test_parse_document_dispatches_by_file_type() -> None:
    txt_pages = parse_document(
        file_type="txt", stream=io.BytesIO(b"hi")
    )
    assert len(txt_pages) == 1 and txt_pages[0].text == "hi"

    blob = _make_pdf_bytes(["dispatcher"])
    pdf_pages = parse_document(
        file_type="pdf", stream=io.BytesIO(blob)
    )
    assert len(pdf_pages) == 1 and "dispatcher" in pdf_pages[0].text


def test_parse_document_rejects_unknown_file_type() -> None:
    with pytest.raises(ValueError):
        parse_document(file_type="docx", stream=io.BytesIO(b""))

def test_parsed_page_strips_nul_bytes() -> None:
    """Postgres TEXT rejects 0x00; PDF extraction can leak them from
    embedded-font glyphs. The ParsedPage funnel must strip them."""
    from app.ingest.parser import ParsedPage

    page = ParsedPage(page_number=1, text="before\x00after")
    assert page.text == "beforeafter"


def test_parse_txt_strips_nul_bytes() -> None:
    pages = parse_txt(io.BytesIO("bad\x00text".encode("utf-8")))
    assert pages[0].text == "badtext"


# ---- CSV / JSON / NDJSON (structured files) ---------------------------


def test_parse_csv_document_column_preferred() -> None:
    body = (
        'id,title,document\n'
        '1,First,"First doc text"\n'
        '2,Second,"Second doc text"'
    ).encode("utf-8")
    pages = parse_csv(io.BytesIO(body))
    assert len(pages) == 1
    assert "First doc text" in pages[0].text
    assert "Second doc text" in pages[0].text
    # non-text columns are not rendered when a document column exists
    assert "title" not in pages[0].text.lower()


def test_parse_csv_generic_rows_render_key_values() -> None:
    body = "name,price\nApple,10\nPear,20".encode("utf-8")
    pages = parse_csv(io.BytesIO(body))
    text = pages[0].text
    assert "name: Apple" in text
    assert "price: 10" in text
    assert "name: Pear" in text


def test_parse_json_array_uses_text_field() -> None:
    import json

    data = [
        {"id": "c0", "text": "chunk zero"},
        {"id": "c1", "text": "chunk one"},
    ]
    pages = parse_json(io.BytesIO(json.dumps(data).encode("utf-8")))
    assert pages[0].page_number == 1
    assert "chunk zero" in pages[0].text
    assert "chunk one" in pages[0].text
    assert '"id"' not in pages[0].text


def test_parse_json_ndjson_fallback() -> None:
    body = b'{"text": "line one"}\n{"text": "line two"}\n'
    pages = parse_json(io.BytesIO(body))
    assert "line one" in pages[0].text
    assert "line two" in pages[0].text


def test_parse_json_invalid_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_json(io.BytesIO(b"{definitely not json"))


def test_parse_csv_strips_nul_bytes() -> None:
    body = 'document\nbad\x00text'.encode("utf-8")
    pages = parse_csv(io.BytesIO(body))
    assert pages[0].text == "badtext"
