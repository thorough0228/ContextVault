"""Phase 15: MinerU output → markdown conversion tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.ingest.mineru_convert import (  # noqa: E402
    clean_mineru_markdown,
    content_list_to_markdown,
    convert_directory,
    html_table_to_markdown,
)

pytestmark = pytest.mark.offline

TABLE_HTML = (
    "<html><body><table><tr><td>Rubber</td><td>Hardness</td></tr>"
    "<tr><td>Tenergy 05</td><td>36</td></tr></table></body></html>"
)


class TestHtmlTableToMarkdown:
    def test_basic_table(self):
        md = html_table_to_markdown(TABLE_HTML)
        lines = md.splitlines()
        assert lines[0] == "| Rubber | Hardness |"
        assert lines[1] == "|---|---|"
        assert lines[2] == "| Tenergy 05 | 36 |"

    def test_ragged_rows_padded(self):
        md = html_table_to_markdown(
            "<table><tr><th>A</th><th>B</th></tr><tr><td>only</td></tr></table>"
        )
        assert "| only |  |" in md

    def test_pipe_escaped(self):
        md = html_table_to_markdown("<table><tr><td>a|b</td></tr></table>")
        assert "a\\|b" in md

    def test_entities_decoded(self):
        md = html_table_to_markdown("<table><tr><td>&lt;tag&gt; &amp; co</td></tr></table>")
        assert "<tag> & co" in md

    def test_no_table_returns_empty(self):
        assert html_table_to_markdown("<p>no table</p>") == ""

    def test_nested_content_ignores_inner_tags(self):
        md = html_table_to_markdown(
            "<table><tr><td><b>bold</b> cell</td></tr></table>"
        )
        assert "bold cell" in md


class TestCleanMineruMarkdown:
    def test_table_inline_in_md_converted(self):
        md = "# Report\n\n" + TABLE_HTML + "\n\n# Next\n"
        out = clean_mineru_markdown(md)
        assert "| Rubber | Hardness |" in out
        assert "<table" not in out
        assert "# Report" in out and "# Next" in out

    def test_image_refs_removed(self):
        md = "before\n\n![](images/abc123.jpg)\n\nafter\n\n![Fig 1](images/def.jpg)"
        out = clean_mineru_markdown(md)
        assert "images/" not in out
        assert "before" in out and "after" in out

    def test_latex_equations_preserved(self):
        md = "$$\nQ = f(P)\n$$\n\ntext after"
        out = clean_mineru_markdown(md)
        assert "Q = f(P)" in out

    def test_html_body_wrappers_stripped(self):
        out = clean_mineru_markdown("<html><body>\ntext\n</body></html>")
        assert out.strip() == "text"

    def test_blank_lines_collapsed(self):
        out = clean_mineru_markdown("a\n\n\n\n\nb")
        assert "\n\n\n" not in out


class TestContentListRebuild:
    def test_full_rebuild_reading_order(self):
        entries = [
            {"type": "text", "text_level": 1, "text": "Chapter One", "page_idx": 0},
            {"type": "text", "text": "Body paragraph.", "page_idx": 0},
            {"type": "table",
             "table_caption": ["Table 1 specs"],
             "table_body": TABLE_HTML,
             "table_footnote": ["source: internal"],
             "page_idx": 1},
            {"type": "image", "img_caption": ["Fig 1 curve"], "page_idx": 1},
            {"type": "equation", "text": "$$E=mc^2$$", "page_idx": 2},
        ]
        md = content_list_to_markdown(entries)
        lines = md.splitlines()
        assert lines[0] == "# Chapter One"
        assert "Body paragraph." in md
        assert "**Table 1 specs**" in md
        assert "| Rubber | Hardness |" in md
        assert "*source: internal*" in md
        assert "*Fig 1 curve*" in md
        assert "$$E=mc^2$$" in md
        # heading before body before table
        assert md.index("# Chapter One") < md.index("Body paragraph")
        assert md.index("Body paragraph") < md.index("| Rubber")

    def test_empty_entries(self):
        assert content_list_to_markdown([]) == ""
        assert content_list_to_markdown(None) == ""

    def test_text_without_level_is_body(self):
        md = content_list_to_markdown([{"type": "text", "text": "plain"}])
        assert md.strip() == "plain"
        assert not md.startswith("#")


class TestConvertDirectory:
    def test_prefers_md_and_cleans(self, tmp_path):
        out_dir = tmp_path / "sample" / "auto"
        out_dir.mkdir(parents=True)
        (out_dir / "sample.md").write_text(
            "# T\n\n" + TABLE_HTML + "\n\n![](images/x.jpg)\n", encoding="utf-8"
        )
        (out_dir / "sample_content_list.json").write_text("[]", encoding="utf-8")
        text = convert_directory(tmp_path)
        assert "| Rubber" not in text or "Hardness" in text  # table converted
        assert "images/" not in text

    def test_falls_back_to_content_list(self, tmp_path):
        out_dir = tmp_path / "s2" / "auto"
        out_dir.mkdir(parents=True)
        (out_dir / "s2_content_list.json").write_text(
            json.dumps([{"type": "text", "text_level": 1, "text": "H"}]),
            encoding="utf-8",
        )
        text = convert_directory(tmp_path)
        assert text.strip() == "# H"

    def test_nothing_found_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            convert_directory(tmp_path)


# ---------------------------------------------------------------------------
# .md file type + parse_markdown + PDF parser dispatch
# ---------------------------------------------------------------------------


def test_file_type_md(tmp_path):
    from app.ingest.file_type import detect_file_type

    body = b"# Title\n\ntext"
    assert detect_file_type(filename="doc.md", body=body) == "md"
    assert detect_file_type(filename="doc.markdown", body=body) == "md"


def test_file_type_md_invalid_utf8(tmp_path):
    from app.ingest.file_type import detect_file_type, FileTypeError

    with pytest.raises(FileTypeError, match="not valid UTF-8"):
        detect_file_type(filename="doc.md", body=b"\xff\xfe binary")


def test_parse_markdown_splits_at_h1_h2():
    import io

    from app.ingest.parser import parse_markdown

    md = "# Chapter One\nbody one\n## Section\nbody two\nplain tail"
    pages = parse_markdown(io.BytesIO(md.encode("utf-8")))
    # H1 and H2 open new pages; the heading-less tail stays with the
    # last chapter.
    assert len(pages) == 2
    assert pages[0].text.startswith("# Chapter One")
    assert pages[1].text.startswith("## Section")
    assert "plain tail" in pages[1].text


def test_parse_markdown_no_headings_single_page():
    import io

    from app.ingest.parser import parse_markdown

    pages = parse_markdown(io.BytesIO("just text\nno headings".encode()))
    assert len(pages) == 1


def test_pdf_parser_dispatch_auto_degrades_to_pymupdf(monkeypatch, tmp_path):
    """mineru fails (CLI missing) and unstructured fails (import error)
    → the auto chain must land on pymupdf and still extract text."""
    import io

    from app.ingest import parser as parser_mod
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PDF_PARSER", "auto")
    get_settings.cache_clear()

    calls = []

    def _fake_mineru(stream, *, timeout_s=600.0):
        calls.append("mineru")
        raise RuntimeError("mineru CLI not installed")

    monkeypatch.setattr(parser_mod, "_parse_pdf_mineru", _fake_mineru)

    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("unstructured"):
            raise ImportError("No module named 'unstructured'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello pymupdf fallback")
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = parser_mod.parse_pdf(io.BytesIO(pdf_bytes))
    assert calls == ["mineru"]
    assert any("pymupdf fallback" in p.text for p in pages)
    get_settings.cache_clear()


def test_pdf_parser_pinned_mineru_skips_other_engines(monkeypatch, tmp_path):
    import io

    from app.ingest import parser as parser_mod
    from app.config import get_settings

    monkeypatch.setenv("PDF_PARSER", "mineru")
    get_settings.cache_clear()

    def _fake_mineru(stream, *, timeout_s=600.0):
        return [type("P", (), {"page_number": 1, "text": "md page"})()]

    monkeypatch.setattr(parser_mod, "_parse_pdf_mineru", _fake_mineru)
    pages = parser_mod.parse_pdf(io.BytesIO(b"%PDF-fake"))
    assert pages[0].text == "md page"
    get_settings.cache_clear()


async def test_e2e_upload_md_ready(client_with_db, db_session, auth_headers_factory):
    from app.models import Document
    from sqlalchemy import select

    headers = await auth_headers_factory("md@example.com")
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "md"}
        )
    ).json()["id"]
    md_body = (
        "# Guide\n\nRubber hardness defines play style. Choose wisely "
        "for attacking strokes.\n\n## Blades\n\nWood gives feedback."
    ).encode()
    up = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": ("guide.md", md_body, "text/markdown")},
    )
    assert up.status_code == 201, up.text
    doc_id = up.json()["id"]
    assert up.json()["file_type"] == "md"

    for _ in range(60):
        detail = await client_with_db.get(
            f"/api/v1/documents/{doc_id}", headers=headers
        )
        if detail.json()["status"] == "READY":
            break
    assert detail.json()["status"] == "READY"

    hits = (
        await client_with_db.post(
            f"/api/v1/rags/{rid}/search",
            headers=headers,
            json={"query": "rubber hardness attacking", "top_k": 3},
        )
    ).json()["hits"]
    assert hits and any("rubber" in h["chunk_text"].lower() for h in hits)
