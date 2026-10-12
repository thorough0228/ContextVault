"""MinerU output → clean markdown converter (Phase 15).

MinerU (https://github.com/opendatalab/MinerU) parses PDFs into a
``{name}.md`` plus ``images/`` and ``{name}_content_list.json``. Two
problems for a text-only RAG pipeline:

1. Tables arrive as inline **HTML fragments** in the .md;
2. Images are referenced with relative ``![](images/<sha>.jpg)`` paths
   that mean nothing once the file leaves the output directory.

This script (and the parser helper functions it exports) normalises
one MinerU output directory into a single clean markdown file ready
for upload:

* inline HTML tables → markdown pipe tables (standard-library
  html.parser; colspan/rowspan cells flatten to empty cells);
* ``![](images/...)`` references → removed;
* equations stay as LaTeX ($$...$$);
* ``--from-content-list`` rebuilds from ``content_list.json``
  instead (reading order, headings via ``text_level``, tables
  converted from ``table_body`` HTML, captions as italic lines).

Usage:
    python scripts/mineru_to_markdown.py <mineru_output_dir> -o out.md
    (the CLI is a thin wrapper over this module)
"""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# HTML table → markdown
# ---------------------------------------------------------------------------


class _TableExtractor(HTMLParser):
    """Collect top-level <table> fragments as row/cell text matrices."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[str]]] = []
        self._depth = 0
        self._rows: List[List[str]] = []
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._rows = []
        elif self._depth and tag == "tr":
            self._rows.append([])
        elif self._depth and tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table" and self._depth:
            self._depth -= 1
            if self._depth == 0:
                self.tables.append(self._rows)
                self._rows = []
        elif self._depth and tag in ("td", "th") and self._cell is not None:
            self._rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(html: str) -> str:
    """Convert the first <table> in ``html`` to a markdown pipe table;
    "" when no table is present."""

    parser = _TableExtractor()
    try:
        parser.feed(html)
    except Exception:
        return ""
    if not parser.tables:
        return ""
    rows = parser.tables[0]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]

    lines = [
        "| " + " | ".join(c.replace("|", "\\|") for c in norm[0]) + " |",
        "|" + "---|" * width,
    ]
    for row in norm[1:]:
        lines.append(
            "| " + " | ".join(c.replace("|", "\\|") for c in row) + " |"
        )
    return "\n".join(lines)


_IMG_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_PAGE_COMMENT_RE = re.compile(r"<!-- page \d+ of \d+ -->")
_HTML_WRAPPER_RE = re.compile(r"</?html>|</?body>", re.IGNORECASE)


def clean_mineru_markdown(md: str) -> str:
    """Normalise a MinerU .md: HTML tables → pipe tables, image
    references removed, html/body wrappers stripped, blank lines
    collapsed. Headings, LaTeX equations and text pass through."""

    out_lines: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        block = "\n".join(buffer)
        for table_html in re.findall(
            r"<table.*?</table>", block, re.DOTALL | re.IGNORECASE
        ):
            md_table = html_table_to_markdown(table_html)
            block = block.replace(table_html, md_table or "")
        block = _HTML_WRAPPER_RE.sub("", block)
        block = _IMG_REF_RE.sub("", block)
        block = re.sub(r"[ \t]+\n", "\n", block)
        block = re.sub(r"\n{3,}", "\n\n", block).strip()
        if block:
            out_lines.append(block)
        buffer.clear()

    for line in md.splitlines():
        if buffer:
            buffer.append(line)
            if "</table>" in line.lower():
                flush_buffer()
        elif "<table" in line.lower():
            buffer.append(line)
            if "</table>" in line.lower():
                flush_buffer()
        else:
            out_lines.append(_HTML_WRAPPER_RE.sub("", _IMG_REF_RE.sub("", line)))
    flush_buffer()

    text = _PAGE_COMMENT_RE.sub("", "\n".join(out_lines))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n" if text else ""


# ---------------------------------------------------------------------------
# content_list.json → markdown (rebuild path)
# ---------------------------------------------------------------------------


def content_list_to_markdown(entries: list) -> str:
    """Rebuild markdown from a MinerU ``content_list.json`` array in
    reading order: headings from ``text_level``, tables from
    ``table_body`` HTML, image captions as italic lines, equations as
    LaTeX."""

    blocks: List[str] = []
    for entry in entries or []:
        etype = entry.get("type")
        if etype == "text":
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            level = int(entry.get("text_level") or 0)
            blocks.append("#" * level + " " + text if level else text)
        elif etype == "table":
            caption = " ".join(entry.get("table_caption") or []).strip()
            md_table = html_table_to_markdown(entry.get("table_body") or "")
            if caption:
                blocks.append(f"**{caption}**")
            if md_table:
                blocks.append(md_table)
            footnote = " ".join(entry.get("table_footnote") or []).strip()
            if footnote:
                blocks.append(f"*{footnote}*")
        elif etype == "image":
            caption = " ".join(entry.get("img_caption") or []).strip()
            if caption:
                blocks.append(f"*{caption}*")
        elif etype == "equation":
            latex = (entry.get("text") or "").strip()
            if latex:
                blocks.append(latex)
    return ("\n\n".join(blocks)).strip() + "\n" if blocks else ""


# ---------------------------------------------------------------------------
# Directory driver
# ---------------------------------------------------------------------------


def convert_directory(directory: Path, *, from_content_list: bool = False) -> str:
    """Convert one MinerU output directory into clean markdown text."""

    md_file = next(iter(sorted(directory.rglob("*.md"))), None)
    cl_file = next(iter(sorted(directory.rglob("*_content_list.json"))), None)

    if from_content_list or md_file is None:
        if cl_file is None:
            raise SystemExit(
                f"no .md or *_content_list.json found under {directory}"
            )
        return content_list_to_markdown(
            json.loads(cl_file.read_text(encoding="utf-8"))
        )
    return clean_mineru_markdown(md_file.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a MinerU output directory into clean markdown."
    )
    ap.add_argument("directory", help="MinerU output directory")
    ap.add_argument(
        "-o", "--output", required=True, help="clean .md path to write"
    )
    ap.add_argument(
        "--from-content-list",
        action="store_true",
        help="rebuild from content_list.json instead of the .md",
    )
    args = ap.parse_args()

    text = convert_directory(
        Path(args.directory), from_content_list=args.from_content_list
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
