"""Phase 20: mineru Agent lightweight API client — mocked wire tests.

The mineru engine parses through the free Agent API (no token;
markdown via CDN link). Every test runs on ``httpx.MockTransport`` —
no network. PDF splitting (>20 pages) is covered in
test_phase20_pdf_split.py.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.ingest import parser as parser_mod
from app.ingest.mineru_api import MineruApiError, fetch_markdown_via_api
from app.ingest.mineru_convert import clean_mineru_markdown

PDF_BYTES = b"%PDF-1.4 fake paper bytes"
MD = "# Paper\n\n## Section 1\nBody text.\n"


def _fetch(transport, *, timeout_s=10.0, poll_interval=0.0, pdf=PDF_BYTES):
    return fetch_markdown_via_api(
        pdf, base_url="https://mineru.net",
        timeout_s=timeout_s, poll_interval=poll_interval,
        transport=transport,
    )


def test_success_path_returns_markdown_pages():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/api/v1/agent/parse/file":
            body = json.loads(request.content)
            assert body["file_name"] == "input.pdf"
            assert body["is_ocr"] is True and body["enable_table"] is True
            assert "Authorization" not in request.headers, "Agent API is auth-free"
            return httpx.Response(200, json={
                "code": 0,
                "data": {
                    "task_id": "t1",
                    "file_url": "https://oss/upload?Expires=1",
                },
            })
        if request.method == "PUT":
            assert request.content == PDF_BYTES
            return httpx.Response(200)
        if request.url.path == "/api/v1/agent/parse/t1":
            polls = calls.count(("GET", "/api/v1/agent/parse/t1")) - 1
            state = "running" if polls == 0 else "done"
            extra = {
                "markdown_url": "https://cdn/full.md"
            } if state == "done" else {}
            return httpx.Response(200, json={
                "code": 0, "data": {"state": state, **extra},
            })
        if request.url.path.endswith("full.md"):
            return httpx.Response(200, text=MD)
        return httpx.Response(500, json={"code": -1})

    md = _fetch(httpx.MockTransport(handler))
    assert md.startswith("# Paper")

    pages = parser_mod._markdown_to_pages(clean_mineru_markdown(md))
    assert [p.page_number for p in pages] == [1, 2]

    # wire order: submit → PUT → poll(running) → poll(done) → markdown
    assert [m for m, _ in calls] == ["POST", "PUT", "GET", "GET", "GET"]


def test_failed_state_raises_with_err_code():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={
                "code": 0,
                "data": {"task_id": "t1", "file_url": "https://oss/upload"},
            })
        if request.method == "PUT":
            return httpx.Response(200)
        return httpx.Response(200, json={
            "code": 0,
            "data": {"state": "failed", "err_code": -30003,
                     "err_msg": "page count exceeds lightweight API limit"},
        })

    with pytest.raises(MineruApiError, match="page count exceeds"):
        _fetch(httpx.MockTransport(handler))


def test_poll_timeout_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={
                "code": 0,
                "data": {"task_id": "t1", "file_url": "https://oss/upload"},
            })
        if request.method == "PUT":
            return httpx.Response(200)
        return httpx.Response(200, json={
            "code": 0, "data": {"state": "running"},
        })

    with pytest.raises(MineruApiError, match="timed out"):
        _fetch(
            httpx.MockTransport(handler), timeout_s=0.05, poll_interval=0.005,
        )


def test_rate_limit_429_is_retried():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            attempts["n"] += 1
            if attempts["n"] < 2:
                return httpx.Response(429, json={"msg": "rate limited"})
            return httpx.Response(200, json={
                "code": 0,
                "data": {"task_id": "t1", "file_url": "https://oss/upload"},
            })
        if request.method == "PUT":
            return httpx.Response(200)
        if request.url.path.endswith("t1"):
            return httpx.Response(200, json={
                "code": 0,
                "data": {"state": "done",
                         "markdown_url": "https://cdn/full.md"},
            })
        return httpx.Response(200, text=MD)

    md = _fetch(httpx.MockTransport(handler))
    assert md.startswith("# Paper")
    assert attempts["n"] == 2, "429 must be retried, not terminal"


def test_submit_rejected_code_surfaces():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": -30001, "msg": "file too large"})

    with pytest.raises(MineruApiError, match="file too large"):
        _fetch(httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# parser-level: engine still degrades through the auto-chain
# ---------------------------------------------------------------------------


def test_parse_pdf_mineru_api_error_degrades(client_with_db, monkeypatch):
    """A terminal API failure must degrade to the next engine
    (pymupdf), matching the documented auto-chain contract."""
    import io

    def _boom(stream, *, timeout_s=600.0):
        raise RuntimeError("mineru api: mineru parse failed: boom")

    monkeypatch.setattr(parser_mod, "_parse_pdf_mineru", _boom)
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "fallback text")
    buf = io.BytesIO()
    doc.save(buf)

    pages = parser_mod.parse_pdf(io.BytesIO(buf.getvalue()))
    assert pages and any("fallback text" in p.text for p in pages)
