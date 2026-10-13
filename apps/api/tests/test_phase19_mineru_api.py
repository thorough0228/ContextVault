"""Phase 19: mineru.net official API client — mocked wire tests.

The mineru engine now parses through mineru.net (v4 精准解析:
file-urls/batch → PUT → poll → result zip → full.md) instead of the
local CLI. Every test runs on ``httpx.MockTransport`` — no network.
"""

from __future__ import annotations

import io
import json
import zipfile

import httpx
import pytest

from app.ingest import parser as parser_mod
from app.ingest.mineru_api import MineruApiError, fetch_markdown_via_api
from app.ingest.mineru_convert import clean_mineru_markdown

PDF_BYTES = b"%PDF-1.4 fake paper bytes"
MD = "# Paper\n\n## Section 1\nBody text.\n"


def _zip_bytes(md: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", md)
    return buf.getvalue()


def _fetch(transport, *, token="tok-123", timeout_s=10.0, poll_interval=0.0,
           pdf=PDF_BYTES):
    return fetch_markdown_via_api(
        pdf, token=token, base_url="https://mineru.net",
        timeout_s=timeout_s, poll_interval=poll_interval,
        transport=transport,
    )


def test_success_path_returns_markdown_pages():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/api/v4/file-urls/batch":
            assert request.headers["Authorization"] == "Bearer tok-123"
            body = json.loads(request.content)
            assert body["files"][0]["name"] == "input.pdf"
            assert body["is_ocr"] is True and body["enable_table"] is True
            return httpx.Response(200, json={
                "code": 0,
                "data": {"batch_id": "b1", "file_urls": ["https://upload/x"]},
            })
        if request.method == "PUT":
            assert request.content == PDF_BYTES
            return httpx.Response(200)
        if request.url.path == "/api/v4/extract-results/batch/b1":
            polls = calls.count(("GET", "/api/v4/extract-results/batch/b1")) - 1
            state = "running" if polls == 0 else "done"
            extra = {"full_zip_url": "https://result/z.zip"} if state == "done" else {}
            return httpx.Response(200, json={
                "code": 0,
                "data": {"extract_result": [{"state": state, **extra}]},
            })
        if request.url.path.endswith("z.zip"):
            return httpx.Response(200, content=_zip_bytes(MD))
        return httpx.Response(500, json={"code": -1})

    md = _fetch(httpx.MockTransport(handler))
    assert md.startswith("# Paper")

    pages = parser_mod._markdown_to_pages(clean_mineru_markdown(md))
    assert [p.page_number for p in pages] == [1, 2]

    # wire order: submit → PUT → poll(running) → poll(done) → zip
    assert [m for m, _ in calls] == ["POST", "PUT", "GET", "GET", "GET"]


def test_failed_state_raises_with_err_msg():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={
                "code": 0,
                "data": {"batch_id": "b1", "file_urls": ["https://upload/x"]},
            })
        if request.method == "PUT":
            return httpx.Response(200)
        return httpx.Response(200, json={
            "code": 0,
            "data": {"extract_result": [
                {"state": "failed", "err_msg": "file is corrupted"}]},
        })

    with pytest.raises(MineruApiError, match="file is corrupted"):
        _fetch(httpx.MockTransport(handler))


def test_poll_timeout_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={
                "code": 0,
                "data": {"batch_id": "b1", "file_urls": ["https://upload/x"]},
            })
        if request.method == "PUT":
            return httpx.Response(200)
        return httpx.Response(200, json={
            "code": 0,
            "data": {"extract_result": [{"state": "running"}]},
        })

    with pytest.raises(MineruApiError, match="timed out"):
        _fetch(
            httpx.MockTransport(handler), timeout_s=0.05, poll_interval=0.005,
        )


def test_auth_failure_is_terminal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": "A0202", "msg": "token error"})

    with pytest.raises(MineruApiError, match="auth failed"):
        _fetch(httpx.MockTransport(handler), token="bad-token")


def test_empty_token_fails_fast_without_network():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call may happen without a token")

    with pytest.raises(MineruApiError, match="MINERU_API_TOKEN"):
        _fetch(httpx.MockTransport(handler), token="")


def test_submit_rejected_code_surfaces():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": -60009, "msg": "queue full"})

    with pytest.raises(MineruApiError, match="queue full"):
        _fetch(httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# parser-level: empty token → auto-chain degrades to pymupdf
# ---------------------------------------------------------------------------


def _tiny_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello mineru api fallback")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_parse_pdf_mineru_without_token_degrades(client_with_db):
    """conftest pins MINERU_API_TOKEN off — _parse_pdf_mineru must fail
    fast with a clear message and parse_pdf must fall through to the
    next engine (pymupdf), matching the documented auto-chain contract."""
    pages = parser_mod.parse_pdf(io.BytesIO(_tiny_pdf()))
    assert pages, "auto-chain must degrade to pymupdf when token is absent"
    assert any("mineru api fallback" in p.text for p in pages)
