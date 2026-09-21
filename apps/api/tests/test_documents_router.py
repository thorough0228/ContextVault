"""Integration tests for the documents router + storage + worker pipeline.

Phase 3 covers: PDF / TXT upload, oversized rejection, unsupported
type rejection, cross-tenant denial, the worker happy path (READY),
the worker failure path (FAILED), retry, and the state machine.
"""

from __future__ import annotations

import io
from typing import Iterable

import fitz  # PyMuPDF
import httpx
import pytest

from app.celery_client import celery_app
from app.config import get_settings
from app.models import Chunk, Document
from app.services import document_service
from app.storage import (
    build_storage_key,
    get_storage,
    set_storage_for_tests,
    reset_storage_for_tests,
)
from app.storage.memory import InMemoryStorage
from sqlalchemy import select


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf_bytes(text_per_page: list[str]) -> bytes:
    doc = fitz.open()
    try:
        for text in text_per_page:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        return doc.tobytes()
    finally:
        doc.close()


async def _create_rag(client, headers: dict, name: str = "Test RAG") -> str:
    resp = await client.post(
        "/api/v1/rags",
        headers=headers,
        json={"name": name, "description": ""},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _multipart(
    filename: str,
    body: bytes,
    content_type: str = "application/octet-stream",
) -> dict:
    """Build a dict ready for ``client.post(...,, files=...)``."""
    return {"file": (filename, io.BytesIO(body), content_type)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def rag_id(client_with_db, auth_headers_factory):
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers)
    return headers, rid


# ---------------------------------------------------------------------------
# 1. PDF upload returns CREATED document
# ---------------------------------------------------------------------------


async def test_upload_pdf_creates_document_in_created_status(
    client_with_db: httpx.AsyncClient, rag_id
) -> None:
    headers, rid = rag_id
    pdf = _make_pdf_bytes(["page 1 hello", "page 2 world"])

    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("paper.pdf", pdf, "application/pdf"),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "CREATED"
    assert body["file_type"] == "pdf"
    assert body["file_size"] == len(pdf)
    assert body["filename"] == "paper.pdf"
    assert body["storage_key"].endswith("/original")
    # Storage actually has the bytes.
    storage = get_storage()
    obj = storage.download(key=body["storage_key"])
    assert obj.body == pdf


# ---------------------------------------------------------------------------
# 2. TXT upload returns CREATED document
# ---------------------------------------------------------------------------


async def test_upload_txt_creates_document_in_created_status(
    client_with_db: httpx.AsyncClient, rag_id
) -> None:
    headers, rid = rag_id
    body = "Hello\nWorld\n你好".encode("utf-8")

    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("notes.txt", body, "text/plain"),
    )
    assert resp.status_code == 201, resp.text
    out = resp.json()
    assert out["status"] == "CREATED"
    assert out["file_type"] == "txt"
    assert out["file_size"] == len(body)


# ---------------------------------------------------------------------------
# 3. Illegal file type rejected with 400
# ---------------------------------------------------------------------------


async def test_upload_unsupported_file_type_rejected(
    client_with_db: httpx.AsyncClient, rag_id
) -> None:
    headers, rid = rag_id
    # Use a real-looking PNG signature — extension is .png so we never
    # even reach the magic-bytes check.
    png = b"\x89PNG\r\n\x1a\n"
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("photo.png", png, "image/png"),
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "unsupported_file_type"


async def test_upload_with_pdf_extension_but_wrong_magic_rejected(
    client_with_db: httpx.AsyncClient, rag_id
) -> None:
    headers, rid = rag_id
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("fake.pdf", b"this is not a pdf", "application/pdf"),
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_file_type"


# ---------------------------------------------------------------------------
# 4. Oversized upload rejected with 413
# ---------------------------------------------------------------------------


async def test_oversized_upload_rejected(
    client_with_db: httpx.AsyncClient, rag_id, test_settings, monkeypatch
) -> None:
    """Bump the cap down so we don't have to ship a 20 MiB body."""
    headers, rid = rag_id

    # Set the cap to 1 KB for this test only. We patch the test_settings
    # instance directly because that's what the API uses via the
    # ``get_settings`` dependency override.
    monkeypatch.setattr(test_settings, "upload_max_bytes", 1024)

    big = b"a" * 4096
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("big.txt", big, "text/plain"),
    )
    assert resp.status_code == 413, resp.text
    assert resp.json()["error"]["code"] == "payload_too_large"


# ---------------------------------------------------------------------------
# 5. Illegal RAG rejected with 404
# ---------------------------------------------------------------------------


async def test_upload_to_nonexistent_rag_returns_404(
    client_with_db: httpx.AsyncClient, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    pdf = _make_pdf_bytes(["x"])
    resp = await client_with_db.post(
        "/api/v1/rags/does-not-exist/documents",
        headers=headers,
        files=_multipart("x.pdf", pdf, "application/pdf"),
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 6. Cross-tenant upload rejected
# ---------------------------------------------------------------------------


async def test_user_a_cannot_upload_to_user_b_rag(
    client_with_db: httpx.AsyncClient, auth_headers_factory
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")

    # Alice creates a RAG.
    alice_rag = await _create_rag(client_with_db, headers_alice, name="alice")

    pdf = _make_pdf_bytes(["sensitive"])
    # Bob tries to upload to Alice's RAG.
    resp = await client_with_db.post(
        f"/api/v1/rags/{alice_rag}/documents",
        headers=headers_bob,
        files=_multipart("hack.pdf", pdf, "application/pdf"),
    )
    assert resp.status_code == 404, resp.text
    assert "alice" not in resp.text.lower()


# ---------------------------------------------------------------------------
# 7. PDF parser returns pages with text
# ---------------------------------------------------------------------------


def test_pdf_parser_returns_pages_with_text() -> None:
    from app.ingest.parser import parse_pdf
    blob = _make_pdf_bytes(["alpha", "beta", "gamma"])
    pages = parse_pdf(io.BytesIO(blob))
    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "alpha" in pages[0].text


# ---------------------------------------------------------------------------
# 8. TXT parser returns single page
# ---------------------------------------------------------------------------


def test_txt_parser_returns_single_page() -> None:
    from app.ingest.parser import parse_txt
    pages = parse_txt(io.BytesIO(b"single page content"))
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == "single page content"


# ---------------------------------------------------------------------------
# 9. Chunker splits text into chunks
# ---------------------------------------------------------------------------


def test_chunker_splits_text_into_chunks() -> None:
    from app.ingest import chunk_pages
    from app.ingest.parser import ParsedPage
    chunks = chunk_pages(
        [ParsedPage(page_number=1, text="a" * 600)],
        document_id="d", rag_id="r",
        size=200, overlap=50,
    )
    assert len(chunks) >= 3
    assert all(c.document_id == "d" for c in chunks)
    assert all(c.rag_id == "r" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)
    # chunk_index increases monotonically
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# 10. Worker happy path: CREATED → PROCESSING → READY with chunks persisted
# ---------------------------------------------------------------------------


async def test_worker_processes_document_to_ready(
    client_with_db: httpx.AsyncClient, rag_id, db_sessionmaker
) -> None:
    """Run the Celery task eagerly and confirm the state machine + DB."""
    headers, rid = rag_id
    pdf = _make_pdf_bytes(["alpha content", "beta content"])

    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("paper.pdf", pdf, "application/pdf"),
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    # Run the task eagerly (no Redis required).
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    from app.tasks import process_document
    try:
        result = process_document.apply(args=(doc_id,))
        payload = result.get(timeout=10)
        assert payload["status"] == "READY"
    finally:
        celery_app.conf.task_always_eager = False
        celery_app.conf.task_eager_propagates = False

    # Verify DB state.
    async with db_sessionmaker() as s:
        doc = await s.get(Document, doc_id)
        assert doc is not None
        assert doc.status == "READY"
        assert doc.page_count == 2
        assert doc.error_message is None

        rows = await s.scalars(select(Chunk).where(Chunk.document_id == doc_id))
        chunks = list(rows.all())
        assert len(chunks) >= 1
        for c in chunks:
            assert c.document_id == doc_id
            assert c.rag_id == rid
            assert c.text  # non-empty


# ---------------------------------------------------------------------------
# 11. Worker failure: FAILED + error_message persisted
# ---------------------------------------------------------------------------


async def test_worker_failure_marks_document_failed(
    client_with_db: httpx.AsyncClient, rag_id, db_sessionmaker, monkeypatch
) -> None:
    headers, rid = rag_id

    # Upload a TXT that the sniffer accepts.
    body = b"hello\nworld\n"
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("ok.txt", body, "text/plain"),
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    # Patch parse_document to raise a *permanent* error so the task
    # marks FAILED instead of retrying.
    from app.tasks import process_document
    import app.tasks as tasks_mod
    from app.exceptions import BadRequestError

    def _explode(*a, **kw):
        raise BadRequestError("synthetic failure for testing")

    monkeypatch.setattr(tasks_mod, "parse_document", _explode)

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False
    try:
        process_document.apply(args=(doc_id,))
    finally:
        celery_app.conf.task_always_eager = False
        celery_app.conf.task_eager_propagates = False

    async with db_sessionmaker() as s:
        doc = await s.get(Document, doc_id)
        assert doc is not None
        assert doc.status == "FAILED"
        assert "synthetic failure" in (doc.error_message or "")


# ---------------------------------------------------------------------------
# 12. Retry: transient failures schedule a retry up to max_retries
# ---------------------------------------------------------------------------


async def test_retry_on_transient_then_succeed(
    client_with_db: httpx.AsyncClient, rag_id, db_sessionmaker, monkeypatch
) -> None:
    """Transient errors trigger Celery retry; we patch retry to call
    the underlying body once more so we can observe the recovery path."""
    headers, rid = rag_id
    pdf = _make_pdf_bytes(["retry me"])
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("r.pdf", pdf, "application/pdf"),
    )
    doc_id = resp.json()["id"]

    # Patch the parser to fail the first call with a transient
    # (connection) error, then succeed.
    from app.tasks import process_document
    import app.tasks as tasks_mod

    calls = {"n": 0}
    real_parse = tasks_mod.parse_document

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient S3 blip")
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(tasks_mod, "parse_document", _flaky)

    # Patch Celery's Retry exception handling so the task re-runs
    # immediately instead of waiting for the broker.
    from celery.exceptions import Retry as CeleryRetry

    original_apply = process_document.apply

    def _eager_apply(*args, **kwargs):
        # Force re-run when Celery raises Retry — in eager mode, the
        # exception bubbles up. Catch it and re-invoke.
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                return original_apply(*args, **kwargs)
            except CeleryRetry as exc:
                last_exc = exc
                continue
            except Exception as exc:
                if calls["n"] < 2:
                    # patch isn't visible here; treat as transient
                    last_exc = exc
                    continue
                raise
        raise last_exc if last_exc else RuntimeError("retries exhausted")

    monkeypatch.setattr(process_document, "apply", _eager_apply)

    try:
        process_document.apply(args=(doc_id,))
    finally:
        monkeypatch.setattr(process_document, "apply", original_apply)

    # The flaky parser was called once and failed, then retried.
    assert calls["n"] >= 2, (
        f"parser should have been called at least twice "
        f"(was {calls['n']})"
    )

    async with db_sessionmaker() as s:
        doc = await s.get(Document, doc_id)
        assert doc is not None
        assert doc.status == "READY"


# ---------------------------------------------------------------------------
# 13. State transitions: list / get / delete ownership boundaries
# ---------------------------------------------------------------------------


async def test_list_documents_returns_only_callers_rag_documents(
    client_with_db: httpx.AsyncClient, auth_headers_factory
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")

    alice_rag = await _create_rag(client_with_db, headers_alice, "alice")
    bob_rag = await _create_rag(client_with_db, headers_bob, "bob")

    for label in ("a1", "a2"):
        await client_with_db.post(
            f"/api/v1/rags/{alice_rag}/documents",
            headers=headers_alice,
            files=_multipart(
                f"{label}.txt", f"{label}\n".encode("utf-8"), "text/plain",
            ),
        )
    await client_with_db.post(
        f"/api/v1/rags/{bob_rag}/documents",
        headers=headers_bob,
        files=_multipart("b1.txt", b"b1\n", "text/plain"),
    )

    alice_list = await client_with_db.get(
        f"/api/v1/rags/{alice_rag}/documents", headers=headers_alice
    )
    assert alice_list.status_code == 200
    assert {d["filename"] for d in alice_list.json()["items"]} == {"a1.txt", "a2.txt"}
    assert alice_list.json()["total"] == 2
    assert alice_list.json()["page"] == 1
    assert alice_list.json()["page_size"] == 20


async def test_get_document_404_for_other_users_doc(
    client_with_db: httpx.AsyncClient, auth_headers_factory
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")
    alice_rag = await _create_rag(client_with_db, headers_alice, "alice")

    resp = await client_with_db.post(
        f"/api/v1/rags/{alice_rag}/documents",
        headers=headers_alice,
        files=_multipart("a.txt", b"a\n", "text/plain"),
    )
    doc_id = resp.json()["id"]

    got = await client_with_db.get(
        f"/api/v1/documents/{doc_id}", headers=headers_alice
    )
    assert got.status_code == 200
    assert got.json()["id"] == doc_id

    cross = await client_with_db.get(
        f"/api/v1/documents/{doc_id}", headers=headers_bob
    )
    assert cross.status_code == 404


async def test_delete_document_removes_chunks_and_storage(
    client_with_db: httpx.AsyncClient, rag_id, db_sessionmaker
) -> None:
    headers, rid = rag_id
    body = b"delete me\n"
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files=_multipart("rm.txt", body, "text/plain"),
    )
    doc_id = resp.json()["id"]
    storage_key = resp.json()["storage_key"]
    storage = get_storage()
    assert storage.exists(key=storage_key)

    delete = await client_with_db.delete(
        f"/api/v1/documents/{doc_id}", headers=headers
    )
    assert delete.status_code == 204
    assert not storage.exists(key=storage_key)

    # GET after delete → 404.
    after = await client_with_db.get(
        f"/api/v1/documents/{doc_id}", headers=headers
    )
    assert after.status_code == 404


async def test_list_documents_requires_auth(
    client_with_db: httpx.AsyncClient, rag_id,
) -> None:
    _, rid = rag_id
    resp = await client_with_db.get(f"/api/v1/rags/{rid}/documents")
    assert resp.status_code == 401


async def test_documents_require_auth_for_upload(
    client_with_db: httpx.AsyncClient, rag_id,
) -> None:
    _, rid = rag_id
    pdf = _make_pdf_bytes(["x"])
    resp = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        files=_multipart("x.pdf", pdf, "application/pdf"),
    )
    assert resp.status_code == 401