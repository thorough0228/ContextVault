"""Phase 13: document update — hash detection, replace, bluegreen."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.offline

OLD = b"quarterly report: revenue was 100 units and costs were 80 units."
NEW = b"quarterly report: revenue was 250 units and costs were 90 units. UPDATED"


async def _setup_rag_with_doc(client_with_db, auth_headers_factory, email):
    headers = await auth_headers_factory(email)
    rid = (
        await client_with_db.post(
            "/api/v1/rags", headers=headers, json={"name": "upd"}
        )
    ).json()["id"]
    up = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": ("report.txt", OLD, "text/plain")},
    )
    assert up.status_code == 201, up.text
    doc_id = up.json()["id"]
    # eager worker: ingestion already finished
    detail = await client_with_db.get(
        f"/api/v1/documents/{doc_id}", headers=headers
    )
    assert detail.json()["status"] == "READY"
    return headers, rid, doc_id


async def test_upload_stores_content_hash(
    client_with_db, db_session, auth_headers_factory
):
    from app.models import Document
    from sqlalchemy import select

    headers, rid, doc_id = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "hash@example.com"
    )
    row = (
        await db_session.execute(
            select(Document.content_hash).where(Document.id == doc_id)
        )
    ).scalar()
    assert row and len(row) == 64


async def test_duplicate_filename_returns_409(
    client_with_db, auth_headers_factory
):
    headers, rid, _ = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "dup@example.com"
    )
    dup = await client_with_db.post(
        f"/api/v1/rags/{rid}/documents",
        headers=headers,
        files={"file": ("report.txt", b"different bytes entirely", "text/plain")},
    )
    assert dup.status_code == 409
    assert "PUT" in dup.json()["error"]["message"]


async def test_put_unchanged_content_is_idempotent(
    client_with_db, auth_headers_factory
):
    headers, rid, doc_id = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "same@example.com"
    )
    resp = await client_with_db.put(
        f"/api/v1/documents/{doc_id}/content",
        headers=headers,
        files={"file": ("report.txt", OLD, "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["updated"] is False
    assert body["reason"] == "content_unchanged"


def _set_strategy(test_settings, value: str) -> None:
    """The app overrides Depends(get_settings) with this fixed instance,
    so the strategy must be set on it — env vars never reach it."""

    test_settings.document_update_strategy = value


async def test_put_replace_reingests_new_content(
    client_with_db, db_session, auth_headers_factory, test_settings
):
    from app.models import DocumentChunk
    from sqlalchemy import select

    _set_strategy(test_settings, "replace")
    headers, rid, doc_id = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "replace@example.com"
    )
    resp = await client_with_db.put(
        f"/api/v1/documents/{doc_id}/content",
        headers=headers,
        files={"file": ("report.txt", NEW, "text/plain")},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["updated"] is True
    assert body["strategy"] == "replace"

    # eager worker: new chunks replaced old ones under the SAME doc id
    rows = (
        await db_session.execute(
            select(DocumentChunk.chunk_text).where(
                DocumentChunk.document_id == doc_id
            )
        )
    ).scalars().all()
    joined = "\n".join(rows)
    assert "UPDATED" in joined
    assert "revenue was 250 units" in joined
    assert "revenue was 100 units" not in joined


async def test_bluegreen_update_supersedes_and_hides_old(
    client_with_db, auth_headers_factory, test_settings
):
    _set_strategy(test_settings, "bluegreen")
    headers, rid, old_id = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "bg@example.com"
    )
    resp = await client_with_db.put(
        f"/api/v1/documents/{old_id}/content",
        headers=headers,
        files={"file": ("report.txt", NEW, "text/plain")},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["strategy"] == "bluegreen"
    new_id = body["new_document_id"]
    assert new_id != old_id

    # eager worker completed: old is superseded, new is READY
    listing = await client_with_db.get(
        f"/api/v1/rags/{rid}/documents", headers=headers
    )
    by_id = {d["id"]: d for d in listing.json()["items"]}
    assert by_id[old_id]["superseded_by"] == new_id
    assert by_id[new_id]["status"] == "READY"

    # retrieval sees ONLY the new content
    hits = (
        await client_with_db.post(
            f"/api/v1/rags/{rid}/search",
            headers=headers,
            json={"query": "revenue", "top_k": 10},
        )
    ).json()["hits"]
    assert hits, "superseded filtering must not empty the results"
    for h in hits:
        assert "UPDATED" in h["chunk_text"]
        assert h["document_id"] == new_id


async def test_rollback_restores_old_and_removes_new(
    client_with_db, auth_headers_factory, test_settings
):
    _set_strategy(test_settings, "bluegreen")
    headers, rid, old_id = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "rb@example.com"
    )
    put = await client_with_db.put(
        f"/api/v1/documents/{old_id}/content",
        headers=headers,
        files={"file": ("report.txt", NEW, "text/plain")},
    )
    new_id = put.json()["new_document_id"]

    rb = await client_with_db.post(
        f"/api/v1/documents/{new_id}/rollback", headers=headers
    )
    assert rb.status_code == 200
    body = rb.json()
    assert body["restored_document_id"] == old_id

    listing = await client_with_db.get(
        f"/api/v1/rags/{rid}/documents", headers=headers
    )
    by_id = {d["id"]: d for d in listing.json()["items"]}
    assert new_id not in by_id, "new document must be gone after rollback"
    assert by_id[old_id]["superseded_by"] is None

    hits = (
        await client_with_db.post(
            f"/api/v1/rags/{rid}/search",
            headers=headers,
            json={"query": "revenue", "top_k": 10},
        )
    ).json()["hits"]
    assert hits and all("revenue was 100 units" in h["chunk_text"] for h in hits)


async def test_superseded_doc_cannot_be_updated_again(
    client_with_db, auth_headers_factory, test_settings
):
    _set_strategy(test_settings, "bluegreen")
    headers, rid, old_id = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "twice@example.com"
    )
    put = await client_with_db.put(
        f"/api/v1/documents/{old_id}/content",
        headers=headers,
        files={"file": ("report.txt", NEW, "text/plain")},
    )
    new_id = put.json()["new_document_id"]
    again = await client_with_db.put(
        f"/api/v1/documents/{old_id}/content",
        headers=headers,
        files={"file": ("report.txt", NEW + b" again", "text/plain")},
    )
    assert again.status_code == 409
    assert "superseded" in again.json()["error"]["message"]
    # the replacement is the update target
    ok = await client_with_db.put(
        f"/api/v1/documents/{new_id}/content",
        headers=headers,
        files={"file": ("report.txt", NEW + b" v3", "text/plain")},
    )
    assert ok.status_code == 202


async def test_put_unknown_strategy_rejected(
    client_with_db, auth_headers_factory, monkeypatch
):
    monkeypatch.setenv("DOCUMENT_UPDATE_STRATEGY", "replace")
    headers, rid, doc_id = await _setup_rag_with_doc(
        client_with_db, auth_headers_factory, "strat@example.com"
    )
    resp = await client_with_db.put(
        f"/api/v1/documents/{doc_id}/content?strategy=yolo",
        headers=headers,
        files={"file": ("report.txt", NEW, "text/plain")},
    )
    assert resp.status_code == 400
