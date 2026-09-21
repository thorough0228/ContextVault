"""End-to-end happy path: register → login → create RAG → upload PDF
→ wait for READY → chat → citation."""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any, Dict, List

import fitz  # PyMuPDF
import httpx
import pytest


def _make_pdf_bytes(text: str) -> bytes:
    """A single-page PDF containing ``text``."""
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        return doc.tobytes()
    finally:
        doc.close()


async def _events(resp) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    async for raw in resp.aiter_lines():
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            raise AssertionError(f"non-JSON event line: {raw!r}")
    return out


# ----- E.1.1: PDF upload + chat returns citation ---------------------


async def test_e2e_pdf_upload_then_chat_returns_citation_with_document_id(
    client_with_db, auth_headers_factory
) -> None:
    """Walk the entire happy path: register → create RAG → upload
    PDF (real ``fitz`` PDF) → poll until READY → chat → assert
    citation includes the uploaded document's id."""
    headers = await auth_headers_factory("e2e@example.com")
    body = (
        '{"answer": "forty two", "summary": "the meaning of life"}\n' * 8
    ).encode("utf-8")

    # 1. Create RAG
    rag_resp = await client_with_db.post(
        "/api/v1/rags", headers=headers,
        json={"name": "e2e", "description": ""},
    )
    assert rag_resp.status_code == 201, rag_resp.text
    rag_id = rag_resp.json()["id"]

    # 2. Upload PDF
    pdf_bytes = _make_pdf_bytes(body.decode("utf-8"))
    upload = await client_with_db.post(
        f"/api/v1/rags/{rag_id}/documents",
        headers=headers,
        files={"file": ("guide.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document_id = upload.json()["id"]

    # 3. Poll until READY (worker is eager in tests but the task
    # body still needs to run).
    for _ in range(20):
        check = await client_with_db.get(
            f"/api/v1/documents/{document_id}", headers=headers
        )
        assert check.status_code == 200
        if check.json()["status"] == "READY":
            break
        await asyncio.sleep(0.1)
    final = check.json()
    assert final["status"] == "READY", (
        f"document never became READY: {final!r}"
    )
    assert final["page_count"] == 1

    # 4. Chat
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rag_id}/chat",
        headers=headers, json={"message": "what is the answer?"},
    ) as resp:
        assert resp.status_code == 200
        events = await _events(resp)
    types = [e["type"] for e in events]
    assert "citation" in types
    assert "token" in types
    assert "done" in types

    # 5. Citation contains the uploaded document_id
    citation = next(e for e in events if e["type"] == "citation")
    assert citation["citations"], "expected at least one citation"
    c = citation["citations"][0]
    assert c["document_id"] == document_id
    assert c["filename"] == "guide.pdf"
    assert c["page_number"] == 1
    assert c["retrieval_score"] >= 0.0

    done = next(e for e in events if e["type"] == "done")
    assert done["message_id"]
    assert done["conversation_id"]


# ----- E.1.2: TXT upload + chat ---------------------------------


async def test_e2e_txt_upload_then_chat(
    client_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("e2e-txt@example.com")
    body = b"the answer is 42\n" * 8

    rag_resp = await client_with_db.post(
        "/api/v1/rags", headers=headers,
        json={"name": "e2e-txt", "description": ""},
    )
    rag_id = rag_resp.json()["id"]

    upload = await client_with_db.post(
        f"/api/v1/rags/{rag_id}/documents",
        headers=headers,
        files={"file": ("x.txt", io.BytesIO(body), "text/plain")},
    )
    document_id = upload.json()["id"]

    for _ in range(20):
        check = await client_with_db.get(
            f"/api/v1/documents/{document_id}", headers=headers
        )
        if check.json()["status"] == "READY":
            break
        await asyncio.sleep(0.1)
    assert check.json()["status"] == "READY"

    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rag_id}/chat",
        headers=headers, json={"message": "what is the answer?"},
    ) as resp:
        events = await _events(resp)
    cit = next(e for e in events if e["type"] == "citation")
    assert cit["citations"][0]["document_id"] == document_id
    assert cit["citations"][0]["filename"] == "x.txt"


# ----- E.1.3: multi-turn conversation continuity ------------------


async def test_e2e_full_flow_with_conversation_continuity(
    client_with_db, auth_headers_factory
) -> None:
    """Two turns in the same conversation — the second turn
    references the first and the citation list stays valid."""
    headers = await auth_headers_factory("e2e-multi@example.com")
    body = b"alpha bravo charlie delta echo\n" * 6

    rag_resp = await client_with_db.post(
        "/api/v1/rags", headers=headers,
        json={"name": "e2e-multi", "description": ""},
    )
    rag_id = rag_resp.json()["id"]

    upload = await client_with_db.post(
        f"/api/v1/rags/{rag_id}/documents",
        headers=headers,
        files={"file": ("m.txt", io.BytesIO(body), "text/plain")},
    )
    document_id = upload.json()["id"]

    for _ in range(20):
        check = await client_with_db.get(
            f"/api/v1/documents/{document_id}", headers=headers
        )
        if check.json()["status"] == "READY":
            break
        await asyncio.sleep(0.1)
    assert check.json()["status"] == "READY"

    # First turn
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rag_id}/chat",
        headers=headers, json={"message": "first question"},
    ) as resp:
        events_1 = await _events(resp)
    conv_id = next(e for e in events_1 if e["type"] == "done")["conversation_id"]

    # Second turn
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rag_id}/chat",
        headers=headers,
        json={"message": "second question", "conversation_id": conv_id},
    ) as resp:
        events_2 = await _events(resp)
    done_2 = next(e for e in events_2 if e["type"] == "done")
    assert done_2["conversation_id"] == conv_id

    # History endpoint
    conv = await client_with_db.get(
        f"/api/v1/conversations/{conv_id}", headers=headers
    )
    assert conv.status_code == 200
    msgs = conv.json()["messages"]
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    assert msgs[3]["role"] == "assistant"
    # Both assistant messages carry the same citation shape.
    for m in msgs:
        if m["role"] == "assistant":
            assert isinstance(m["citations"], list)
            assert m["citations"]
            assert m["citations"][0]["document_id"] == document_id
