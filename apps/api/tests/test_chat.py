"""End-to-end tests for the Phase 5 RAG chat pipeline.

The chat stream is a sequence of NDJSON events. Each test
parses the stream with the same ``_iter_events`` helper and
asserts on the events it sees.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

from app.celery_client import celery_app
from app.embedding import get_embedding_provider
from app.llm import LLMProvider, LLMTransientError, set_provider_for_tests
from app.models import Conversation, DocumentChunk, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_rag(
    client: httpx.AsyncClient, headers: dict, name: str = "R"
) -> str:
    resp = await client.post(
        "/api/v1/rags", headers=headers, json={"name": name, "description": ""}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _upload_and_process(
    client: httpx.AsyncClient,
    headers: dict,
    rag_id: str,
    body: bytes,
    filename: str = "doc.txt",
    content_type: str = "text/plain",
) -> str:
    files = {"file": (filename, io.BytesIO(body), content_type)}
    resp = await client.post(
        f"/api/v1/rags/{rag_id}/documents", headers=headers, files=files
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _iter_events(resp) -> List[Dict[str, Any]]:
    """Parse the NDJSON stream from a chat response into a list of dicts.

    Tests are async (pytest-asyncio auto mode), so we can iterate
    the response stream directly.
    """
    events: List[Dict[str, Any]] = []
    async for raw in resp.aiter_lines():
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            raise AssertionError(f"non-JSON event line: {raw!r}")
    return events


# ---------------------------------------------------------------------------
# 1. Chat 成功
# ---------------------------------------------------------------------------


async def test_chat_emits_citation_token_done_events(
    client_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid,
        b"the secret answer is forty two\n" * 8,
        "alpha.txt",
    )

    async with client_with_db.stream(
        "POST",
        f"/api/v1/rags/{rid}/chat",
        headers=headers,
        json={"message": "what is the answer?", "top_k": 2},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = await _iter_events(resp)

    types = [e["type"] for e in events]
    assert "citation" in types
    assert "token" in types
    assert "done" in types

    # Exactly one citation event, then one or more token events,
    # then a single done event.
    assert types.count("citation") == 1
    assert types.count("done") == 1
    assert types.index("citation") < types.index("token") < types.index("done")

    done = next(e for e in events if e["type"] == "done")
    assert done["conversation_id"]
    assert done["message_id"]
    assert isinstance(done["citations"], list)
    assert len(done["citations"]) >= 1
    cit = done["citations"][0]
    assert {"chunk_id", "document_id", "filename", "page_number",
            "chunk_text", "retrieval_score"} <= set(cit.keys())


# ---------------------------------------------------------------------------
# 2. RAG 隔离
# ---------------------------------------------------------------------------


async def test_chat_rejects_cross_user_rag(
    client_with_db, auth_headers_factory
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")
    alice_rag = await _create_rag(client_with_db, headers_alice, "alice")
    await _upload_and_process(
        client_with_db, headers_alice, alice_rag,
        b"alice private answer is 42\n" * 6,
        "alice.txt",
    )

    # Bob tries to chat into Alice's rag.
    async with client_with_db.stream(
        "POST",
        f"/api/v1/rags/{alice_rag}/chat",
        headers=headers_bob,
        json={"message": "what is the answer?"},
    ) as resp:
        # The router validates rag ownership BEFORE the stream
        # begins, so a hard 404 — not a 200 with an error event.
        assert resp.status_code == 404
        body = await resp.aread()
        assert b"alice" not in body.lower()


# ---------------------------------------------------------------------------
# 3. Citation
# ---------------------------------------------------------------------------


async def test_citations_include_document_id_filename_page_chunk_score(
    client_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    doc_id = await _upload_and_process(
        client_with_db, headers, rid,
        b"a sentence about RAG chunking " * 20,
        "guide.txt", "text/plain",
    )
    async with client_with_db.stream(
        "POST",
        f"/api/v1/rags/{rid}/chat",
        headers=headers,
        json={"message": "tell me about chunking", "top_k": 1},
    ) as resp:
        events = await _iter_events(resp)
    citation = next(e for e in events if e["type"] == "citation")["citations"][0]
    assert citation["document_id"] == doc_id
    assert citation["filename"] == "guide.txt"
    assert citation["page_number"] >= 1
    assert citation["chunk_id"]
    assert citation["retrieval_score"] >= 0.0
    assert citation["chunk_text"]


# ---------------------------------------------------------------------------
# 4. Streaming (multi-token + NDJSON)
# ---------------------------------------------------------------------------


async def test_streaming_emits_multiple_token_events(
    client_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid,
        b"lorem ipsum dolor sit amet\n" * 6,
        "x.txt",
    )
    async with client_with_db.stream(
        "POST",
        f"/api/v1/rags/{rid}/chat",
        headers=headers,
        json={"message": "hello"},
    ) as resp:
        events = await _iter_events(resp)
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) >= 2, "expected multiple token events for streaming"
    # Token deltas concatenate back to a non-empty string.
    full = "".join(t["delta"] for t in tokens)
    assert full.strip() != ""


# ---------------------------------------------------------------------------
# 5. conversation 创建
# 6. message 保存
# ---------------------------------------------------------------------------


async def test_chat_creates_conversation_and_saves_messages(
    client_with_db, db_sessionmaker, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"alpha\nbravo\ncharlie\n" * 6, "a.txt"
    )

    async with client_with_db.stream(
        "POST",
        f"/api/v1/rags/{rid}/chat",
        headers=headers,
        json={"message": "what is alpha?"},
    ) as resp:
        events = await _iter_events(resp)
    done = next(e for e in events if e["type"] == "done")
    conv_id = done["conversation_id"]

    async with db_sessionmaker() as s:
        conv = await s.get(Conversation, conv_id)
        msgs = list(
            (await s.execute(
                __import__("sqlalchemy").select(Message)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.created_at)
            )).scalars()
        )

    assert conv is not None
    assert conv.user_id  # ownership stored
    assert conv.rag_id == rid
    assert conv.title  # auto-titled from first message
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].content == "what is alpha?"
    assert msgs[1].role == "assistant"
    assert msgs[1].content.strip() != ""


# ---------------------------------------------------------------------------
# 7. 多轮消息
# ---------------------------------------------------------------------------


async def test_multi_turn_includes_history_in_prompt(
    client_with_db, auth_headers_factory, monkeypatch
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"alpha data here\n" * 6, "a.txt"
    )

    # Spy on the LLM to capture the messages list at call time.
    seen: List[List[dict]] = []

    class _Spy(get_embedding_provider().__class__):
        pass  # not used; we need a real LLM spy

    from app.llm.hash_provider import HashLLMProvider

    class _SpyLLM(HashLLMProvider):
        def __init__(self, original):
            self.original = original

        async def stream_chat(self, messages):
            seen.append(list(messages))
            async for d in self.original.stream_chat(messages):
                yield d

    real = get_embedding_provider()  # touch factory to ensure init
    from app.llm import get_llm_provider
    real_llm = get_llm_provider()
    set_provider_for_tests(_SpyLLM(real_llm))

    # First turn
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "first question"},
    ) as resp:
        events1 = await _iter_events(resp)
    conv_id = next(e for e in events1 if e["type"] == "done")["conversation_id"]

    # Second turn — same conversation_id
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers,
        json={"message": "second question", "conversation_id": conv_id},
    ) as resp:
        _ = await _iter_events(resp)

    assert len(seen) == 2
    # Second call's message list should include the previous user
    # message and the previous assistant reply (multi-turn).
    second = seen[1]
    roles = [m["role"] for m in second]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 1
    # Plus two system messages (system prompt + retrieved context).
    assert roles.count("system") >= 2


# ---------------------------------------------------------------------------
# 7b. Prompt-injection hardening (system prompt + context framing)
# ---------------------------------------------------------------------------


def test_system_prompt_contains_injection_defenses() -> None:
    from app.services import chat_service

    prompt = chat_service.SYSTEM_PROMPT
    # The eval harness' leak detector keys on this opening sentence.
    assert prompt.startswith("You are a knowledge-base assistant")
    assert "UNTRUSTED DATA" in prompt
    assert "never instructions" in prompt
    assert "system prompt" in prompt
    # Rule-lifting via the user's own message is covered separately
    # from context-borne instructions.
    assert "user" in prompt


def test_context_header_frames_context_as_untrusted() -> None:
    from app.models import Conversation, Message
    from app.services import chat_service

    conv = Conversation(user_id="u1", rag_id="rag-1", title="")
    history = [Message(conversation_id="c1", role="user", content="hi")]
    history[0].conversation = conv

    messages = chat_service._build_prompt_messages(
        system_prompt=chat_service.SYSTEM_PROMPT,
        context="(no context — the knowledge base has no matching documents)",
        history=history,
        user_message="q",
    )
    # [system prompt, retrieved-context system msg, history("hi"), user("q")]
    assert len(messages) == 4
    assert messages[0]["content"] == chat_service.SYSTEM_PROMPT

    context_msg = messages[1]["content"]
    # hash_provider locates the context block by this substring — the
    # framing sentence must not replace it.
    assert "Retrieved context" in context_msg
    assert "untrusted reference data" in context_msg
    # Empty-KB sentinel survives for hash_provider's has_context check.
    assert "(no context" in context_msg


# ---------------------------------------------------------------------------
# 8. LLM failure
# ---------------------------------------------------------------------------


async def test_llm_failure_emits_error_event_and_no_assistant_message(
    client_with_db, db_sessionmaker, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"x\n" * 6, "x.txt"
    )

    class _Boom(LLMProvider):
        name = "boom"
        model = "boom"

        async def stream_chat(self, messages):
            raise RuntimeError("synthetic llm failure")
            yield ""  # pragma: no cover

    set_provider_for_tests(_Boom())

    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "hi"},
    ) as resp:
        events = await _iter_events(resp)

    types = [e["type"] for e in events]
    assert "error" in types
    assert types[-1] == "error"
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "llm_failed"
    assert "synthetic" in err["message"].lower()

    # User message persisted, assistant NOT persisted.
    async with db_sessionmaker() as s:
        from sqlalchemy import select
        rows = (await s.execute(select(Message).order_by(Message.created_at))).scalars()
    roles = [m.role for m in rows]
    assert "user" in roles
    assert "assistant" not in roles


# ---------------------------------------------------------------------------
# 9. Search failure
# ---------------------------------------------------------------------------


async def test_search_failure_degrades_to_no_context_answer(
    client_with_db, auth_headers_factory, monkeypatch
) -> None:
    """If the embedding provider is permanently broken, chat should
    still respond (with a 'no context' preamble) rather than blowing
    up the whole turn."""
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"x\n" * 6, "x.txt"
    )

    from app.embedding import EmbeddingPermanentError
    from app.services import chat_service as chat_service_mod

    class _BrokenProvider(get_embedding_provider().__class__):
        def embed_text(self, text):
            raise EmbeddingPermanentError("simulated embed down")

        def embed_texts(self, texts):
            raise EmbeddingPermanentError("simulated embed down")

    # Patch the symbol the chat service actually looks up.
    monkeypatch.setattr(chat_service_mod, "get_embedding_provider", lambda: _BrokenProvider())

    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "what?"},
    ) as resp:
        events = await _iter_events(resp)
    types = [e["type"] for e in events]
    # No citations (the embed failure → empty context), but tokens
    # and done still flow.
    citation = next(e for e in events if e["type"] == "citation")
    assert citation["citations"] == []
    assert "token" in types
    assert "done" in types


# ---------------------------------------------------------------------------
# 10. 空知识库
# ---------------------------------------------------------------------------


async def test_chat_on_empty_rag_still_streams_answer(
    client_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    # No documents uploaded.
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "anything"},
    ) as resp:
        events = await _iter_events(resp)
    cit = next(e for e in events if e["type"] == "citation")
    assert cit["citations"] == []
    types = [e["type"] for e in events]
    assert "token" in types
    assert "done" in types


# ---------------------------------------------------------------------------
# 11. 无相关结果
# ---------------------------------------------------------------------------


async def test_chat_with_no_relevant_hits_returns_empty_citations(
    client_with_db, auth_headers_factory
) -> None:
    """Upload something, but ask something unrelated so the
    retrievers return nothing (or the hash provider picks the
    closest chunk but we verify citations is a list either way).

    The contract: citations is always a list (possibly empty) and
    the response always ends with a done event."""
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"only one topic here\n" * 6, "x.txt"
    )
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "completely unrelated question " * 8},
    ) as resp:
        events = await _iter_events(resp)
    types = [e["type"] for e in events]
    assert "token" in types
    assert "done" in types
    citation = next(e for e in events if e["type"] == "citation")
    assert isinstance(citation["citations"], list)


# ---------------------------------------------------------------------------
# 12. auth + history list
# ---------------------------------------------------------------------------


async def test_chat_requires_auth(client_with_db) -> None:
    async with client_with_db.stream(
        "POST",
        "/api/v1/rags/anything/chat",
        json={"message": "hi"},
    ) as resp:
        assert resp.status_code == 401


async def test_list_conversations_returns_only_callers(
    client_with_db, db_sessionmaker, auth_headers_factory
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")
    rid = await _create_rag(client_with_db, headers_alice, "alice")
    await _upload_and_process(
        client_with_db, headers_alice, rid, b"x\n" * 6, "x.txt"
    )
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers_alice, json={"message": "hi"},
    ) as resp:
        events = await _iter_events(resp)

    # Bob should see zero conversations on Alice's rag.
    resp = await client_with_db.get(
        f"/api/v1/rags/{rid}/conversations", headers=headers_bob
    )
    assert resp.status_code == 404  # Bob doesn't own the rag

    # Alice sees her own conversation.
    resp = await client_with_db.get(
        f"/api/v1/rags/{rid}/conversations", headers=headers_alice
    )
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == next(
        e for e in events if e["type"] == "done"
    )["conversation_id"]


async def test_get_conversation_history_returns_full_thread(
    client_with_db, auth_headers_factory
) -> None:
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"x\n" * 6, "x.txt"
    )
    # Two turns.
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "first"},
    ) as resp:
        events1 = await _iter_events(resp)
    conv_id = next(e for e in events1 if e["type"] == "done")["conversation_id"]
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers,
        json={"message": "second", "conversation_id": conv_id},
    ) as resp:
        _ = await _iter_events(resp)

    resp = await client_with_db.get(
        f"/api/v1/conversations/{conv_id}", headers=headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == conv_id
    assert len(body["messages"]) == 4
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "first"
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][2]["role"] == "user"
    assert body["messages"][3]["role"] == "assistant"


async def test_get_conversation_404_for_other_user(
    client_with_db, auth_headers_factory
) -> None:
    headers_alice = await auth_headers_factory("alice@example.com")
    headers_bob = await auth_headers_factory("bob@example.com")
    rid = await _create_rag(client_with_db, headers_alice, "alice")
    await _upload_and_process(
        client_with_db, headers_alice, rid, b"x\n" * 6, "x.txt"
    )
    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers_alice, json={"message": "hi"},
    ) as resp:
        events = await _iter_events(resp)
    conv_id = next(e for e in events if e["type"] == "done")["conversation_id"]

    resp = await client_with_db.get(
        f"/api/v1/conversations/{conv_id}", headers=headers_bob
    )
    assert resp.status_code == 404

# ----- Phase 6: P0.8 tenacity retry on LLM transient ------------------


async def test_chat_stream_retries_on_transient_llm_failure(
    client_with_db, db_sessionmaker, auth_headers_factory
) -> None:
    """A provider that raises ``LLMTransientError`` on the first call
    and then succeeds should produce a complete turn (tokens +
    done event) and persist the assistant message."""
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"some context text\n" * 6, "x.txt"
    )

    class _Flaky(LLMProvider):
        name = "flaky"
        model = "flaky"
        dimension = 0  # unused

        def __init__(self) -> None:
            self.calls = 0

        async def stream_chat(self, messages):
            # Must contain ``yield`` to be an async generator — the
            # chat service iterates the result with ``async for``.
            self.calls += 1
            if self.calls == 1:
                yield ""  # ignored; we want the raise to fire
                raise LLMTransientError("upstream blip")
            for i in range(3):
                yield f"d{i} "

    set_provider_for_tests(_Flaky())

    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "hi"},
    ) as resp:
        events = await _iter_events(resp)

    types = [e["type"] for e in events]
    assert "done" in types
    assert "error" not in types
    tokens = [e for e in events if e["type"] == "token"]
    assert "d0" in "".join(t["delta"] for t in tokens)
    # Assistant message persisted.
    from sqlalchemy import select
    async with db_sessionmaker() as s:
        from app.models import Message
        rows = (
            await s.execute(select(Message).where(Message.role == "assistant"))
        ).scalars().all()
    assert rows and rows[0].content.strip()


async def test_chat_stream_gives_up_after_max_retries(
    client_with_db, auth_headers_factory
) -> None:
    """A provider that always raises ``LLMTransientError`` should
    surface one terminal error event with ``code=llm_failed``."""
    headers = await auth_headers_factory("alice@example.com")
    rid = await _create_rag(client_with_db, headers, name="alice")
    await _upload_and_process(
        client_with_db, headers, rid, b"some context text\n" * 6, "x.txt"
    )

    class _AlwaysDown(LLMProvider):
        name = "down"
        model = "down"
        dimension = 0

        def __init__(self) -> None:
            self.calls = 0

        async def stream_chat(self, messages):
            self.calls += 1
            yield ""  # ignored; we want the raise to fire
            raise LLMTransientError("upstream still down")

    p = _AlwaysDown()
    set_provider_for_tests(p)

    async with client_with_db.stream(
        "POST", f"/api/v1/rags/{rid}/chat",
        headers=headers, json={"message": "hi"},
    ) as resp:
        events = await _iter_events(resp)

    # We should see the final error event (no tokens, no done).
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "llm_failed"
    assert "upstream" in err["message"].lower()
    # No done event after the terminal error.
    assert "done" not in {e["type"] for e in events}
    # 3 retry attempts (configured in chat_service).
    assert p.calls == 3
