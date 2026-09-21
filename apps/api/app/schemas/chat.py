"""Chat schemas — request body, streaming events, history shapes.

Streaming protocol
==================

The chat endpoint returns ``application/x-ndjson``. Each line is a
JSON object with a ``type`` field. The client reads lines and
reacts:

* ``{"type": "citation", "citations": [...]}`` — sent once, before
  any tokens. The array is the list of :class:`Citation` shapes
  the LLM will see in its context.
* ``{"type": "token", "delta": "..."}`` — incremental assistant
  text. Many of these over the lifetime of a turn.
* ``{"type": "done", "message_id": "...", "conversation_id": "..."}``
  — terminal. After this the client should consider the turn
  complete; further writes are not expected.
* ``{"type": "error", "code": "...", "message": "..."}`` —
  terminal. Followed by the response stream closing.

SSE (text/event-stream) was avoided because EventSource cannot
attach a bearer token — the chat endpoint requires JWT auth, so
``fetch + ReadableStream`` is the smallest workable contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """One source backing the assistant answer.

    Mirrors the data we already return from /search: enough to let
    the UI render a chip ("page 3 of foo.pdf") and link to the
    source document."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_text: str = Field(description="Text the LLM was given for this hit")
    retrieval_score: float = Field(
        description="Cosine similarity in [0, 1] used to rank this hit"
    )
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """POST /api/v1/rags/{rag_id}/chat payload."""

    message: str = Field(
        min_length=1,
        max_length=4096,
        description="User's question. Empty or whitespace-only → 422.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing conversation to continue. If absent (or belongs to "
            "another user/rag), a new conversation is created."
        ),
    )
    top_k: int = Field(
        default=0,
        ge=0,
        le=20,
        description="How many chunks to retrieve. 0 → server default.",
    )


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class ChatEventCitation(BaseModel):
    type: Literal["citation"] = "citation"
    citations: List[Citation]


class ChatEventToken(BaseModel):
    type: Literal["token"] = "token"
    delta: str


class ChatEventDone(BaseModel):
    type: Literal["done"] = "done"
    message_id: str
    conversation_id: str
    citations: List[Citation] = Field(default_factory=list)


class ChatEventError(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class MessagePublic(BaseModel):
    """Single message in a conversation's history."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    citations: Optional[List[Citation]] = None
    created_at: datetime


class ConversationPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    rag_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationPublic):
    """Conversation with its message history, oldest first."""

    messages: List[MessagePublic] = Field(default_factory=list)