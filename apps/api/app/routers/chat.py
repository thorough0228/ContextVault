"""Chat router: streaming RAG chat over NDJSON.

Endpoints:

* ``POST /api/v1/rags/{rag_id}/chat`` — streaming turn.
* ``GET  /api/v1/rags/{rag_id}/conversations`` — list.
* ``GET  /api/v1/conversations/{conversation_id}`` — full history.

The streaming endpoint returns ``application/x-ndjson`` with one
JSON event per line (see ``app.schemas.chat`` for the four event
shapes). Auth is JWT bearer; rag ownership is re-verified on every
turn so a stolen ``conversation_id`` cannot leak history across
tenants.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps.auth import get_current_user
from app.exceptions import NotFoundError
from app.models import Conversation, Message, User
from app.schemas.chat import (
    ChatRequest,
    ConversationDetail,
    ConversationPublic,
)
from app.services import chat_service
from app.services.rag_service import get_user_rag

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming chat turn
# ---------------------------------------------------------------------------


@router.post(
    "/rags/{rag_id}/chat",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    summary="Stream one RAG chat turn (NDJSON)",
)
async def chat_turn(
    rag_id: str,
    payload: ChatRequest,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Run the full retrieval + prompt + LLM stream pipeline.

    The response is ``application/x-ndjson`` — one event per line
    (citation, token, done, error). See :mod:`app.schemas.chat`.
    """

    # Validate rag ownership up-front. If the rag doesn't belong to
    # the caller, we return a 404 BEFORE opening the stream — the
    # "is the rag yours?" check must be a hard boundary, not a soft
    # mid-stream error event.
    await get_user_rag(db, user_id=current.id, rag_id=rag_id)

    async def _stream() -> AsyncIterator[str]:
        try:
            async for line in chat_service.stream_chat_turn(
                db,
                user_id=current.id,
                rag_id=rag_id,
                user_message=payload.message,
                conversation_id=payload.conversation_id,
                top_k=payload.top_k,
            ):
                yield line
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat.stream_unhandled err=%s", exc)
            # Phase 6: this line is intentionally a flat
            # ``{type,code,message}`` shape (NOT the global
            # ``{error:{code,message,details}}`` HTTP envelope). The
            # chat endpoint serves ``application/x-ndjson`` with one
            # JSON object per line, and ``ChatEventError`` (see
            # ``app/schemas/chat.py:107-110``) is the canonical stream
            # error shape. Clients that parse chat output MUST read
            # the ``type`` discriminator, not the HTTP envelope.
            yield (
                '{"type": "error", "code": "internal_error", '
                '"message": "chat failed unexpectedly"}\n'
            )

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Conversation listing + history
# ---------------------------------------------------------------------------


@router.get(
    "/rags/{rag_id}/conversations",
    response_model=list[ConversationPublic],
    status_code=status.HTTP_200_OK,
)
async def list_conversations(
    rag_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ConversationPublic]:
    # 404 if the rag doesn't belong to the caller.
    await get_user_rag(db, user_id=current.id, rag_id=rag_id)
    rows = await db.scalars(
        select(Conversation)
        .where(
            Conversation.user_id == current.id,
            Conversation.rag_id == rag_id,
        )
        .order_by(Conversation.updated_at.desc())
        .limit(100)
    )
    return [
        ConversationPublic(
            id=c.id,
            user_id=c.user_id,
            rag_id=c.rag_id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in rows.all()
    ]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    status_code=status.HTTP_200_OK,
)
async def get_conversation(
    conversation_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationDetail:
    conv = await db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != current.id:
        raise NotFoundError("conversation not found")

    # Phase 6: re-verify the parent Rag still belongs to the caller.
    # Today the CASCADE FK on Rag deletion makes this branch
    # unreachable (an orphaned conversation can't exist), but the
    # explicit check defends against future invariants drifting.
    await get_user_rag(db, user_id=current.id, rag_id=conv.rag_id)

    rows = await db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    msgs = list(rows.all())
    return ConversationDetail(
        id=conv.id,
        user_id=conv.user_id,
        rag_id=conv.rag_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            {
                "id": m.id,
                "conversation_id": m.conversation_id,
                "role": m.role,
                "content": m.content,
                "citations": m.citations,
                "created_at": m.created_at,
            }
            for m in msgs
        ],
    )