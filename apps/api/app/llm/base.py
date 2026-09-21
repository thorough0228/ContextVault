"""LLM provider base.

Defines the :class:`LLMProvider` Protocol, the error taxonomy, and
the chat-message shape. The chat service uses
``stream_chat(messages)`` for the live endpoint; ``chat`` is a
non-streaming convenience for batched jobs.
"""

from __future__ import annotations

from typing import AsyncIterator, List, Protocol


class LLMPermanentError(Exception):
    """Permanent upstream failure (4xx-equivalent). The chat stream
    should surface this to the user; retrying the same input won't
    help."""


class LLMTransientError(Exception):
    """Transient upstream failure (5xx / timeout / rate-limit). The
    caller may retry."""


class ChatMessage(dict):
    """OpenAI-compatible message. Subclassing ``dict`` keeps the
    Provider implementation free of Pydantic imports and round-trips
    cleanly through ``json.dumps``."""


class LLMProvider(Protocol):
    """Provider contract — both ``chat`` and ``stream_chat`` must be
    implemented by every concrete provider.

    ``messages`` is a list of dicts with at least ``role`` and
    ``content`` keys. ``role`` ∈ ``{"system", "user", "assistant"}``.
    """

    name: str
    model: str

    def stream_chat(
        self, messages: List[ChatMessage]
    ) -> AsyncIterator[str]: ...

    async def chat(
        self, messages: List[ChatMessage]
    ) -> str:
        """Non-streaming helper — joins all ``stream_chat`` deltas."""
        parts: List[str] = []
        async for delta in self.stream_chat(messages):
            parts.append(delta)
        return "".join(parts)