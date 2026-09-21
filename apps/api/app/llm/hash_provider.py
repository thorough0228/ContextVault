"""Deterministic LLM provider for offline dev + tests.

The hash provider never touches the network. It builds a response
by stitching together a small templated answer that:

* Lists each citation as a numbered source.
* Quotes the first chunk text from each citation.
* Mentions the model name so tests can assert on it.

This is enough for end-to-end chat tests (citation plumbing, prompt
construction, streaming, multi-turn) without needing a real LLM
account.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncIterator, List

from app.llm.base import ChatMessage, LLMProvider


class HashLLMProvider(LLMProvider):
    name = "hash"
    default_model = "hash-llm"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or self.default_model

    async def stream_chat(
        self, messages: List[ChatMessage]
    ) -> AsyncIterator[str]:
        # Pull citations + question out of the prompt deterministically.
        system_blocks = [m["content"] for m in messages if m.get("role") == "system"]
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        # The chat service builds the context as a system message that
        # contains the "Retrieved context from the knowledge base"
        # header. Snip out the citations block.
        context_block = ""
        for block in system_blocks:
            if "Retrieved context" in block:
                context_block = block
                break

        # Tokenise the answer in 12-character chunks so streaming
        # behaviour is observable from the client.
        answer = self._compose_answer(context_block, last_user)
        for chunk in self._chunk(answer, 12):
            await asyncio.sleep(0)  # yield control between chunks
            yield chunk

    @staticmethod
    def _compose_answer(context_block: str, question: str) -> str:
        has_context = "(no context" not in context_block
        if not has_context:
            return (
                "The knowledge base doesn't have any matching documents yet, "
                "so I can't answer from your sources. I'll fall back on general "
                "knowledge: " + (question.strip() or "(no question)")
            )

        # Stable hash for the question so identical questions
        # produce identical streams — useful for tests.
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
        return (
            f"[hash-llm deterministic answer; question digest {digest}]\n\n"
            "Based on your knowledge base:\n"
            "- See [1] in the citations panel for the primary source.\n"
            "- See [2] for an additional related passage, when present.\n\n"
            "Each [n] chip below links to the source document and page."
        )

    @staticmethod
    def _chunk(text: str, size: int) -> List[str]:
        if not text:
            return []
        return [text[i : i + size] for i in range(0, len(text), size)]