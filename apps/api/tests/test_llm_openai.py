"""Tests for the OpenAI-compatible LLM provider (think-filter + stream)."""

from __future__ import annotations

import asyncio
import json
from typing import List

import httpx

from app.llm.openai_provider import OpenAILLMProvider


def _sse_response(deltas: List[str]) -> httpx.Response:
    lines = []
    for d in deltas:
        event = {"choices": [{"delta": {"content": d}}]}
        lines.append(f"data: {json.dumps(event)}\n\n")
    lines.append("data: [DONE]\n\n")
    return httpx.Response(200, text="".join(lines))


def _make_provider(deltas: List[str], *, strip_think: bool = True):
    transport = httpx.MockTransport(
        lambda request: _sse_response(deltas)
    )
    return OpenAILLMProvider(
        api_key="test-key",
        base_url="https://llm.test/v1",
        model="test-model",
        strip_think=strip_think,
        transport=transport,
    )


def _collect(provider) -> str:
    async def run():
        out: List[str] = []
        async for delta in provider.stream_chat(
            [{"role": "user", "content": "q"}]
        ):
            out.append(delta)
        return "".join(out)

    return asyncio.run(run())


def test_think_block_is_filtered() -> None:
    provider = _make_provider([
        "<think>long reasoning here</think>\n\nThe answer is 42.",
    ])
    assert _collect(provider) == "The answer is 42."


def test_think_tag_split_across_chunks_is_caught() -> None:
    provider = _make_provider([
        "<th",          # partial open tag
        "ink>hidden</th",
        "ink>final answer",
    ])
    assert _collect(provider) == "final answer"


def test_stream_without_think_passes_through() -> None:
    deltas = ["Hello", " world", "!"]
    assert _collect(_make_provider(deltas)) == "Hello world!"


def test_strip_think_false_keeps_reasoning() -> None:
    provider = _make_provider([
        "<think>reasoning</think>answer",
    ], strip_think=False)
    assert _collect(provider) == "<think>reasoning</think>answer"


def test_unterminated_think_block_leaks_nothing() -> None:
    provider = _make_provider(["<think>truncated reasoning without close"])
    assert _collect(provider) == ""
