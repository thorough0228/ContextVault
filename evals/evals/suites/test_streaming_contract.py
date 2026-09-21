"""Suite ③ — streaming chat contract (offline, programmatic).

Two halves:

* **Pipeline contract** — a ScriptedLLMProvider is injected through the
  app's own ``set_llm_for_tests`` seam, then the full HTTP chat
  endpoint is exercised: event order (citation strictly before the
  first token), terminal event presence, citation persistence vs. the
  citation event, error paths (assistant message must NOT persist),
  and tenacity retry behaviour on transient LLM failures.
* **Think-filter** — the ``<think>`` stripping lives INSIDE
  ``OpenAILLMProvider`` (not in the chat service), so it is evaluated
  there directly: an httpx MockTransport streams adversarial SSE delta
  splits (tags broken mid-delta, unterminated blocks) and the joined
  output must never leak reasoning text.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, List

import httpx
import pytest

from evals.harness.environment import SqliteEvalEnv, chat_turn

pytestmark = pytest.mark.offline


class ScriptedLLMProvider:
    """Deterministic provider yielding fixed deltas, or raising.

    Subclasses the protocol implicitly; only ``stream_chat`` is needed
    because the chat service never calls ``chat`` on it.
    """

    name = "scripted"
    model = "scripted-llm"

    def __init__(self, deltas: List[str] | None = None, error: Exception | None = None):
        self.deltas = deltas if deltas is not None else ["Hello ", "world"]
        self.error = error
        self.calls = 0

    async def stream_chat(self, messages) -> AsyncIterator[str]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        for delta in self.deltas:
            if delta:
                yield delta


async def _make_env():
    from app.llm import set_provider_for_tests as set_llm_for_tests

    env = SqliteEvalEnv(
        corpora=["injection"],  # tiny handcrafted corpus, instant
        embedding="hash",
        llm="hash",
        corpus_size=8,
        seed=1,
        ingest_timeout_s=60.0,
    )
    await env.__aenter__()
    return env, set_llm_for_tests


class TestHappyPathContract:
    async def test_event_order_and_persistence(self):
        env, set_llm = await _make_env()
        try:
            scripted = ScriptedLLMProvider(["答案", "第一", "部分"])
            set_llm(scripted)
            rag_id = env.rag_ids["injection"]

            turn = await chat_turn(
                env.client, env.auth_headers,
                rag_id=rag_id, message="反胶和长胶有什么区别？",
            )

            # Event order: citation first, token stream, done terminal.
            types = [e["type"] for e in turn["events"]]
            assert turn["terminal"] == "done"
            assert types[0] == "citation"
            assert types[-1] == "done"
            assert "token" in types
            assert types.index("token") == 1, (
                "tokens must directly follow the citation event"
            )
            assert turn["answer"] == "答案第一部分"

            # Citations persisted on the assistant message equal the
            # emitted citation event payload.
            done = turn["events"][-1]
            conv = await env.client.get(
                f"/api/v1/conversations/{done['conversation_id']}",
                headers=env.auth_headers,
            )
            assert conv.status_code == 200
            messages = conv.json()["messages"]
            assert [m["role"] for m in messages] == ["user", "assistant"]
            persisted = messages[1]["citations"] or []
            assert len(persisted) == len(turn["citations"])
        finally:
            await env.__aexit__(None, None, None)

    async def test_citation_precedes_any_token(self):
        env, set_llm = await _make_env()
        try:
            set_llm(ScriptedLLMProvider(["x"]))
            turn = await chat_turn(
                env.client, env.auth_headers,
                rag_id=env.rag_ids["injection"], message="海绵硬度怎么选？",
            )
            for event in turn["events"]:
                if event["type"] == "token":
                    break
            assert turn["citations"], "retrieval must run before the LLM stream"
        finally:
            await env.__aexit__(None, None, None)


class TestErrorContract:
    async def test_permanent_error_emits_error_and_does_not_persist(self):
        from app.llm.base import LLMPermanentError

        env, set_llm = await _make_env()
        try:
            scripted = ScriptedLLMProvider(error=LLMPermanentError("upstream 401"))
            set_llm(scripted)
            rag_id = env.rag_ids["injection"]

            turn = await chat_turn(
                env.client, env.auth_headers, rag_id=rag_id, message="底板材质有哪些？",
            )
            assert turn["terminal"] == "error"
            assert turn["error"]["code"] == "llm_failed"
            assert scripted.calls == 1, "permanent errors must not retry"

            # The failed turn must not persist an assistant message.
            listing = await env.client.get(
                f"/api/v1/rags/{rag_id}/conversations", headers=env.auth_headers
            )
            # The endpoint returns a bare JSON array (list[ConversationPublic]).
            conv_id = listing.json()[0]["id"]
            conv = await env.client.get(
                f"/api/v1/conversations/{conv_id}", headers=env.auth_headers
            )
            roles = [m["role"] for m in conv.json()["messages"]]
            assert roles == ["user"], (
                f"failed turn must not persist an assistant message, got {roles}"
            )
        finally:
            await env.__aexit__(None, None, None)

    async def test_transient_error_retries_then_errors(self):
        from app.llm.base import LLMTransientError

        env, set_llm = await _make_env()
        try:
            scripted = ScriptedLLMProvider(error=LLMTransientError("upstream 503"))
            set_llm(scripted)

            turn = await chat_turn(
                env.client, env.auth_headers,
                rag_id=env.rag_ids["injection"], message="球拍怎么保养？",
            )
            assert turn["terminal"] == "error"
            assert turn["error"]["code"] == "llm_failed"
            assert scripted.calls == 3, (
                f"tenacity must retry transient errors 3 times, got {scripted.calls}"
            )
        finally:
            await env.__aexit__(None, None, None)

    async def test_empty_deltas_still_complete(self):
        env, set_llm = await _make_env()
        try:
            set_llm(ScriptedLLMProvider(["", "ok", ""]))
            turn = await chat_turn(
                env.client, env.auth_headers,
                rag_id=env.rag_ids["injection"], message="保养",
            )
            assert turn["terminal"] == "done"
            assert turn["answer"] == "ok"
        finally:
            await env.__aexit__(None, None, None)


class TestThinkFilter:
    """Adversarial <think> splits against the real OpenAILLMProvider.

    The filter is provider-internal (MiniMax-M3 streams reasoning
    inline), so the contract is asserted at the provider boundary with
    a mocked SSE transport — no network, exact delta control.
    """

    @staticmethod
    def _provider_with_sse(deltas: List[str]) -> "OpenAILLMProvider":
        from app.llm.openai_provider import OpenAILLMProvider

        lines = []
        for d in deltas:
            body = json.dumps({"choices": [{"delta": {"content": d}}]})
            lines.append(f"data: {body}\n\n")
        lines.append("data: [DONE]\n\n")
        payload = "".join(lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=payload, headers={
                "content-type": "text/event-stream",
            })

        return OpenAILLMProvider(
            api_key="test-key",
            base_url="http://mock",
            model="mock-model",
            strip_think=True,
            transport=httpx.MockTransport(handler),
        )

    @staticmethod
    async def _chat_out(provider) -> str:
        parts: List[str] = []
        async for delta in provider.stream_chat([{"role": "user", "content": "q"}]):
            parts.append(delta)
        return "".join(parts)

    async def test_think_block_split_across_deltas(self):
        provider = self._provider_with_sse(
            ["<thi", "nk>hidden reasoning", "</th", "ink>", "visible answer"]
        )
        out = await self._chat_out(provider)
        assert out == "visible answer"

    async def test_text_before_and_after_think(self):
        provider = self._provider_with_sse(
            ["before ", "<think>reasoning</think>", " after"]
        )
        out = await self._chat_out(provider)
        # _trim_lead strips whitespace after a think block by design so
        # the answer starts immediately — " " + " after" collapses.
        assert out == "before after"

    async def test_unterminated_think_leaks_nothing(self):
        provider = self._provider_with_sse(
            ["answer part ", "<think>never closed reasoning..."]
        )
        out = await self._chat_out(provider)
        assert out == "answer part "
        assert "reasoning" not in out

    async def test_multiple_think_blocks(self):
        provider = self._provider_with_sse(
            ["<think>r1</think>", "A", "<think>r2</think>", "B"]
        )
        out = await self._chat_out(provider)
        assert out == "AB"

    async def test_no_think_passthrough(self):
        provider = self._provider_with_sse(["plain ", "stream"])
        out = await self._chat_out(provider)
        assert out == "plain stream"
