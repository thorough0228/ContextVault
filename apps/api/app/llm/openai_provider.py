"""OpenAI-compatible LLM provider.

Default endpoint is the public OpenAI ``/v1/chat/completions`` URL.
A self-hosted OpenAI-compatible gateway (vLLM, llama.cpp, Azure OpenAI
with the v1 API) can be used by changing ``base_url`` — that's the
single line of integration that the rest of the system needs.

Streaming
---------

The provider uses ``stream=true`` and reads SSE lines from the
``httpx.AsyncClient``. Each non-empty ``data:`` line carries a JSON
delta; the provider yields the content fragments as they arrive so
the chat stream can pipe them straight to the user.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, List, Optional

import httpx

from app.llm.base import (
    ChatMessage,
    LLMPermanentError,
    LLMProvider,
    LLMTransientError,
)

logger = logging.getLogger(__name__)


class _ThinkFilter:
    """Stateful stream filter that drops ``<think>...</think>`` blocks.

    Reasoning models (MiniMax-M3, DeepSeek-R1, ...) stream their chain
    of thought inline in ``content``. The filter holds back any trailing
    text that could be a partial tag, so tags split across chunk
    boundaries are still caught, and strips leading newlines after a
    think block so the answer starts immediately.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""
        self._trim_lead = False

    @staticmethod
    def _partial_len(buf: str, tag: str) -> int:
        """Longest suffix of ``buf`` that is a proper prefix of ``tag``."""
        for n in range(min(len(buf), len(tag) - 1), 0, -1):
            if buf.endswith(tag[:n]):
                return n
        return 0

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: List[str] = []
        while True:
            if self._inside:
                end = self._buf.find(self._CLOSE)
                if end == -1:
                    # Everything buffered is think content — discard it,
                    # keeping only a suffix that might start the close tag.
                    hold = self._partial_len(self._buf, self._CLOSE)
                    self._buf = self._buf[-hold:] if hold else ""
                    break
                self._buf = self._buf[end + len(self._CLOSE):]
                self._inside = False
                self._trim_lead = True
                continue
            start = self._buf.find(self._OPEN)
            if start == -1:
                hold = self._partial_len(self._buf, self._OPEN)
                if hold:
                    out.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    out.append(self._buf)
                    self._buf = ""
                break
            out.append(self._buf[:start])
            self._buf = self._buf[start + len(self._OPEN):]
            self._inside = True

        emitted = "".join(out)
        if self._trim_lead and emitted:
            emitted = emitted.lstrip()
            if emitted:
                self._trim_lead = False
        return emitted

    def flush(self) -> str:
        """Emit held-back text at stream end. An unterminated think
        block leaks nothing."""
        tail, self._buf = self._buf, ""
        if self._inside:
            return ""
        if self._trim_lead and tail:
            tail = tail.lstrip()
            self._trim_lead = False
        return tail


class OpenAILLMProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_context_tokens: int = 8000,
        strip_think: bool = True,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenAI provider requires LLM_OPENAI_API_KEY to be set"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model or self.default_model
        self.timeout_seconds = timeout_seconds
        self.max_context_tokens = max_context_tokens
        self._strip_think = strip_think
        self._transport = transport

    async def stream_chat(
        self, messages: List[ChatMessage]
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "max_tokens": self.max_context_tokens,
        }

        timeout = httpx.Timeout(self.timeout_seconds, connect=10.0)
        think_filter = _ThinkFilter() if self._strip_think else None
        async with httpx.AsyncClient(
            timeout=timeout, transport=self._transport
        ) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400 and resp.status_code < 500:
                        body = await resp.aread()
                        raise LLMPermanentError(
                            f"openai {resp.status_code}: {body.decode('utf-8', 'ignore')[:300]}"
                        )
                    if resp.status_code >= 500:
                        raise LLMTransientError(
                            f"openai {resp.status_code}"
                        )
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            chunk = line[len("data:"):].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                event = json.loads(chunk)
                            except json.JSONDecodeError:
                                continue
                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content")
                                if not content:
                                    continue
                                if think_filter is None:
                                    yield content
                                    continue
                                emitted = think_filter.feed(content)
                                if emitted:
                                    yield emitted
                if think_filter is not None:
                    tail = think_filter.flush()
                    if tail:
                        yield tail
            except httpx.TimeoutException as exc:
                raise LLMTransientError(f"openai timeout: {exc}") from exc
            except httpx.HTTPError as exc:
                raise LLMTransientError(f"openai http error: {exc}") from exc