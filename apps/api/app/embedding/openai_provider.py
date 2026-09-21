"""OpenAI-compatible embedding provider (Phase 4 stub).

This is a thin wrapper over the ``/v1/embeddings`` endpoint. The
factory only wires it up when ``EMBEDDING_PROVIDER=openai`` and a
non-empty ``EMBEDDING_OPENAI_API_KEY`` is present. The provider
itself is functional but ships without a hard dependency on
``openai`` — it uses ``httpx`` (already in requirements) so the
deployment surface stays small.

To use a local OpenAI-compatible server (e.g. Ollama, LM Studio,
vLLM) point ``EMBEDDING_OPENAI_BASE_URL`` at it.

Rate limiting (Phase 12): a sliding-window limiter over BOTH requests
per minute (RPM) and estimated tokens per minute (TPM) keeps bulk
ingestion just under the provider's published quotas — e.g.
SiliconFlow's 2,000 RPM / 500,000 TPM — without the fixed-sleep
bottleneck. Set ``EMBEDDING_OPENAI_RPM_LIMIT`` / ``_TPM_LIMIT`` to the
provider's quotas; ``0`` disables the limiter.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import List, Optional

import httpx

from app.embedding.base import (
    EmbeddingPermanentError,
    EmbeddingProvider,
    EmbeddingTransientError,
)

logger = logging.getLogger(__name__)


class _RpmTpmLimiter:
    """Sliding 60s-window limiter over (requests, estimated tokens).

    ``acquire`` blocks until one more request of ``est_tokens`` tokens
    fits inside the configured RPM/TPM budgets, then records it. Both
    budgets are optional (``0`` = unlimited on that axis).
    """

    WINDOW = 60.0

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm = max(0, rpm)
        self.tpm = max(0, tpm)
        self._events: deque[tuple[float, int]] = deque()

    @staticmethod
    def estimate_tokens(texts: List[str]) -> int:
        """Conservative token estimate: ~0.5 tokens/char (mixed CJK +
        ASCII — jieba/BGE tokenizers land near there) plus a small
        per-item framing cost. Overestimating is safe: it only makes
        us wait a little longer, never exceeds the quota.
        """

        return sum(math.ceil((len(t) + 8) / 2) for t in texts)

    def _drop_expired(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.WINDOW:
            self._events.popleft()

    def _sleep_time(self, now: float, est: int) -> float:
        used_requests = len(self._events)
        used_tokens = sum(t for _, t in self._events)
        sleep = 0.0
        if self.rpm and used_requests + 1 > self.rpm:
            sleep = max(sleep, self.WINDOW - (now - self._events[0][0]))
        if self.tpm and used_tokens + est > self.tpm:
            sleep = max(sleep, self.WINDOW - (now - self._events[0][0]))
        return sleep

    def acquire(self, est_tokens: int) -> None:
        if self.rpm <= 0 and self.tpm <= 0:
            return
        while True:
            now = time.monotonic()
            self._drop_expired(now)
            wait = self._sleep_time(now, est_tokens)
            if wait <= 0:
                break
            logger.debug("ratelimit.sleep %.2fs", wait)
            time.sleep(min(wait, self.WINDOW))
        self._events.append((time.monotonic(), est_tokens))


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int,
        timeout_seconds: float = 30.0,
        dimensions: Optional[int] = None,
        request_interval_seconds: float = 0.0,
        rpm_limit: int = 0,
        tpm_limit: int = 0,
    ) -> None:
        if not api_key:
            raise EmbeddingPermanentError(
                "EMBEDDING_OPENAI_API_KEY is empty — refusing to start"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._dimensions = dimensions
        self._timeout = timeout_seconds
        self._request_interval = max(0.0, request_interval_seconds)
        self._last_request_monotonic: Optional[float] = None
        self._limiter = _RpmTpmLimiter(rpm_limit, tpm_limit)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _throttle(self, payload_texts: Optional[List[str]] = None) -> None:
        """Rate-limit before one request, preferring the RPM/TPM budgets.

        * RPM/TPM configured → sliding-window token-bucket limiter that
          saturates the published quotas without exceeding them (bulk
          ingestion runs ~10× faster than a fixed sleep would allow).
        * Otherwise → legacy fixed ``request_interval`` spacing, kept
          for simple deployments.
        """

        if payload_texts is not None and (self._limiter.rpm or self._limiter.tpm):
            self._limiter.acquire(
                _RpmTpmLimiter.estimate_tokens(payload_texts)
            )
            return
        if self._request_interval <= 0:
            return
        now = time.monotonic()
        if self._last_request_monotonic is not None:
            elapsed = now - self._last_request_monotonic
            wait = self._request_interval - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_monotonic = time.monotonic()

    def _post(self, payload: dict, payload_texts: Optional[List[str]] = None) -> dict:
        self._throttle(payload_texts)
        url = f"{self._base_url}/embeddings"
        try:
            resp = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise EmbeddingTransientError(f"network/timeout: {exc}") from exc

        if resp.status_code >= 500:
            raise EmbeddingTransientError(
                f"upstream {resp.status_code}: {resp.text[:200]}"
            )
        if resp.status_code in (401, 403):
            raise EmbeddingPermanentError(
                f"auth failed ({resp.status_code}) — check API key"
            )
        if resp.status_code >= 400:
            raise EmbeddingPermanentError(
                f"upstream {resp.status_code}: {resp.text[:200]}"
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise EmbeddingTransientError(f"bad json from upstream: {exc}") from exc

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingPermanentError("cannot embed empty text")
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"model": self._model, "input": texts}
        # Optional dimension override — providers that support it
        # (OpenAI text-embedding-3, Zhipu embedding-3) shorten/lengthen
        # the output to match the configured column dimension.
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        body = self._post(payload, payload_texts=texts)
        items = body.get("data") or []
        if len(items) != len(texts):
            raise EmbeddingTransientError(
                f"upstream returned {len(items)} embeddings for {len(texts)} inputs"
            )
        vectors: List[List[float]] = []
        for idx, item in enumerate(items):
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise EmbeddingTransientError(
                    f"upstream item {idx} missing 'embedding' list"
                )
            if len(vec) != self._dimension:
                # The configured dimension does not match upstream.
                # This is a configuration bug — fail permanently.
                raise EmbeddingPermanentError(
                    f"upstream dim {len(vec)} != configured {self._dimension}; "
                    "set EMBEDDING_DIMENSION to match the model output"
                )
            vectors.append([float(x) for x in vec])
        return vectors