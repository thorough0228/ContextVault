"""SiliconFlow /rerank implementation (BAAI/bge-reranker-v2-m3).

POST {base_url}/rerank with {"model", "query", "documents"} returns
``results: [{index, relevance_score}]`` — scores come back in
result order, so they are mapped back positionally onto the caller's
document order. Shares the embedding endpoint's key and rate-limit
behaviour (interval throttle supported for the free tier).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

import httpx

from app.rerank.base import RerankPermanentError, RerankTransientError

logger = logging.getLogger(__name__)


class SiliconFlowRerankProvider:
    name = "siliconflow"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        request_interval_seconds: float = 0.0,
    ) -> None:
        if not api_key:
            raise RerankPermanentError(
                "rerank provider requires an API key"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout_seconds
        self._interval = max(0.0, request_interval_seconds)
        self._lock = threading.Lock()
        self._last_request: Optional[float] = None

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        self._throttle()
        try:
            resp = httpx.post(
                f"{self._base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                },
                timeout=self._timeout,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RerankTransientError(f"network/timeout: {exc}") from exc

        if resp.status_code >= 500:
            raise RerankTransientError(f"rerank upstream {resp.status_code}")
        if resp.status_code >= 400:
            raise RerankPermanentError(
                f"rerank upstream {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json().get("results", [])
        if len(data) != len(documents):
            raise RerankTransientError(
                f"rerank returned {len(data)} results for {len(documents)} docs"
            )
        scores = [0.0] * len(documents)
        for item in data:
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores

    def _throttle(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if self._last_request is not None:
                wait = self._interval - (now - self._last_request)
                if wait > 0:
                    time.sleep(wait)
            self._last_request = time.monotonic()
