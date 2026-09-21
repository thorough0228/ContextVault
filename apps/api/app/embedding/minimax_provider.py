"""MiniMax embedding provider (``embo-01``).

MiniMax's ``/v1/embeddings`` is **not** OpenAI-compatible:

* the account ``GroupId`` is a required query parameter;
* the body is ``{"model", "texts", "type"}`` — ``texts`` is a string
  array and ``type`` selects the embedding space, ``"db"`` for texts
  being indexed and ``"query"`` for retrieval-time queries;
* the response is ``{"vectors": [[...]], "total_tokens", "base_resp"}``
  with ``base_resp.status_code == 0`` meaning success.

The provider maps the call shapes onto that protocol: singular
:meth:`embed_text` calls (the search query path) use ``type="query"``,
batched :meth:`embed_texts` calls (the worker's indexing path) use
``type="db"``. ``embo-01`` vectors are 1536-dimensional — keep
``EMBEDDING_DIMENSION`` in sync, mismatches fail loudly.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from app.embedding.base import (
    EmbeddingPermanentError,
    EmbeddingProvider,
    EmbeddingTransientError,
)

logger = logging.getLogger(__name__)

# base_resp.status_code values worth retrying (rate limit / timeout).
_TRANSIENT_STATUS = {1001, 1002}


class MiniMaxEmbeddingProvider(EmbeddingProvider):
    name = "minimax"
    default_model = "embo-01"

    def __init__(
        self,
        *,
        api_key: str,
        group_id: str,
        base_url: str = "https://api.minimax.chat",
        model: Optional[str] = None,
        dimension: int = 1536,
        timeout_seconds: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        if not api_key:
            raise EmbeddingPermanentError(
                "MiniMax provider requires EMBEDDING_MINIMAX_API_KEY"
            )
        if not group_id:
            raise EmbeddingPermanentError(
                "MiniMax provider requires EMBEDDING_MINIMAX_GROUP_ID"
            )
        self._api_key = api_key
        self._group_id = group_id
        self._base_url = base_url.rstrip("/")
        self._model = model or self.default_model
        self._dimension = dimension
        self._timeout = timeout_seconds
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        vectors = self._request([text], purpose="query")
        return vectors[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._request(texts, purpose="db")

    def _request(
        self, texts: List[str], *, purpose: str
    ) -> List[List[float]]:
        if not texts:
            return []

        url = f"{self._base_url}/v1/embeddings"
        params = {"GroupId": self._group_id}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model, "texts": list(texts), "type": purpose}

        timeout = httpx.Timeout(self._timeout, connect=10.0)
        with httpx.Client(timeout=timeout, transport=self._transport) as client:
            try:
                resp = client.post(
                    url, params=params, headers=headers, json=payload
                )
            except httpx.TimeoutException as exc:
                raise EmbeddingTransientError(f"minimax timeout: {exc}") from exc
            except httpx.HTTPError as exc:
                raise EmbeddingTransientError(f"minimax http error: {exc}") from exc

        if resp.status_code == 401 or resp.status_code == 403:
            raise EmbeddingPermanentError(
                f"minimax auth failed ({resp.status_code}); "
                "check EMBEDDING_MINIMAX_API_KEY / GroupId"
            )
        if resp.status_code >= 500:
            raise EmbeddingTransientError(f"minimax {resp.status_code}")
        if resp.status_code != 200:
            raise EmbeddingPermanentError(
                f"minimax {resp.status_code}: {resp.text[:300]}"
            )

        body = resp.json()
        base_resp = body.get("base_resp") or {}
        status = base_resp.get("status_code", 0)
        if status != 0:
            msg = f"minimax base_resp {status}: {base_resp.get('status_msg', '')}"
            if status in _TRANSIENT_STATUS:
                raise EmbeddingTransientError(msg)
            raise EmbeddingPermanentError(msg)

        vectors = body.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingPermanentError(
                f"minimax returned {len(vectors) if isinstance(vectors, list) else '?'} "
                f"vectors for {len(texts)} inputs"
            )

        for vec in vectors:
            if not isinstance(vec, list) or len(vec) != self._dimension:
                actual = len(vec) if isinstance(vec, list) else "?"
                raise EmbeddingPermanentError(
                    f"minimax returned {actual} dims, expected "
                    f"{self._dimension} — set EMBEDDING_DIMENSION to the "
                    "model's real output dimension"
                )
        return vectors
