"""Deterministic, dependency-free hash-based embedding provider.

Used as the default in tests + offline development. The output is
deterministic — the same text always maps to the same vector — so
tests can assert on similarity scores without an external service.

It is **not** a real embedding model; do not use in production.
"""

from __future__ import annotations

import hashlib
import math
from typing import List

from app.embedding.base import (
    EmbeddingPermanentError,
    EmbeddingProvider,
    EmbeddingTransientError,
)


class HashEmbeddingProvider(EmbeddingProvider):
    """Pseudo-embedding built from SHA-256 + L2 normalisation.

    Algorithm:
    1. Hash the input text with SHA-256.
    2. Stretch the digest into ``dimension`` floats by iterating the
       hash (SHA-256(digest || counter)).
    3. L2-normalise so cosine similarity = dot product.

    Two near-identical inputs do NOT produce near vectors — that's
    a property of real models we don't try to fake. Phase 4 tests
    rely on the deterministic + normalised properties, not on
    semantic similarity between texts.
    """

    def __init__(self, *, dimension: int = 256, model: str = "hash-256") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self._dimension = dimension
        self._model = model

    @property
    def name(self) -> str:
        return "hash"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise EmbeddingPermanentError("text must be a string")
        if not text or not text.strip():
            # Empty input is a permanent error — embeddings on "" are
            # undefined and the worker shouldn't retry.
            raise EmbeddingPermanentError("cannot embed empty text")

        # Stretch SHA-256 digest to N dimensions.
        out = [0.0] * self._dimension
        i = 0
        counter = 0
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        while i < self._dimension:
            block = hashlib.sha256(
                seed + counter.to_bytes(4, "big", signed=False)
            ).digest()
            # 32 bytes per block, treat each byte as 0..255 then map
            # to [-1, 1] floats.
            for b in block:
                if i >= self._dimension:
                    break
                out[i] = ((b / 255.0) * 2.0) - 1.0
                i += 1
            counter += 1

        # L2-normalise.
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        out = [x / norm for x in out]

        if len(out) != self._dimension:
            # Defensive — should be impossible given the loop above.
            raise EmbeddingTransientError(
                f"hash provider returned {len(out)} dims, expected {self._dimension}"
            )
        return out

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]