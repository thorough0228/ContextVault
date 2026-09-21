"""In-process local embedding provider (sentence-transformers).

Runs the model inside the API / worker process — no network round
trip, no API key, no external service. Recommended model:
``BAAI/bge-small-zh-v1.5`` (512 dims, strong Chinese semantic
retrieval, CPU-friendly).

Call-shape convention shared with the other providers:

* :meth:`embed_texts` — the worker's indexing path. BGE-family models
  index passages WITHOUT any instruction prefix.
* :meth:`embed_text` — the search query path. BGE-family retrieves
  better when the query carries the model's instruction prefix
  (``EMBEDDING_LOCAL_QUERY_INSTRUCTION``).

The dependency is optional: ``sentence_transformers`` is imported
lazily so the app installs and boots without it; choosing
``EMBEDDING_PROVIDER=local`` without it fails with the install hint.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.embedding.base import (
    EmbeddingPermanentError,
    EmbeddingProvider,
)

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        query_instruction: str = "",
        fp16: bool = False,
        model: Optional[object] = None,  # test seam — inject a stub model
    ) -> None:
        if dimension <= 0:
            raise EmbeddingPermanentError(
                f"EMBEDDING_DIMENSION must be positive, got {dimension}"
            )
        self._model_name = model_name
        self._dimension = dimension
        self._query_instruction = query_instruction
        self._model = model
        self._fp16 = fp16

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingPermanentError(
                    "EMBEDDING_PROVIDER=local requires sentence-transformers: "
                    "pip install sentence-transformers"
                ) from exc
            try:
                self._model = SentenceTransformer(model_name)
            except Exception as exc:
                raise EmbeddingPermanentError(
                    f"failed to load local embedding model {model_name!r}: {exc}"
                ) from exc

        if self._fp16:
            self._maybe_half()

        actual = self._model.get_sentence_embedding_dimension()
        if actual != self._dimension:
            raise EmbeddingPermanentError(
                f"model {model_name!r} outputs {actual} dims but "
                f"EMBEDDING_DIMENSION is {self._dimension}"
            )
        logger.info(
            "local.embedding.ready model=%s dimension=%d fp16=%s",
            model_name, self._dimension, self._fp16,
        )

    def _maybe_half(self) -> None:
        """Cast model weights to fp16 on CUDA GPUs.

        Halves VRAM and roughly doubles throughput on fp16-capable
        cards (Turing+). Silently skipped when the model is on CPU
        (fp16 CPU compute is slower) or on GPUs without fp16 support
        (compute capability < 7.0) — fp32 correctness is unaffected
        either way; the provider converts outputs back to fp32.
        """

        try:
            import torch
        except ImportError:
            return
        device = getattr(self._model, "device", None)
        device_type = getattr(device, "type", None) or str(device or "")
        if device_type != "cuda":
            logger.info(
                "local.embedding.fp16_skipped reason=not_cuda device=%s",
                device_type,
            )
            return
        major, _minor = torch.cuda.get_device_capability()
        if major < 7:
            logger.info(
                "local.embedding.fp16_skipped reason=gpu_capability major=%s",
                major,
            )
            return
        self._model.half()
        logger.info("local.embedding.fp16_ok")

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, sentences: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            sentences,
            normalize_embeddings=True,  # cosine distance expects unit vectors
            convert_to_numpy=True,
        )
        # fp16 models emit float16 arrays — normalise back to fp32 so
        # every downstream consumer (pgvector binding, JSON vectors,
        # numpy cosine) sees ordinary floats.
        return [[float(x) for x in row] for row in embeddings]

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingPermanentError("cannot embed empty text")
        sentence = f"{self._query_instruction}{text}" if self._query_instruction else text
        return self._encode([sentence])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode(list(texts))
