"""Semantic chunking (Phase 14, plan 1): embedding-based breakpoints.

Splits each page into sentences, embeds them with the configured
embedding provider (SiliconFlow — shares the RPM/TPM token bucket),
finds semantic boundaries where the cosine similarity between adjacent
sentences drops below a percentile threshold, and groups sentences at
those boundaries into chunks. Oversized groups fall back to the
recursive splitter; undersized groups merge forward.

Failure policy: any provider error is logged and the page falls back
to :class:`RecursiveChunker` — semantic chunking never fails a
document (same graceful style as search degradation).

The sentence vectors are transient intermediates only; the stored
vectors are still computed from the FINAL chunk texts by the
ingestion task.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np

from app.embedding.base import EmbeddingProvider
from app.ingest.chunker import Chunk, RecursiveChunker
from app.ingest.parser import ParsedPage
from app.ingest.recursive_splitter import sentence_spans

logger = logging.getLogger(__name__)


def _cosine_matrix(vectors: List[List[float]]) -> np.ndarray:
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat @ mat.T


class SemanticChunker:
    def __init__(
        self,
        *,
        size: int,
        overlap: int,
        percentile: float = 95.0,
        min_chars: int = 100,
        embedding_provider: Optional[EmbeddingProvider] = None,
        fallback: Optional[RecursiveChunker] = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        self.size = size
        self.overlap = overlap
        self.percentile = float(percentile)
        self.min_chars = max(1, min_chars)
        self._provider = embedding_provider
        self._fallback = fallback or RecursiveChunker(size=size, overlap=overlap)

    def _provider_lazy(self) -> EmbeddingProvider:
        if self._provider is None:
            from app.embedding import get_embedding_provider

            self._provider = get_embedding_provider()
        return self._provider

    def chunk_pages(
        self,
        pages: Iterable[ParsedPage],
        *,
        document_id: str,
        rag_id: str,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        idx = 0
        for page in pages:
            text = page.text
            if not text or not text.strip():
                continue
            try:
                page_chunks = self._chunk_page_semantic(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "chunker.semantic.degraded page=%s err=%s",
                    page.page_number, str(exc)[:120],
                )
                for c in self._fallback.chunk_pages(
                    [page], document_id=document_id, rag_id=rag_id
                ):
                    chunks.append(c)
                    idx += 1
                logger.info(
                    "chunker.semantic.fallback_chunks document_id=%s page=%s",
                    document_id, page.page_number,
                )
                continue
            for piece in page_chunks:
                chunks.append(
                    Chunk(
                        text=piece["text"],
                        document_id=document_id,
                        rag_id=rag_id,
                        page_number=page.page_number,
                        chunk_index=idx,
                        metadata={
                            "char_start": piece["char_start"],
                            "char_end": piece["char_end"],
                            "chunk_size": self.size,
                            "chunk_overlap": self.overlap,
                            "chunker": "semantic",
                        },
                    )
                )
                idx += 1
        logger.info(
            "chunker.semantic.ok document_id=%s chunks=%d", document_id, len(chunks)
        )
        return chunks

    # -- internals ---------------------------------------------------------

    def _chunk_page_semantic(self, text: str) -> List[dict]:
        spans = sentence_spans(text)
        if len(spans) < 2:
            return self._split_oversized(text, 0, len(text))

        sentences = [text[s:e] for s, e in spans]
        provider = self._provider_lazy()
        # Batched embedding — a large structured page can contain
        # thousands of sentences; one giant request would time out.
        vectors: List[List[float]] = []
        batch = 32
        for i in range(0, len(sentences), batch):
            vectors.extend(provider.embed_texts(sentences[i : i + batch]))
        sims = _cosine_matrix(vectors)

        # similarity of each sentence with its successor
        adjacent = [float(sims[i, i + 1]) for i in range(len(sentences) - 1)]
        threshold = float(np.percentile(adjacent, self.percentile))

        boundaries = {len(sentences)}
        for i, sim in enumerate(adjacent):
            if sim < threshold:
                boundaries.add(i + 1)

        groups: List[Tuple[int, int]] = []  # (first_span_idx, last_span_idx)
        g_start = 0
        for i in range(1, len(sentences) + 1):
            if i in boundaries:
                groups.append((g_start, i - 1))
                g_start = i

        # merge undersized groups forward
        merged: List[Tuple[int, int]] = []
        for g in groups:
            g_start_char = spans[g[0]][0]
            g_end_char = spans[g[1]][1]
            if merged and g_end_char - g_start_char < self.min_chars:
                prev = merged[-1]
                merged[-1] = (prev[0], g[1])
            else:
                merged.append(g)

        out: List[dict] = []
        for g_start_idx, g_end_idx in merged:
            g_start_char = spans[g_start_idx][0]
            g_end_char = spans[g_end_idx][1]
            out.extend(self._split_oversized(text, g_start_char, g_end_char))
        return out

    def _split_oversized(self, text: str, start: int, end: int) -> List[dict]:
        segment = text[start:end]
        if len(segment) <= self.size:
            return [
                {
                    "char_start": start,
                    "char_end": end,
                    "text": segment,
                }
            ]
        return [
            {
                "char_start": start + s,
                "char_end": start + e,
                "text": piece,
            }
            for s, e, piece in self._fallback_recursive(segment)
        ]

    def _fallback_recursive(self, segment: str):
        from app.ingest.recursive_splitter import recursive_split

        return recursive_split(segment, size=self.size, overlap=self.overlap)
