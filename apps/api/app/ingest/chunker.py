"""Chunker.

A :class:`Chunker` turns an iterable of :class:`ParsedPage` into a flat
list of :class:`Chunk`. The default :class:`FixedWindowChunker` uses a
character window with overlap; the bounds live in settings so they
can change without code edits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol

from app.ingest.parser import ParsedPage

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk produced from one document page."""

    text: str
    document_id: str
    rag_id: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class Chunker(Protocol):
    """Chunker protocol — any object that exposes :meth:`chunk_pages`."""

    def chunk_pages(
        self,
        pages: Iterable[ParsedPage],
        *,
        document_id: str,
        rag_id: str,
    ) -> List[Chunk]:
        ...


@dataclass
class FixedWindowChunker:
    """Slice each page's text into ``size``-char windows with ``overlap``.

    Behaviour notes:

    * Whitespace is preserved as-is; callers can pre/post-process if they
      want normalised text.
    * If ``size <= 0`` we fall back to ``size=1`` to avoid infinite loops.
    * Overlap must be ``>= 0`` and ``< size``. ``>= size`` would make
      the window slide backwards and produce nothing useful, so we
      clamp it to ``size - 1``.
    * Empty / whitespace-only pages are skipped — they would produce
      zero chunks anyway.
    """

    size: int = 1500
    overlap: int = 200

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.overlap >= self.size:
            # Clamp silently — easier to debug than a runtime raise.
            self.overlap = self.size - 1

    def chunk_pages(
        self,
        pages: Iterable[ParsedPage],
        *,
        document_id: str,
        rag_id: str,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        # Global, monotonic index across the whole document — lets us
        # keep the ``(document_id, chunk_index)`` unique index in the DB.
        idx = 0
        for page in pages:
            text = page.text
            if not text or not text.strip():
                continue
            step = self.size - self.overlap
            start = 0
            n = len(text)
            while start < n:
                end = min(start + self.size, n)
                piece = text[start:end]
                if piece.strip():
                    chunks.append(
                        Chunk(
                            text=piece,
                            document_id=document_id,
                            rag_id=rag_id,
                            page_number=page.page_number,
                            chunk_index=idx,
                            metadata={
                                "char_start": start,
                                "char_end": end,
                                "chunk_size": self.size,
                                "chunk_overlap": self.overlap,
                            },
                        )
                    )
                    idx += 1
                if end == n:
                    break
                start += step
        logger.info(
            "chunker.fixed_window.ok document_id=%s chunks=%d size=%d overlap=%d",
            document_id,
            len(chunks),
            self.size,
            self.overlap,
        )
        return chunks


def chunk_pages(
    pages: Iterable[ParsedPage],
    *,
    document_id: str,
    rag_id: str,
    size: int,
    overlap: int,
) -> List[Chunk]:
    """Convenience wrapper used by the worker — single call site,
    no need to instantiate a :class:`FixedWindowChunker` ourselves."""
    return FixedWindowChunker(size=size, overlap=overlap).chunk_pages(
        pages, document_id=document_id, rag_id=rag_id
    )