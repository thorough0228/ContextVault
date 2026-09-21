"""Tests for the chunker."""

from __future__ import annotations

import pytest

from app.ingest.chunker import (
    Chunk,
    FixedWindowChunker,
    chunk_pages,
)
from app.ingest.parser import ParsedPage


DOC_ID = "doc-1"
RAG_ID = "rag-1"


def _pages(*texts: str) -> list[ParsedPage]:
    return [ParsedPage(page_number=i + 1, text=t) for i, t in enumerate(texts)]


def test_empty_input_yields_no_chunks() -> None:
    chunks = chunk_pages([], document_id=DOC_ID, rag_id=RAG_ID, size=100, overlap=10)
    assert chunks == []


def test_whitespace_only_page_skipped() -> None:
    chunks = chunk_pages(
        _pages("   \n  "),
        document_id=DOC_ID, rag_id=RAG_ID, size=100, overlap=10,
    )
    assert chunks == []


def test_short_page_single_chunk() -> None:
    chunks = chunk_pages(
        _pages("hello world"),
        document_id=DOC_ID, rag_id=RAG_ID, size=100, overlap=10,
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == "hello world"
    assert c.page_number == 1
    assert c.chunk_index == 0
    assert c.document_id == DOC_ID
    assert c.rag_id == RAG_ID


def test_long_page_split_with_overlap() -> None:
    text = "a" * 100
    chunks = chunk_pages(
        _pages(text),
        document_id=DOC_ID, rag_id=RAG_ID, size=30, overlap=10,
    )
    # size=30, overlap=10, step=20. chars 0..30, 20..50, 40..70, 60..90, 80..100 → 5 chunks
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]
    assert len(chunks[0].text) == 30
    assert len(chunks[-1].text) == 20
    # Adjacent chunks should share 10 chars (the overlap window).
    for prev, curr in zip(chunks, chunks[1:]):
        assert prev.text.endswith(curr.text[:10])


def test_overlap_clamped_to_size_minus_one() -> None:
    """Asking for overlap >= size shouldn't crash or infinite-loop."""
    chunker = FixedWindowChunker(size=20, overlap=50)
    chunks = chunker.chunk_pages(
        _pages("a" * 100),
        document_id=DOC_ID, rag_id=RAG_ID,
    )
    # overlap must be < size; expect 19 after clamping.
    assert chunker.overlap == 19
    assert len(chunks) >= 2


def test_invalid_size_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindowChunker(size=0, overlap=0)
    with pytest.raises(ValueError):
        FixedWindowChunker(size=10, overlap=-1)


def test_metadata_carries_window_bounds() -> None:
    chunks = chunk_pages(
        _pages("abcdefghij" * 5),
        document_id=DOC_ID, rag_id=RAG_ID, size=15, overlap=5,
    )
    assert chunks[0].metadata["chunk_size"] == 15
    assert chunks[0].metadata["chunk_overlap"] == 5
    assert chunks[0].metadata["char_start"] == 0


def test_multi_page_indexes_are_monotonic_per_document() -> None:
    chunks = chunk_pages(
        _pages("aaa", "bbb"),
        document_id=DOC_ID, rag_id=RAG_ID, size=2, overlap=0,
    )
    page_one = [c for c in chunks if c.page_number == 1]
    page_two = [c for c in chunks if c.page_number == 2]
    assert page_one and page_two
    # ``chunk_index`` increases monotonically across the whole
    # document (NOT reset per page). Page numbers do partition them.
    assert all(page_one[i].chunk_index < page_one[i + 1].chunk_index for i in range(len(page_one) - 1))
    assert all(page_two[i].chunk_index < page_two[i + 1].chunk_index for i in range(len(page_two) - 1))
    assert page_one[-1].chunk_index < page_two[0].chunk_index


def test_chunk_is_dataclass_with_required_fields() -> None:
    chunks = chunk_pages(
        _pages("x"),
        document_id=DOC_ID, rag_id=RAG_ID, size=10, overlap=0,
    )
    c = chunks[0]
    assert isinstance(c, Chunk)
    for field in ("text", "document_id", "rag_id", "page_number", "chunk_index", "metadata"):
        assert hasattr(c, field)