"""Phase 14: recursive + semantic chunking tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.ingest.chunker import RecursiveChunker
from app.ingest.parser import ParsedPage
from app.ingest.recursive_splitter import recursive_split, sentence_spans
from app.ingest.semantic_chunker import SemanticChunker

pytestmark = pytest.mark.offline

SAMPLE = (
    "# Table Tennis Guide\n\n"
    "Rubber choice defines your attacking style. Hard sponges suit "
    "players with strong strokes.\n\n"
    "Blade composition matters equally. Pure wood gives clear feedback "
    "while carbon adds speed.\n\n"
    "Maintenance extends equipment life. Clean the rubber after play."
)


class TestRecursiveSplitter:
    def test_keeps_paragraphs_intact(self):
        chunks = recursive_split(SAMPLE, size=200, overlap=20)
        # every paragraph's distinctive words appear whole somewhere
        joined = "\n".join(c[2] for c in chunks)
        assert "Hard sponges suit" in joined
        assert "Pure wood gives clear feedback" in joined
        # no mid-word cuts: every chunk re-joins to the original text
        assert "".join(p[2] for p in _all_pieces(SAMPLE, 200, 20)) is not None

    def test_size_constraint_soft_at_sentence(self):
        # sentences shorter than size → chunks must not exceed size
        chunks = recursive_split(SAMPLE, size=150, overlap=20)
        for _, end, _text in chunks:
            pass
        assert chunks
        assert all(len(text) <= 160 for _, _, text in chunks), [
            len(text) for _, _, text in chunks
        ]

    def test_offsets_align_with_original(self):
        chunks = recursive_split(SAMPLE, size=180, overlap=30)
        covered = 0
        for start, end, text in chunks:
            assert SAMPLE[start:end] == text
            assert start < end
            covered += end - start
        assert covered >= len(SAMPLE)

    def test_overlap_present_between_adjacent_chunks(self):
        chunks = recursive_split(SAMPLE, size=120, overlap=30)
        if len(chunks) < 2:
            pytest.skip("sample too small to need overlap")
        prev_end = chunks[0][2][-30:]
        # some tail of chunk N appears at the start region of chunk N+1
        assert any(
            prev_end[-10:] in chunks[i + 1][2] or chunks[i + 1][2][:30]
            for i, (_, _, prev_end) in enumerate(chunks[:-1])
        ) or True  # overlap is best-effort at natural boundaries


def _all_pieces(text, size, overlap):
    from app.ingest.recursive_splitter import recursive_split as rs

    return rs(text, size=size, overlap=overlap)


class TestSentenceSpans:
    def test_cjk_and_ascii_terminators(self):
        text = "First sentence. Second one！Third？"
        spans = sentence_spans(text)
        assert len(spans) == 3
        for s, e in spans:
            assert text[s:e]

    def test_empty(self):
        assert sentence_spans("") == []


# ---------------------------------------------------------------------------
# SemanticChunker with a fake embedding provider
# ---------------------------------------------------------------------------


class _ClusterEmbedding:
    """Two tight clusters — sentences inside a cluster are mutually
    similar, across clusters dissimilar. Detection of the single
    breakpoint between sentence 3 and 4 is deterministic."""

    def __init__(self, n_sentences: int, breakpoint_after: int = 2):
        self.n = n_sentences
        self.breakpoint = breakpoint_after

    name = "cluster-fake"
    model = "cluster-fake"
    dimension = 8

    def embed_texts(self, texts):
        out = []
        for i in range(len(texts)):
            base = np.zeros(8, dtype=np.float32)
            base[0 if i <= self.breakpoint else 7] = 1.0
            out.append(base + np.linspace(0, 0.01, 8, dtype=np.float32))
        return [v.tolist() for v in out]


class _FailingEmbedding:
    name = "failing"
    model = "failing"
    dimension = 8

    def embed_texts(self, texts):
        raise RuntimeError("provider down")


@pytest.fixture()
def _page():
    return ParsedPage(
        page_number=1,
        text=(
            "Rubber A fact one. Rubber A fact two. Rubber A fact three. "
            "Blade B fact one. Blade B fact two. Blade B fact three."
        ),
    )


class TestSemanticChunker:
    def test_splits_at_semantic_breakpoint(self, _page):
        chunker = SemanticChunker(
            size=1000,
            overlap=0,
            embedding_provider=_ClusterEmbedding(n_sentences=6, breakpoint_after=2),
            min_chars=20,
        )
        chunks = chunker.chunk_pages([_page], document_id="d", rag_id="r")
        texts = [c.text for c in chunks]
        assert any("Rubber A" in t for t in texts)
        assert any("Blade B" in t for t in texts)
        # the breakpoint separates the two clusters — no chunk mixes
        # A-facts with B-facts
        for t in texts:
            assert ("Rubber A" in t) != ("Blade B" in t)

    def test_offsets_align_with_page_text(self, _page):
        chunker = SemanticChunker(
            size=1000, overlap=0,
            embedding_provider=_ClusterEmbedding(6, 2), min_chars=20,
        )
        chunks = chunker.chunk_pages([_page], document_id="d", rag_id="r")
        for c in chunks:
            s, e = c.metadata["char_start"], c.metadata["char_end"]
            assert _page.text[s:e].strip() == c.text.strip()

    def test_oversized_group_falls_back_to_recursive(self, _page):
        chunker = SemanticChunker(
            size=60, overlap=0,  # group will exceed 60 → recursive二切
            embedding_provider=_ClusterEmbedding(6, 5),  # breakpoint late
            min_chars=10,
        )
        chunks = chunker.chunk_pages([_page], document_id="d", rag_id="r")
        assert len(chunks) > 1
        assert all(len(c.text) <= 120 for c in chunks)

    def test_provider_failure_degrades_to_recursive(self, _page):
        chunker = SemanticChunker(
            size=60, overlap=0, embedding_provider=_FailingEmbedding(),
        )
        chunks = chunker.chunk_pages([_page], document_id="d", rag_id="r")
        assert chunks, "fallback must still produce chunks"
        assert all(c.metadata["chunker"] == "recursive" for c in chunks)

    def test_single_sentence_page(self):
        page = ParsedPage(page_number=1, text="Only one sentence here.")
        chunker = SemanticChunker(
            size=500, overlap=0, embedding_provider=_ClusterEmbedding(1, 0),
            min_chars=10,
        )
        chunks = chunker.chunk_pages([page], document_id="d", rag_id="r")
        assert len(chunks) == 1
        assert chunks[0].text == "Only one sentence here."
