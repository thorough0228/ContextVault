"""Unit tests for corpus alignment — the spec of the ingestion serializer.

The CorpusIndex must replicate ``parser._record_to_text`` +
``parser._pages_from_records`` + ``chunker`` offsets EXACTLY, otherwise
golden ids silently detach from retrieved chunks and every retrieval
metric becomes noise.
"""

from __future__ import annotations

import pytest

from evals.harness.alignment import CorpusIndex

pytestmark = pytest.mark.offline

RECORDS = [
    {"id": "r1", "text": "alpha rubber review"},       # 19 chars
    {"id": "r2", "text": "  blade handle shapes  "},   # strips to 21 chars
    {"id": "r3", "text": ""},                          # dropped: empty render
    {"id": "r4", "text": "   "},                       # dropped: blank render
    {"id": "r5", "text": "sponge thickness guide"},
]


@pytest.fixture(scope="module")
def index() -> CorpusIndex:
    return CorpusIndex(RECORDS)


class TestPageLayout:
    def test_page_text_is_stripped_records_joined_by_blank_line(self, index):
        # r3/r4 never reach a page; r2 is stripped — mirrors parse_json.
        assert index.page_text(1) == (
            "alpha rubber review\n\nblade handle shapes\n\nsponge thickness guide"
        )

    def test_record_ids_skip_empty_renders(self, index):
        assert index.record_ids == {"r1", "r2", "r5"}

    def test_record_text_lookup(self, index):
        assert index.record_text("r2") == "blade handle shapes"
        assert index.record_text("nope") is None


class TestSpanMapping:
    def test_chunk_inside_one_record(self, index):
        page = index.page_text(1)
        start = page.find("blade handle")
        meta = {"char_start": start, "char_end": start + len("blade handle")}
        assert index.map_chunk(1, meta) == {"r2"}

    def test_chunk_spanning_record_boundary(self, index):
        page = index.page_text(1)
        # Window from mid-r1 to mid-r2 crosses the "\n\n" boundary.
        s = page.find("rubber")
        e = page.find("handle") + len("handle")
        meta = {"char_start": s, "char_end": e}
        assert index.map_chunk(1, meta) == {"r1", "r2"}

    def test_full_page_covers_all_records(self, index):
        page = index.page_text(1)
        meta = {"char_start": 0, "char_end": len(page)}
        assert index.map_chunk(1, meta) == {"r1", "r2", "r5"}

    def test_unknown_page_maps_empty(self, index):
        meta = {"char_start": 0, "char_end": 5}
        assert index.map_chunk(99, meta) == set()

    def test_bad_metadata_maps_empty(self, index):
        assert index.map_chunk(1, None) == set()
        assert index.map_chunk(1, {}) == set()
        assert index.map_chunk(1, {"char_start": "x", "char_end": "y"}) == set()
        assert index.map_chunk(1, {"char_start": -1, "char_end": 4}) == set()


class TestTextFallback:
    def test_chunk_text_locates_span(self, index):
        assert index.map_chunk(1, None, chunk_text="sponge thickness") == {"r5"}

    def test_unknown_chunk_text_maps_empty(self, index):
        assert index.map_chunk(1, None, chunk_text="not in page") == set()

    def test_metadata_wins_over_text(self, index):
        meta = {"char_start": 0, "char_end": 5}
        # Text says r5 but metadata is authoritative.
        assert index.map_chunk(1, meta, chunk_text="sponge thickness") == {"r1"}


class TestPagination:
    def test_records_split_across_pages(self):
        records = [
            {"id": f"r{i}", "text": f"text {i}"} for i in range(450)
        ]
        index = CorpusIndex(records)
        assert index.page_count == 3
        assert index.page_text(2).startswith("text 200")
        meta_last_p1 = {"char_start": 0, "char_end": 6}
        # First characters of page 1 -> r0; page 3 starts at r400.
        assert index.map_chunk(1, meta_last_p1) == {"r0"}
        assert index.map_chunk(3, meta_last_p1) == {"r400"}
