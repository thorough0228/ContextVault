"""Golden-chunk alignment: retrieved DB chunks -> source record ids.

The eval corpus is serialized by the harness itself (``corpus.py``),
ingested through the REAL pipeline, and scored back against golden
source-record ids. The mapping is deterministic because every step of
the ingestion serialization is replicable:

1. Each source record renders as ``str(record["text"]).strip()`` —
   ``parser._record_to_text`` with the recognised ``text`` field.
2. Empty renders are dropped BEFORE page grouping (``parse_json`` skips
   falsy record texts).
3. Records are grouped ``_RECORDS_PER_PAGE`` (200) per virtual page and
   joined with ``"\\n\\n"`` (``parser._pages_from_records``).
4. The chunker stores ``char_start`` / ``char_end`` relative to that
   page text in chunk metadata (``chunker.FixedWindowChunker``).

So ``(page_number, char span)`` maps onto source records exactly — no
fuzzy text matching involved. A text-based fallback exists only as a
safety net for hits whose metadata was lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Import the live constant from the app rather than duplicating it —
# if the parser's page grouping ever changes, alignment follows.
from app.ingest import parser as _api_parser

_RECORDS_PER_PAGE = getattr(_api_parser, "_RECORDS_PER_PAGE", 200)


@dataclass(frozen=True)
class PageLayout:
    """One virtual page: its exact text and per-record char spans."""

    page_number: int
    text: str
    # (start, end, record_id) — spans are non-overlapping, ordered.
    boundaries: Tuple[Tuple[int, int, str], ...]


class CorpusIndex:
    """Deterministic record alignment for one ingested corpus file."""

    def __init__(self, records: Sequence[dict]) -> None:
        """``records``: [{"id": str, "text": str}, ...] in file order."""

        rendered: List[Tuple[str, str]] = []
        for rec in records:
            text = str(rec["text"]).strip()
            if text:  # mirror parse_json: empty renders never reach a page
                rendered.append((str(rec["id"]), text))

        self._pages: Dict[int, PageLayout] = {}
        self._record_texts: Dict[str, str] = {rid: t for rid, t in rendered}
        for page_no, start in enumerate(range(0, len(rendered), _RECORDS_PER_PAGE), start=1):
            batch = rendered[start : start + _RECORDS_PER_PAGE]
            page_text = "\n\n".join(t for _, t in batch)
            boundaries: List[Tuple[int, int, str]] = []
            pos = 0
            for rid, text in batch:
                boundaries.append((pos, pos + len(text), rid))
                pos += len(text) + 2  # the "\n\n" separator between records
            self._pages[page_no] = PageLayout(
                page_number=page_no, text=page_text,
                boundaries=tuple(boundaries),
            )

    @property
    def record_ids(self) -> Set[str]:
        return set(self._record_texts)

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def record_text(self, record_id: str) -> Optional[str]:
        return self._record_texts.get(record_id)

    def page_text(self, page_number: int) -> Optional[str]:
        layout = self._pages.get(page_number)
        return layout.text if layout else None

    def map_chunk(
        self,
        page_number: int,
        metadata: Optional[dict],
        chunk_text: Optional[str] = None,
    ) -> Set[str]:
        """Record ids overlapped by one retrieved chunk.

        Primary path uses the chunker's ``char_start`` / ``char_end``
        metadata. Fallback (metadata missing) locates the chunk text
        inside the page and uses its span. Returns an empty set when
        nothing matches — the caller treats that as an unmapped hit.
        """

        layout = self._pages.get(page_number)
        if layout is None:
            return set()

        span = self._span_from_metadata(metadata) or self._span_from_text(
            layout, chunk_text
        )
        if span is None:
            return set()
        cs, ce = span
        return {
            rid
            for (s, e, rid) in layout.boundaries
            if s < ce and e > cs  # intervals [s, e) and [cs, ce) overlap
        }

    @staticmethod
    def _span_from_metadata(metadata: Optional[dict]) -> Optional[Tuple[int, int]]:
        if not metadata:
            return None
        try:
            cs = int(metadata["char_start"])
            ce = int(metadata["char_end"])
        except (KeyError, TypeError, ValueError):
            return None
        if cs < 0 or ce <= cs:
            return None
        return cs, ce

    @staticmethod
    def _span_from_text(
        layout: PageLayout, chunk_text: Optional[str]
    ) -> Optional[Tuple[int, int]]:
        if not chunk_text:
            return None
        found = layout.text.find(chunk_text)
        if found < 0:
            return None
        return found, found + len(chunk_text)
