"""Recursive character splitting (Phase 14, plan 2).

Splits at the most natural boundary available: paragraph → line →
sentence → space → character, recursing into finer separators only
when a piece still exceeds the target size. Pieces are then greedily
merged up to ``size`` with ``overlap`` characters carried over from
the previous chunk — so sentences and paragraphs survive intact
instead of being cut mid-word like a fixed sliding window.

Every piece carries its (start, end) offset in the ORIGINAL text so
produced chunks keep the ``char_start`` / ``char_end`` metadata that
the eval harness' record-alignment depends on.
"""

from __future__ import annotations

import re
from typing import List, Tuple

SEPARATORS: Tuple[str, ...] = ("\n\n", "\n", "。", ". ", "！", "？", "! ", "? ", " ", "")


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) spans of sentences in ``text``.

    Splits after CJK/ASCII terminators (`.!?。！？…`) with optional
    trailing whitespace; offsets are absolute into ``text``.
    """

    if not text:
        return []
    spans: List[Tuple[int, int]] = []
    pos = 0
    for piece in re.split(r"(?<=[.!?。！？…])\s*", text):
        if not piece:
            continue
        found = text.find(piece, pos)
        if found < 0:  # paranoia — pieces come from the text itself
            found = pos
        end = found + len(piece)
        spans.append((found, end))
        pos = end
    return spans


def _split_keeping_offset(
    text: str, sep: str, start_offset: int
) -> List[Tuple[int, int, str]]:
    """Split ``text`` on ``sep`` keeping absolute (start, end, piece)
    triples; the separator stays attached to the preceding piece."""

    if sep == "":
        return [
            (start_offset + i, start_offset + i + 1, text[i])
            for i in range(len(text))
        ]
    pieces: List[Tuple[int, int, str]] = []
    pos = 0
    while pos < len(text):
        found = text.find(sep, pos)
        if found < 0:
            pieces.append((start_offset + pos, start_offset + len(text), text[pos:]))
            break
        piece = text[pos : found + len(sep)]
        if piece:
            pieces.append(
                (start_offset + pos, start_offset + found + len(sep), piece)
            )
        pos = found + len(sep)
    return pieces


def _recursive_pieces(
    text: str,
    start_offset: int,
    separators: Tuple[str, ...],
    size: int,
) -> List[Tuple[int, int, str]]:
    """Pieces of ``text`` (absolute offsets), each ≤ ``size`` when a
    separator can achieve it; hard-cut as the last resort."""

    if len(text) <= size:
        return [(start_offset, start_offset + len(text), text)] if text else []

    for i, sep in enumerate(separators):
        if sep and sep not in text:
            continue
        pieces = _split_keeping_offset(text, sep, start_offset)
        if len(pieces) <= 1:
            continue
        rest = separators[i + 1 :]
        out: List[Tuple[int, int, str]] = []
        for s, e, piece in pieces:
            if len(piece) <= size:
                out.append((s, e, piece))
            elif rest:
                out.extend(_recursive_pieces(piece, s, rest, size))
            else:
                out.extend(
                    (s + j, s + j + size, piece[j : j + size])
                    for j in range(0, len(piece), size)
                )
        return out
    return []


def _merge_with_overlap(
    pieces: List[Tuple[int, int, str]], size: int, overlap: int
) -> List[Tuple[int, int, str]]:
    """Greedily merge fine pieces up to ``size`` characters, carrying
    ≥ ``overlap`` characters of the previous chunk's tail pieces into
    the next chunk."""

    chunks: List[Tuple[int, int, str]] = []
    cur: List[Tuple[int, int, str]] = []
    cur_len = 0

    def flush() -> None:
        if cur:
            chunks.append(
                (cur[0][0], cur[-1][1], "".join(p for _, _, p in cur))
            )

    for piece in pieces:
        piece_len = piece[1] - piece[0]
        if cur and cur_len + piece_len > size:
            flush()
            # carry tail pieces for overlap — bounded so the NEXT piece
            # still fits (overlap is best-effort, never re-overflows).
            budget = size - piece_len
            carried: List[Tuple[int, int, str]] = []
            carried_len = 0
            if budget > 0:
                for s, e, p in reversed(cur):
                    if carried_len + (e - s) > budget:
                        break
                    carried.insert(0, (s, e, p))
                    carried_len += e - s
                    if carried_len >= overlap:
                        break
            cur = carried
            cur_len = carried_len
        cur.append(piece)
        cur_len += piece_len
    flush()
    return chunks


def recursive_split(
    text: str, *, size: int, overlap: int
) -> List[Tuple[int, int, str]]:
    """Public helper: (start, end, text) chunks of ``text``."""

    pieces = _recursive_pieces(text, 0, SEPARATORS, size)
    return _merge_with_overlap(pieces, size, max(0, min(overlap, size // 2)))
