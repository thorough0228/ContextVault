"""Parser + chunker + file-type detection for Phase 3.

Three independent pieces, kept here so the worker can pull them in
without dragging router / DB code along:

* :mod:`.file_type`  — sniff extension + magic bytes
* :mod:`.parser`     — PDF (PyMuPDF) + TXT, both yielding ``ParsedPage``
* :mod:`.chunker`    — fixed-window character chunker with overlap
"""

from app.ingest.chunker import (
    Chunk,
    Chunker,
    FixedWindowChunker,
    chunk_pages,
)
from app.ingest.file_type import (
    detect_file_type,
    FileTypeError,
)
from app.ingest.parser import (
    ParsedPage,
    parse_document,
)

__all__ = [
    "Chunk",
    "Chunker",
    "FixedWindowChunker",
    "chunk_pages",
    "ParsedPage",
    "parse_document",
    "detect_file_type",
    "FileTypeError",
]