"""File-type detection.

We never trust the user-supplied ``Content-Type``. ``detect_file_type``
combines:

1. extension (lowercased basename)
2. magic-bytes sniff for PDF
3. UTF-8 decodability probe for TXT

The result is one of ``"pdf"``, ``"txt"``, or it raises
:class:`FileTypeError`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Final, Union

_PDF_MAGIC: Final = b"%PDF-"
# Text-family files are encoding-validated on this bounded prefix when
# the upload arrives as a file-like object — decoding a multi-hundred-MB
# body would double the memory footprint at the choke point. A malformed
# byte deeper in the file surfaces later as a parse failure with a
# clear document-level error.
_PROBE_BYTES: Final = 8 * 1024 * 1024


class FileTypeError(ValueError):
    """Raised when an upload is neither a known PDF nor decodable as text."""


def _extension(filename: str | None) -> str:
    if not filename:
        return ""
    _, ext = os.path.splitext(filename)
    return ext.lower().lstrip(".")


def detect_file_type(
    *, filename: str | None, body: Union[bytes, BinaryIO]
) -> str:
    """Return ``"pdf"``, ``"txt"``, ``"csv"`` or ``"json"``.

    Raise :class:`FileTypeError` otherwise.

    ``body`` may be raw bytes or a seekable binary file (large uploads
    are disk-spooled). The decision is conservative: we accept a file
    as PDF when the extension is .pdf *and* the magic bytes match.
    Text-family files (TXT / CSV / JSON / JSONL) must decode as UTF-8
    (BOM allowed) — validated on a bounded prefix for file-like bodies.
    Anything else — wrong extension, missing magic, invalid UTF-8 — is
    rejected.
    """
    ext = _extension(filename)

    if hasattr(body, "read"):
        pos = body.tell()
        body.seek(0)
        data = body.read(_PROBE_BYTES)
        body.seek(pos)
    else:
        data = body

    head = data[:8]

    if ext == "pdf":
        if not head.startswith(_PDF_MAGIC):
            raise FileTypeError(
                "file has .pdf extension but does not start with the PDF magic bytes"
            )
        return "pdf"

    if ext == "txt":
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FileTypeError(
                f"file has .txt extension but is not valid UTF-8: {exc}"
            ) from exc
        return "txt"

    if ext in {"csv", "json", "jsonl"}:
        # utf-8-sig strips a leading BOM (Excel-exported CSVs carry one).
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise FileTypeError(
                f"file has .{ext} extension but is not valid UTF-8: {exc}"
            ) from exc
        return "json" if ext in {"json", "jsonl"} else "csv"

    raise FileTypeError(
        f"unsupported file type (extension={ext!r}); "
        "only .pdf, .txt, .csv, .json and .jsonl are accepted"
    )


def safe_basename(filename: str | None, fallback: str) -> str:
    """Sanitize the original filename for the metadata column. Strips
    path components and trims whitespace."""
    if not filename:
        return fallback
    base = Path(filename).name.strip()
    return base or fallback