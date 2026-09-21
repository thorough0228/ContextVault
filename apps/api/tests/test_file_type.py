"""Tests for the file-type sniffer."""

from __future__ import annotations

import io

import pytest

from app.ingest.file_type import FileTypeError, detect_file_type


PDF_BYTES = b"%PDF-1.4\nfake body\n%%EOF\n"
TXT_BYTES = "Hello, world!\n你好，世界！\n".encode("utf-8")
NOT_PDF = b"just some text pretending to be pdf"


def test_pdf_detected_by_magic() -> None:
    assert detect_file_type(filename="report.pdf", body=PDF_BYTES) == "pdf"


def test_txt_detected_by_decodable_utf8() -> None:
    assert detect_file_type(filename="notes.txt", body=TXT_BYTES) == "txt"


def test_pdf_rejected_without_magic_bytes() -> None:
    with pytest.raises(FileTypeError):
        detect_file_type(filename="report.pdf", body=NOT_PDF)


def test_pdf_rejected_with_wrong_extension() -> None:
    with pytest.raises(FileTypeError):
        detect_file_type(filename="report.docx", body=PDF_BYTES)


def test_txt_rejected_on_invalid_utf8() -> None:
    bad = b"\xff\xfe\x00invalid utf8"
    with pytest.raises(FileTypeError):
        detect_file_type(filename="bad.txt", body=bad)


def test_unsupported_extension_rejected() -> None:
    with pytest.raises(FileTypeError):
        detect_file_type(filename="image.png", body=b"\x89PNG\r\n\x1a\n")

def test_csv_accepted() -> None:
    assert (
        detect_file_type(filename="data.csv", body="a,b\n1,2".encode("utf-8"))
        == "csv"
    )


def test_json_and_jsonl_accepted() -> None:
    assert (
        detect_file_type(filename="data.json", body=b'{"a": 1}') == "json"
    )
    assert (
        detect_file_type(filename="data.jsonl", body=b'{"a": 1}\n{"a": 2}')
        == "json"
    )


def test_csv_with_bom_accepted() -> None:
    body = b"\xef\xbb\xbf" + "name,price\nApple,10".encode("utf-8")
    assert detect_file_type(filename="x.csv", body=body) == "csv"


def test_csv_invalid_utf8_rejected() -> None:
    import pytest

    with pytest.raises(FileTypeError, match="not valid UTF-8"):
        detect_file_type(filename="x.csv", body=b"\xd6\xd0\xce\xc4")


def test_docx_still_rejected() -> None:
    import pytest

    with pytest.raises(FileTypeError, match="unsupported file type"):
        detect_file_type(filename="x.docx", body=b"PK\x03\x04xxxx")


def test_file_like_csv_accepted() -> None:
    import io

    body = io.BytesIO("a,b\n1,2".encode("utf-8"))
    assert detect_file_type(filename="data.csv", body=body) == "csv"


def test_file_like_invalid_utf8_rejected() -> None:
    import io

    import pytest

    with pytest.raises(FileTypeError, match="not valid UTF-8"):
        detect_file_type(filename="x.csv", body=io.BytesIO(b"\xd6\xd0\xce\xc4"))


def test_deep_bytes_beyond_probe_are_not_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trade-off documentation: encoding is validated on a bounded
    prefix for file-like bodies; garbage deeper in the file surfaces
    at parse time instead."""
    import io

    from app.ingest import file_type as ft

    monkeypatch.setattr(ft, "_PROBE_BYTES", 16)
    body = io.BytesIO(b"a,b\n" + b"x" * 16 + b"\xd6\xd0\xce\xc4")
    assert detect_file_type(filename="x.csv", body=body) == "csv"
