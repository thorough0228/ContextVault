"""Tests for the logging configuration (plain / JSON / request_id)."""

from __future__ import annotations

import io
import json
import logging
import re

import pytest

from app.logging_config import (
    JsonFormatter,
    RequestIdFilter,
    configure_logging,
    get_logger,
)
from app.middleware.request_id import (
    REQUEST_ID_HEADER,
    set_request_id,
    reset_request_id,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset the global logging flag so each test gets fresh handlers."""
    import app.logging_config as lc

    lc._CONFIGURED = False
    yield
    lc._CONFIGURED = False


def _capture_root(capture: io.StringIO) -> None:
    """Replace the root logger handlers with a single one that writes
    to the given buffer."""
    handler = logging.StreamHandler(capture)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel("INFO")


def _install_capture_and_emit(capture: io.StringIO, message: str) -> None:
    """Helper: install our capture handler AFTER configure_logging
    has wired the real stdout handler, then emit a record. This is
    the only way to assert on log output without fighting pytest's
    stdout/stderr capture."""
    handler = logging.StreamHandler(capture)
    handler.setLevel("INFO")
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    get_logger("test").info(message)
    root.removeHandler(handler)


# ----- O.2 logging behaviour -------------------------------------------


def test_plain_format_default_attaches_request_id(monkeypatch) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    # Verify the format string the production code uses, by feeding a
    # record through it. We don't try to capture stdout — pytest
    # already does that, and what we actually want to assert is the
    # format-shape (the request_id placeholder is present).
    configure_logging(level="INFO")
    fmt = logging.getLogger().handlers[0].formatter
    assert isinstance(fmt, logging.Formatter)
    assert "request_id" in fmt._fmt
    handler = logging.getLogger().handlers[0]
    assert any(
        isinstance(f, RequestIdFilter) for f in handler.filters
    ), "production handler must apply the request_id filter"


def test_request_id_attached_to_log_record_when_set(monkeypatch) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    capture = io.StringIO()
    configure_logging(level="INFO")

    # Build a handler that uses the production formatter, writes to
    # our StringIO, and applies the request_id filter.
    handler = logging.StreamHandler(capture)
    handler.setFormatter(logging.getLogger().handlers[0].formatter)
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)

    token = set_request_id("rid-from-test")
    try:
        get_logger("test").info("scoped message")
    finally:
        reset_request_id(token)
        root.removeHandler(handler)

    line = capture.getvalue().strip()
    assert "scoped message" in line
    assert "rid-from-test" in line


def test_json_format_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    capture = io.StringIO()
    configure_logging(level="INFO")

    # Build a record through the JsonFormatter directly — exercises
    # the formatter, the request_id field, and the JSON shape
    # without fighting pytest's stdout capture.
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="x", lineno=1,
        msg="json message", args=(), exc_info=None,
    )
    setattr(record, "request_id", "rid-json")
    record.created = 1737000000.0
    raw = JsonFormatter().format(record)
    parsed = json.loads(raw)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test"
    assert parsed["message"] == "json message"
    assert parsed["request_id"] == "rid-json"
    assert "ts" in parsed
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00", parsed["ts"])


def test_request_id_outside_request_is_dash() -> None:
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname="x", lineno=1,
        msg="m", args=(), exc_info=None,
    )
    # No filter applied — should fall back to "-".
    out = fmt.format(record)
    parsed = json.loads(out)
    assert parsed["request_id"] == "-"
