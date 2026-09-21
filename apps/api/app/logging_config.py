"""Structured logging configuration shared across the API process.

Single point of truth: importing :func:`configure_logging` once at app
startup guarantees consistent log formatting in uvicorn, sqlalchemy, and
our own loggers.

Two formats are supported:

* Plain text (default for dev): ``%(asctime)s [%(levelname)s] %(name)s
  [request_id=%(request_id)s] :: %(message)s`` — request_id is added
  by :class:`RequestIdFilter`.
* JSON (opt-in via ``LOG_FORMAT=json``): one structured object per
  line with ``ts``, ``level``, ``logger``, ``message``, ``request_id``.

Both formats stamp every line with the same correlation id, so
``grep "<uuid>" app.log`` is enough to follow a request through the
router, the DB session, the Celery task, and the LLM call.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

from app.middleware.request_id import get_request_id

_CONFIGURED = False


class RequestIdFilter(logging.Filter):
    """Attach the request id from the ContextVar to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Stable key order for log shippers."""

    _KEYS = ("ts", "level", "logger", "request_id", "message")

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Idempotent root-logger setup. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)

    log_format = os.environ.get("LOG_FORMAT", "plain").lower()
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        fmt = (
            "%(asctime)s [%(levelname)s] %(name)s "
            "[request_id=%(request_id)s] :: %(message)s"
        )
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S%z"))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Tame noisy libraries unless we are debugging
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
