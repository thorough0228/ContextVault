"""Worker-side request-id propagation.

Mirrors :mod:`apps.api.app.middleware.request_id` for the
worker process: a ``ContextVar`` + a log filter. The API passes the
``X-Request-Id`` value via Celery's task headers (see
``apps/api/app/tasks.py``); the worker sets it on the ContextVar
before invoking the task body so every log line carries the same id
the API emitted.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Optional

REQUEST_ID_HEADER = "X-Request-Id"

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> Optional[str]:
    return request_id_var.get()


def set_request_id(value: str) -> contextvars.Token:
    return request_id_var.set(value)


def reset_request_id(token: contextvars.Token) -> None:
    request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True
