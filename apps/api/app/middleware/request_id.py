"""Request-ID middleware.

Reads ``X-Request-Id`` (or generates a uuid4) per request, stores it
on ``request.state.request_id`` and pushes it into a
:class:`contextvars.ContextVar` so every log line in the request's
call tree carries the same correlation id.

The header is echoed back on the response so a curl trace
immediately shows the id. Client error responses (e.g. 413 from
SizeLimitMiddleware) get the id too because this middleware sits at
the same level as the others in the FastAPI stack.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-Id"
_MAX_LEN = 128  # reject absurdly long inbound ids

# Module-level ContextVar. The logging filter reads it; the worker
# tasks read it on their side via apps/worker/app/request_id.py.
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> Optional[str]:
    """Return the current request's id, or ``None`` outside a request."""
    return request_id_var.get()


def set_request_id(value: str) -> contextvars.Token:
    return request_id_var.set(value)


def reset_request_id(token: contextvars.Token) -> None:
    request_id_var.reset(token)


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header: str = REQUEST_ID_HEADER) -> None:
        super().__init__(app)
        self.header = header

    async def dispatch(self, request: Request, call_next):
        inbound = request.headers.get(self.header)
        if not inbound or len(inbound) > _MAX_LEN:
            request_id = str(uuid.uuid4())
        else:
            request_id = inbound
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.header] = request_id
        return response
