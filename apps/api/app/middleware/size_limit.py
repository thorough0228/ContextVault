"""Content-Length pre-check middleware.

`python-multipart` buffers the entire multipart body in memory / on
spool before the FastAPI handler ever sees the file. A 1 GB POST
with a 20 MiB cap would still be fully read before our in-handler
``_read_with_cap`` rejects it.

This middleware inspects the ``Content-Length`` header BEFORE the body
is read and rejects oversized uploads with the same
``{"error":{"code":"payload_too_large","message":...}}`` envelope
the upload router uses, so the wire contract is consistent.
"""

from __future__ import annotations

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with a ``Content-Length`` header above the cap.

    Only ``Content-Length`` is checked — chunked uploads without a
    length header are NOT bounded here. Phase 6 stops at the simple
    case; ``Transfer-Encoding: chunked`` is restricted to internal
    service-to-service traffic in the deploy notes.
    """

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = int(max_bytes)

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                length = int(cl)
            except ValueError:
                length = -1
            if length > self.max_bytes:
                logger.warning(
                    "size_limit.rejected path=%s length=%d cap=%d",
                    request.url.path, length, self.max_bytes,
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "payload_too_large",
                            "message": f"request body {length} bytes exceeds {self.max_bytes}",
                        }
                    },
                )
        return await call_next(request)
