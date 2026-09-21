"""Process-local storage used by tests and offline dev."""

from __future__ import annotations

import io
from threading import Lock
from typing import BinaryIO, Dict

from app.exceptions import NotFoundError
from app.storage.base import Storage, StorageDownload


class InMemoryStorage:
    """Keyed by storage key; values are (body, content_type)."""

    def __init__(self) -> None:
        self._store: Dict[str, tuple[bytes, str]] = {}
        self._lock = Lock()

    def upload(
        self,
        *,
        key: str,
        body: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> None:
        data = body.read() if hasattr(body, "read") else body
        with self._lock:
            self._store[key] = (bytes(data), content_type)

    def download(self, *, key: str) -> StorageDownload:
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            raise NotFoundError(f"object {key} not found")
        body, ctype = entry
        return StorageDownload(key=key, body=body, content_type=ctype)

    def delete(self, *, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def exists(self, *, key: str) -> bool:
        with self._lock:
            return key in self._store

    def open_stream(self, *, key: str) -> BinaryIO:
        return io.BytesIO(self.download(key=key).body)

    def reset(self) -> None:
        with self._lock:
            self._store.clear()