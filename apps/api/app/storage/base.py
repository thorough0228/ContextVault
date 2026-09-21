"""Storage protocol + key builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol, runtime_checkable


@dataclass(frozen=True)
class StorageUpload:
    """Pre-computed arguments for :meth:`Storage.upload`."""

    key: str
    content_type: str


@dataclass(frozen=True)
class StorageDownload:
    """Result of :meth:`Storage.download`. Bytes are read once and held
    in memory — Phase 3 keeps uploads under ``upload_max_bytes`` so this
    is fine; Phase 4+ should switch to streaming for larger files."""

    key: str
    body: bytes
    content_type: str | None


def build_storage_key(*, user_id: str, rag_id: str, document_id: str) -> str:
    """Build the canonical key for an uploaded original.

    The user_id / rag_id / document_id come from the database (server-
    generated UUIDs) — never from the client.
    """
    return f"user/{user_id}/rag/{rag_id}/documents/{document_id}/original"


@runtime_checkable
class Storage(Protocol):
    """Object-storage protocol. Two implementations satisfy it."""

    def upload(
        self,
        *,
        key: str,
        body: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> None:
        ...

    def download(self, *, key: str) -> StorageDownload:
        ...

    def delete(self, *, key: str) -> None:
        ...

    def exists(self, *, key: str) -> bool:
        ...

    def open_stream(self, *, key: str) -> BinaryIO:
        ...