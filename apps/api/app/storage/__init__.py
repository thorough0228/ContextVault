"""Object storage abstraction.

Two implementations:

* :class:`S3Storage` — talks to MinIO / S3 via ``boto3``.
* :class:`InMemoryStorage` — process-local dict, used by tests so they
  can run without a real MinIO.

Both expose the same :class:`Storage` protocol. The factory
:func:`get_storage` chooses based on ``settings.storage_in_memory``.

Key keys follow the layout::

    user/{user_id}/rag/{rag_id}/documents/{document_id}/original

The original filename is **never** used as a key component — callers
pass a freshly-generated key from ``build_storage_key``.
"""

from app.storage.base import (
    Storage,
    StorageDownload,
    StorageUpload,
    build_storage_key,
)
from app.storage.factory import (
    get_storage,
    reset_storage_for_tests,
    set_storage_for_tests,
)

__all__ = [
    "Storage",
    "StorageUpload",
    "StorageDownload",
    "build_storage_key",
    "get_storage",
    "set_storage_for_tests",
    "reset_storage_for_tests",
]