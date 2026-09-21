"""Storage factory.

Tests call ``set_storage_for_tests(...)`` to swap in an
:class ``InMemoryStorage``; production code goes through
:func:`get_storage, which reads ``settings.storage_in_memory``.
"""

from __future__ import annotations

from threading import Lock

from app.config import Settings, get_settings
from app.storage.base import Storage
from app.storage.memory import InMemoryStorage
from app.storage.s3 import S3Storage


_instance: Storage | None = None
_lock = Lock()


def get_storage(settings: Settings | None = None) -> Storage:
    """Return the singleton :class:`Storage` for the current settings."""
    global _instance
    with _lock:
        settings = settings or get_settings()
        if _instance is None:
            if settings.storage_in_memory:
                _instance = InMemoryStorage()
            else:
                _instance = S3Storage(settings)
        return _instance


def set_storage_for_tests(storage: Storage) -> None:
    """Swap in a test storage. Always pair with :func:`reset_storage_for_tests`."""
    global _instance
    with _lock:
        _instance = storage


def reset_storage_for_tests() -> None:
    global _instance
    with _lock:
        _instance = None