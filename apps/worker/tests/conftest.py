"""Phase 6 — worker test setup.

The Worker's test files import from the API package (``app.db``,
``app.models``) because the heavy ingestion logic lives there. Make
the API directory importable on Windows, where relative
``pythonpath = ["../api"]`` doesn't resolve reliably.

The worker's own ``app`` package shadows the API's ``app`` because
both are on sys.path and both register an ``app`` module in
``sys.modules``. The cleanest workaround for tests that need the
API's ``app`` is to use ``importlib.import_module("app.db")`` —
that forces the path-resolved lookup, bypassing whichever ``app``
pytest already loaded.
"""

import os  # noqa: E402  (must be first)
import sys  # noqa: E402  (must be first)
from pathlib import Path  # noqa: E402

# Tests must be deterministic regardless of the developer's .env —
# real providers would make network calls from inside unit tests.
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("EMBEDDING_MODEL", "hash-256")
os.environ.setdefault("EMBEDDING_DIMENSION", "256")
os.environ.setdefault("LLM_PROVIDER", "hash")
os.environ.setdefault("LLM_MODEL", "hash-llm")

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) in sys.path:
    sys.path.remove(str(_API_DIR))
sys.path.insert(0, str(_API_DIR))