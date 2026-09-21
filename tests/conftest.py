"""Shared pytest configuration for the cross-app test runner.

The individual apps ship their own ``pyproject.toml`` + ``pytest.ini``.
This conftest exists for the top-level ``pytest`` invocation that fans
out to each app's suite.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "apps" / "api"
WORKER_DIR = REPO_ROOT / "apps" / "worker"
SHARED_PY_DIR = REPO_ROOT / "packages" / "shared" / "python"

for p in (API_DIR, WORKER_DIR, SHARED_PY_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)