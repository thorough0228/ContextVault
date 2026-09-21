"""Celery worker package.

The worker runs the API's heavy tasks (document ingestion), which live
in ``apps/api/app/tasks.py`` and import the API's models / services /
storage stack. Both apps are named ``app``, and the worker's own
package wins the import (``celery -A app.celery_app`` needs it to) —
so a plain ``import app.tasks`` would silently load the worker's
Phase-1 smoke module and the ingestion task would never register.

We therefore append the API's ``app`` directory to this package's
``__path__``: worker-local modules keep shadowing their API twins
(``celery_app``, ``lifespan``, ``worker_tasks``), while API-only
modules (``tasks``, ``config``, ``models``, ``services``, ...) resolve
transparently. The sibling layout holds both in the repo and in the
worker Docker image (``/app/apps/worker`` + ``/app/apps/api``).
"""
__version__ = "0.1.0"

from pathlib import Path

_API_APP = Path(__file__).resolve().parents[2] / "api" / "app"
if _API_APP.is_dir() and str(_API_APP) not in __path__:
    __path__.append(str(_API_APP))
