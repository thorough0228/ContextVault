"""Celery application factory.

Boot a worker with:

    celery -A app.celery_app worker --loglevel=INFO --pool=solo

(``--pool=solo`` is required on Windows; the default prefork pool
does not start there. Linux/Docker can omit it.)

The app uses Redis for both the broker and the result backend.

Phase 3: the heavy tasks (document ingestion) live in
``apps/api/app/tasks.py`` so the API and the worker share a single
definition. The worker package extends its ``__path__`` with the
API's ``app`` directory (see ``app/__init__.py``), which makes
``app.tasks`` resolve to the API's task module while worker-local
modules keep shadowing their API twins.
"""

from __future__ import annotations

import logging
import os
import sys

from celery import Celery  # noqa: E402
from celery.schedules import crontab as _crontab  # noqa: E402

from app.config import get_settings  # noqa: E402  (the API's superset config)
from app.request_id import RequestIdFilter  # noqa: E402

settings = get_settings()

logger = logging.getLogger(__name__)

# Configure the root logger with the request_id-aware formatter. Same
# shape as the API uses — see apps/api/app/logging_config.py.

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s "
            "[request_id=%(request_id)s] :: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
)
_handler.addFilter(RequestIdFilter())
root = logging.getLogger()
root.handlers.clear()
root.addHandler(_handler)
root.setLevel(settings.log_level.upper())

celery_app = Celery(
    "contextvault_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks", "app.worker_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=os.environ.get("TZ", "UTC"),
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="default",
    broker_connection_retry_on_startup=True,
    # Phase 17 wiki auto-maintenance. Runs only when a beat process is
    # up (`celery -A app.celery_app beat`); the tasks themselves are
    # safe no-ops when there is nothing to do.
    beat_schedule={
        "wiki-maintenance-every-30min": {
            "task": "app.tasks.wiki_maintenance",
            "schedule": _crontab(minute="*/30"),
        },
        "wiki-contradiction-scan-daily": {
            "task": "app.tasks.wiki_contradiction_scan",
            "schedule": _crontab(hour=4, minute=0),
        },
    },
    # Windows + billiard prefork crashes every task with
    # "not enough values to unpack (expected 3, got 0)" — force the
    # single-process solo pool there. Linux/Docker keep prefork.
    worker_pool="solo" if sys.platform == "win32" else "prefork",
)

logger.info(
    "celery.config.ready broker=%s backend=%s env=%s",
    settings.celery_broker_url,
    settings.celery_result_backend,
    settings.app_env,
)


# Phase 6: run the doc-stray sweeper once per worker boot. The
# ``worker_init`` signal fires once at the start of every worker
# process — one sweep per worker, no Celery beat dependency.
from app.lifespan import run_sweep_at_startup  # noqa: E402

from celery import signals  # noqa: E402


def _on_worker_init(**_kwargs) -> None:
    """Celery's signal receivers must accept keyword args; wrap
    the zero-arg :func:`run_sweep_at_startup`."""
    run_sweep_at_startup()


signals.worker_init.connect(_on_worker_init)