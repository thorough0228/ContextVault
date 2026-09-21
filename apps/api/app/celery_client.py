"""Lightweight Celery client used by the API process.

The API process only sends messages — it does not register workers.
We import :mod:`app.tasks` here so ``@shared_task`` decorators bind
to this ``celery_app`` instance, which means
``celery_app.send_task("app.tasks.process_document", ...)`` works
without the worker code being importable from the API.

The worker process has its own ``celery_app`` defined in
``apps/worker/app/celery_app.py`` and registers the same tasks via
the shared task name. The two apps share no state.
"""

from __future__ import annotations

import logging

from celery import Celery

from app.config import get_settings

logger = logging.getLogger(__name__)

# Make sure task decorators register against this instance when
# ``app.tasks`` is imported.
_settings = get_settings()
celery_app = Celery(
    "contextvault_api_client",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_always_eager=False,
    task_acks_late=True,
    task_default_queue=_settings.celery_default_queue,
)

# Stash the database URL on the celery app so tasks can construct
# their own engine without re-reading settings. (Tests can override
# this in their fixtures.)
celery_app.conf.database_url = _settings.async_database_url

# Import the task module so its @shared_task decorators run against
# ``celery_app`` above. Keep this last — it pulls in models etc.
from app import tasks  # noqa: E402, F401

logger.info(
    "celery.client.ready broker=%s queue=%s",
    _settings.celery_broker_url,
    _settings.celery_default_queue,
)