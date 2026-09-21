"""Celery tasks.

Phase 1 only ships a smoke task — ``ping`` — that proves the worker is
alive, can receive a message, and returns a result. Subsequent phases
add the actual RAG ingestion / embedding / answer-generation jobs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from celery import shared_task

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.ping", bind=True, max_retries=3)
def ping(self, message: str = "pong") -> dict:
    """Smoke task. Echoes back ``message`` plus worker metadata."""
    started = datetime.now(timezone.utc).isoformat()
    logger.info("task.ping.start message=%s attempt=%s", message, self.request.retries + 1)
    time.sleep(0.05)  # simulate a small amount of work
    return {
        "task_id": self.request.id,
        "echo": message,
        "started_at": started,
        "worker": "contextvault-worker",
        "status": "ok",
    }


@shared_task(name="app.tasks.health_check")
def health_check() -> dict:
    """Returns ``{"status": "ok"}`` — used by liveness probes in Phase 2+."""
    return {"status": "ok"}