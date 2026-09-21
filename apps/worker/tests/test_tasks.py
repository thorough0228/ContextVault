"""Smoke tests for the Celery worker.

These run the tasks ``eagerly`` (in-process) so no Redis broker is
required. End-to-end broker tests are run via docker compose.
"""

from __future__ import annotations

from app.celery_app import celery_app
from app.worker_tasks import ping


def setup_module(_module):
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True


def test_ping_returns_metadata() -> None:
    result = ping.apply(args=("hello",))
    payload = result.get(timeout=5)
    assert payload["status"] == "ok"
    assert payload["echo"] == "hello"
    assert payload["worker"] == "contextvault-worker"
    assert payload["task_id"]  # non-empty


def test_ping_default_message() -> None:
    payload = ping.apply().get(timeout=5)
    assert payload["echo"] == "pong"