"""Worker request_id propagation: API → Celery task body.

These tests live in apps/api (not apps/worker) because they need
the API's import path to reach the task module + the same DB
engine. They verify that ``process_document`` calls
``set_request_id`` with the inbound header before invoking any
log statement.
"""

from __future__ import annotations

import io
import logging
from unittest.mock import patch

import pytest


@pytest.fixture
def captured_worker_log():
    """A StringIO bound to the root logger; logs written while
    ``request_id`` is set inside the task should land here."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel("DEBUG")
    # Re-use the same formatter / filter the API configures.
    from app.logging_config import RequestIdFilter

    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield buf
    finally:
        root.removeHandler(handler)


def test_task_sets_request_id_from_headers(captured_worker_log) -> None:
    """Patch ``_process_document_impl`` to capture the request_id
    seen by the task body, then verify the API's wrapper sets it
    from the inbound header."""
    captured: dict = {}

    def _fake_impl(self, document_id, on_ready_supersede=None):
        from app.middleware.request_id import get_request_id

        captured["rid"] = get_request_id()
        logging.getLogger("app.tasks").info("stub log line")
        return {"document_id": document_id, "status": "READY"}

    import app.tasks as tasks_mod
    from app.celery_client import celery_app

    with patch.object(tasks_mod, "_process_document_impl", _fake_impl):
        # ``send_task`` ignores ``task_always_eager`` (Celery quirk);
        # ``apply(headers=...)`` is the test-friendly way to inject
        # task headers in-process.
        celery_app.tasks["app.tasks.process_document"].apply(
            args=["x"],
            headers={"request_id": "rid-from-test"},
        )

    assert captured.get("rid") == "rid-from-test", (
        f"process_document did not propagate request_id; saw {captured!r}"
    )


def test_task_falls_back_to_no_id_when_header_missing(captured_worker_log) -> None:
    """No request_id header → id is None. Logs still go out, just
    with ``request_id=-``."""
    captured: dict = {}

    def _fake_impl(self, document_id, on_ready_supersede=None):
        from app.middleware.request_id import get_request_id

        captured["rid"] = get_request_id()
        return {}

    import app.tasks as tasks_mod
    from app.celery_client import celery_app

    with patch.object(tasks_mod, "_process_document_impl", _fake_impl):
        celery_app.tasks["app.tasks.process_document"].apply(args=["x"])

    assert captured.get("rid") is None
