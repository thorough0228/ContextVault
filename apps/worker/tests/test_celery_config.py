"""Celery app configuration smoke tests."""

from __future__ import annotations

from app.celery_app import celery_app


def test_celery_app_name():
    assert celery_app.main == "contextvault_worker"


def test_celery_serializers_are_json():
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert "json" in celery_app.conf.accept_content


def test_celery_retry_policy():
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.broker_connection_retry_on_startup is True