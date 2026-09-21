"""Worker startup hooks.

``sweep_stray_processing`` is called once per worker boot via the
``worker_process_init`` signal. It catches documents that were
abandoned in ``PROCESSING`` because the previous worker died
between :func:`mark_processing` and :func:`mark_ready` /
:func:`mark_failed`. Without this sweeper a row could stay stuck
in ``PROCESSING`` forever, blocking retries.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


async def sweep_stray_processing(
    *,
    timeout_seconds: int = 300,
    batch_limit: int = 100,
    database_url: Optional[str] = None,
) -> int:
    """Mark long-running ``PROCESSING`` documents as ``FAILED``.

    Returns the number of rows updated. The sweeper is idempotent —
    running it twice in a row moves no extra rows.
    """
    settings = get_settings()
    url = database_url or settings.async_database_url

    engine = create_async_engine(url, future=True)
    factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, autoflush=False
    )
    threshold = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
    rows_updated = 0
    try:
        async with factory() as session:  # type: AsyncSession
            from app.models import Document as DocumentModel

            stmt = (
                update(DocumentModel)
                .where(
                    DocumentModel.status == "PROCESSING",
                    DocumentModel.updated_at < threshold,
                )
                .values(
                    status="FAILED",
                    error_message="sweeper: worker died mid-processing",
                )
                .execution_options(synchronize_session=False)
            )
            # ``Update.limit()`` is not cross-dialect; PostgreSQL
            # ignores it on UPDATE, SQLite errors. Apply the cap at
            # the application layer instead.
            if batch_limit:
                stmt = stmt.where(
                    DocumentModel.id.in_(
                        select(DocumentModel.id)
                        .where(
                            DocumentModel.status == "PROCESSING",
                            DocumentModel.updated_at < threshold,
                        )
                        .order_by(DocumentModel.updated_at)
                        .limit(batch_limit)
                    )
                )
            result = await session.execute(stmt)
            await session.commit()
            rows_updated = result.rowcount or 0
    finally:
        await engine.dispose()

    if rows_updated:
        logger.warning(
            "sweeper.stray_processing.failed count=%d threshold_seconds=%d",
            rows_updated, timeout_seconds,
        )
    return rows_updated


def run_sweep_at_startup() -> None:
    """Synchronous wrapper for the ``worker_process_init`` signal."""
    try:
        asyncio.run(sweep_stray_processing())
    except Exception:  # noqa: BLE001
        logger.exception("sweeper.startup_failed")
