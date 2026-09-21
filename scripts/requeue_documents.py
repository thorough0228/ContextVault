"""Re-enqueue document ingestion tasks.

Use cases:
* documents stuck in CREATED/FAILED (worker was down, task was lost);
* re-embedding after an embedding-provider/dimension switch — run with
  ``--all`` so READY documents are re-ingested at the new dimension.

Usage (repo root, project venv):

    python scripts/requeue_documents.py           # CREATED + FAILED
    python scripts/requeue_documents.py --all     # + READY (re-embed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

import psycopg2  # noqa: E402

from app.celery_client import celery_app  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="also re-ingest READY documents (e.g. after an embedding switch)",
    )
    args = parser.parse_args()

    settings = get_settings()
    statuses = ["CREATED", "FAILED"] + (["READY"] if args.all else [])

    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    with conn.cursor() as cur:
        placeholders = ", ".join(f"%s" for _ in statuses)
        cur.execute(
            f"SELECT id, filename, status FROM documents "
            f"WHERE status IN ({placeholders})",
            statuses,
        )
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("nothing to requeue")
        return 0

    for doc_id, filename, status in rows:
        result = celery_app.send_task("app.tasks.process_document", args=[doc_id])
        print(f"requeued {status:8} {filename} ({doc_id}) -> task {result.id}")

    print(f"{len(rows)} document(s) requeued — watch the worker log for READY/FAILED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
