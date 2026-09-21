"""Backfill embeddings for one document directly via the API provider.

Operational tool (same tier as requeue_documents.py): the Celery solo
worker on Windows can silently stall on very large ingestion tasks, so
this script embeds the document's chunks in small, independently
retried batches and writes the pgvector literals straight to
Postgres. Resumable — only rows with a NULL embedding are processed.

    python scripts/backfill_embeddings.py <document_id>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

import psycopg2
import psycopg2.extras

from app.config import get_settings
from app.embedding.factory import get_embedding_provider


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    document_id = sys.argv[1]

    settings = get_settings()
    provider = get_embedding_provider()
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )

    def remaining(cur) -> list[tuple[str, str]]:
        cur.execute(
            "SELECT id, chunk_text FROM document_chunks "
            "WHERE document_id = %s AND embedding IS NULL "
            "ORDER BY chunk_index",
            (document_id,),
        )
        return cur.fetchall()

    batch_size = 8
    done = 0
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT status FROM documents WHERE id = %s", (document_id,))
            row = cur.fetchone()
            if row is None:
                print(f"document {document_id} not found")
                sys.exit(1)
            print(f"document status: {row[0]}")

            while True:
                rows = remaining(cur)
                if not rows:
                    break
                print(f"remaining: {len(rows)}", flush=True)
                for start in range(0, len(rows), batch_size):
                    batch = rows[start : start + batch_size]
                    ids = [r[0] for r in batch]
                    texts = [r[1] for r in batch]
                    for attempt in range(5):
                        try:
                            vectors = provider.embed_texts(texts)
                            break
                        except Exception as exc:  # noqa: BLE001
                            wait = 3 * (attempt + 1)
                            print(
                                f"  batch retry {attempt + 1} in {wait}s: "
                                f"{str(exc)[:100]}",
                                flush=True,
                            )
                            time.sleep(wait)
                    else:
                        print("  batch permanently failed — aborting (resumable)")
                        sys.exit(2)
                    psycopg2.extras.execute_values(
                        cur,
                        "UPDATE document_chunks AS dc SET embedding = data.vec "
                        "FROM (VALUES %s) AS data(id, vec) WHERE dc.id = data.id",
                        [
                            (rid, "[" + ",".join(repr(float(x)) for x in vec) + "]")
                            for rid, vec in zip(ids, vectors)
                        ],
                        template="(%s, %s::vector)",
                    )
                    conn.commit()
                    done += len(batch)
                    print(f"  embedded {done} chunks", flush=True)
                    time.sleep(1.2)

            cur.execute(
                "UPDATE documents SET status = 'READY', error_message = NULL, "
                "updated_at = now() WHERE id = %s",
                (document_id,),
            )
            cur.execute(
                "SELECT count(*) FROM document_chunks "
                "WHERE document_id = %s AND embedding IS NOT NULL",
                (document_id,),
            )
            conn.commit()
            print(f"DONE: document READY, {cur.fetchone()[0]} vectors")
    conn.close()


if __name__ == "__main__":
    main()
