"""phase13 document update — content_hash + bluegreen supersede chain

Revision ID: 0013_phase13_document_update
Revises: 0012_phase12_hybrid_memory
Create Date: 2026-09-19

Document-update support:

* ``documents.content_hash`` — SHA-256 of the stored object; change
  detection for the PUT update endpoint (unchanged content is a cheap
  idempotent no-op).
* ``documents.superseded_by`` / ``supersedes`` — bluegreen update
  chain: a superseded document stays in the DB (rollbackable) but is
  excluded from retrieval.

Backfill: existing rows get their content_hash computed from the
stored object. Rows whose object is unavailable (lost originals) stay
NULL and self-heal on the first PUT.
"""

from __future__ import annotations

import hashlib
from typing import Sequence, Union

from alembic import context, op
import sqlalchemy as sa

revision: str = "0013_phase13_document_update"
down_revision: Union[str, None] = "0012_phase12_hybrid_memory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite (tests): ORM models + create_all carry the schema.
        return

    op.add_column(
        "documents", sa.Column("content_hash", sa.String(64), nullable=True)
    )
    op.add_column(
        "documents",
        sa.Column("superseded_by", sa.String(36), nullable=True),
    )
    op.add_column(
        "documents", sa.Column("supersedes", sa.String(36), nullable=True)
    )
    op.create_index(
        "ix_documents_rag_filename", "documents", ["rag_id", "filename"]
    )
    op.create_index(
        "ix_documents_superseded_by", "documents", ["superseded_by"]
    )

    # Backfill content hashes from object storage. Storage failures
    # (lost originals) leave the hash NULL — the first PUT self-heals.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT id, storage_key FROM documents WHERE content_hash IS NULL"
        )
    ).fetchall()
    from app.storage import get_storage

    storage = get_storage()
    computed = failed = 0
    for doc_id, key in rows:
        try:
            # StorageDownload.body is the full bytes payload.
            downloaded = storage.download(key=key)
            conn.execute(
                sa.text(
                    "UPDATE documents SET content_hash = :h WHERE id = :id"
                ),
                {
                    "h": hashlib.sha256(downloaded.body).hexdigest(),
                    "id": doc_id,
                },
            )
            computed += 1
        except Exception:  # noqa: BLE001 — lost object: leave NULL
            failed += 1
    print(f"[0013] content_hash backfill: computed={computed} failed={failed}")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.drop_index("ix_documents_superseded_by", table_name="documents")
    op.drop_index("ix_documents_rag_filename", table_name="documents")
    op.drop_column("documents", "supersedes")
    op.drop_column("documents", "superseded_by")
    op.drop_column("documents", "content_hash")
