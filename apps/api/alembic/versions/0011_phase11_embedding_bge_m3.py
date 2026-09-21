"""phase11 embedding dimension — realign the vector column with
BAAI/bge-m3 (1024 dims, multilingual)

Revision ID: 0011_phase11_embedding_bge_m3
Revises: 0010_phase10_csv_json_file_types
Create Date: 2026-09-19

Same mechanism as 0007/0008/0009: switching the embedding provider
(this time to the in-process ``BAAI/bge-m3`` at 1024 dims) requires
the pgvector column to be re-aligned, and old-dimension vectors are
dropped (documents are re-embedded on their next ingestion run).
No-op on SQLite (tests).

NOTE: NULLing embeddings is unconditional across all documents — the
6 paper PDFs whose source objects are lost in MinIO therefore lose
searchability until their originals are re-uploaded and requeued
(handover doc §5). Accepted trade-off (evals-driven model upgrade).
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import context, op

revision: str = "0011_phase11_embedding_bge_m3"
down_revision: Union[str, None] = "0010_phase10_csv_json_file_types"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite path — JSON column, dimension is implicit.
        return

    dimension = context.config.get_main_option("embedding_dimension") or "256"
    try:
        dim_int = int(dimension)
    except ValueError as exc:  # pragma: no cover — guarded in env.py
        raise RuntimeError(
            f"embedding_dimension must be an integer, got {dimension!r}"
        ) from exc

    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute(
        f"ALTER TABLE document_chunks "
        f"ALTER COLUMN embedding TYPE vector({dim_int}) "
        f"USING embedding::text::vector({dim_int})"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(512) "
        "USING embedding::text::vector(512)"
    )
