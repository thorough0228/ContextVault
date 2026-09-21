"""phase9 embedding dimension — realign the vector column with the
configured embedding provider

Revision ID: 0009_phase9_embedding_dimension
Revises: 0008_phase8_embedding_dimension
Create Date: 2026-09-17

Same mechanism as 0007/0008: switching the embedding provider (this
time to the in-process ``BAAI/bge-small-zh-v1.5`` at 512 dims) requires
the pgvector column to be re-aligned, and old-dimension vectors are
dropped (documents are re-embedded on their next ingestion run).
No-op on SQLite (tests).
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import context, op

revision: str = "0009_phase9_embedding_dimension"
down_revision: Union[str, None] = "0008_phase8_embedding_dimension"
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
        "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(1024) "
        "USING embedding::text::vector(1024)"
    )
