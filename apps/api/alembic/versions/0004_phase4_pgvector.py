"""phase4 pgvector — convert embedding to vector(N) + HNSW index

Revision ID: 0004_phase4_pgvector
Revises: 0003_phase4_document_chunks
Create Date: 2026-09-15

This migration is a no-op on SQLite (tests). On PostgreSQL it:

1. Enables the ``vector`` extension.
2. Converts the ``embedding`` column from ``jsonb`` to ``vector(N)``
   where ``N`` comes from the alembic config option
   ``embedding_dimension`` (env.py reads ``EMBEDDING_DIMENSION`` and
   stashes it on the config so this migration stays parameterised).
3. Builds an HNSW index using cosine distance — the right choice for
   ``< 1M`` rows on a single-node Postgres, which is the expected
   Phase 4 scale.

If the dimension is wrong, this migration fails loudly — better than
silently building an index that won't be used.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import context, op

revision: str = "0004_phase4_pgvector"
down_revision: Union[str, None] = "0003_phase4_document_chunks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect != "postgresql":
        # SQLite path — JSON column is already correct.
        return

    dimension = context.config.get_main_option("embedding_dimension") or "256"
    try:
        dim_int = int(dimension)
    except ValueError as exc:  # pragma: no cover — guarded in env.py
        raise RuntimeError(
            f"embedding_dimension must be an integer, got {dimension!r}"
        ) from exc

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Neither json nor jsonb casts directly to vector; pgvector accepts
    # the text form, so hop through ``::text``. Embeddings are always
    # written as float arrays, which the vector text parser understands.
    op.execute(
        f"ALTER TABLE document_chunks "
        f"ALTER COLUMN embedding TYPE vector({dim_int}) "
        f"USING embedding::text::vector({dim_int})"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE jsonb "
              "USING embedding::jsonb")