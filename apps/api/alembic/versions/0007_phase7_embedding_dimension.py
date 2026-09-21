"""phase7 embedding dimension — realign the vector column with the
configured embedding provider

Revision ID: 0007_phase7_embedding_dimension
Revises: 0006_phase6_user_is_admin
Create Date: 2026-09-17

Switching ``EMBEDDING_PROVIDER`` away from ``hash`` changes the model's
output dimension (hash-256 → e.g. embo-01's 1536). The column dimension
was baked in by migration ``0004_phase4_pgvector`` from the config
active at that time, so this migration re-ALTers it to the dimension
currently configured — same mechanism, same env.py injection.

Existing vectors cannot survive a dimension change (a 256-dim vector
cannot cast to vector(1536)), so they are nulled first. Documents stay
``READY`` but retrieve nothing until the ingestion task re-runs for
them (re-uploading or re-enqueueing), which re-embeds at the new
dimension. No-op on SQLite (tests).
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import context, op

revision: str = "0007_phase7_embedding_dimension"
down_revision: Union[str, None] = "0006_phase6_user_is_admin"
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

    # Old-dimension vectors cannot be cast to the new type; drop them
    # (documents are re-embedded on their next ingestion run).
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
        "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(256) "
        "USING embedding::text::vector(256)"
    )
