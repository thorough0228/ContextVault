"""phase10 add csv/json document file types

Revision ID: 0010_phase10_csv_json_file_types
Revises: 0009_phase9_embedding_dimension
Create Date: 2026-09-17

Extends the ``document_file_type`` enum with ``csv`` and ``json`` so
structured uploads (CSV / JSON / JSONL) can be stored. Parse support
lives in ``app/ingest/parser.py`` (``parse_csv`` / ``parse_json``,
with an NDJSON fallback); ``.jsonl`` uploads reuse the ``json`` value.

Downgrade is a no-op: PostgreSQL cannot remove values from an enum
type, and orphaned 'csv'/'json' rows would become invalid. No-op on
SQLite (tests use plain VARCHAR).
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0010_phase10_csv_json_file_types"
down_revision: Union[str, None] = "0009_phase9_embedding_dimension"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("ALTER TYPE document_file_type ADD VALUE IF NOT EXISTS 'csv'")
    op.execute("ALTER TYPE document_file_type ADD VALUE IF NOT EXISTS 'json'")


def downgrade() -> None:
    # Enum values cannot be removed in PostgreSQL; keeping them is
    # harmless. Intentional no-op.
    pass
