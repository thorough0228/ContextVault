"""phase14 markdown file type

Revision ID: 0014_phase14_md_file_type
Revises: 0013_phase13_document_update
Create Date: 2026-09-20

Adds 'md' to the document_file_type enum (MinerU-converted markdown
uploads). SQLite (tests) is a no-op — the column is a plain VARCHAR.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import context, op

revision: str = "0014_phase14_md_file_type"
down_revision: Union[str, None] = "0013_phase13_document_update"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TYPE document_file_type ADD VALUE IF NOT EXISTS 'md'")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    # Cannot remove enum values in PostgreSQL — no-op by design.
    pass
