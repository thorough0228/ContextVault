"""phase17 wiki maintenance — notes column for contradiction findings

Revision ID: 0016_phase17_wiki_notes
Revises: 0015_phase15_wiki
Create Date: 2026-10-12

Adds ``wiki_pages.notes`` (nullable JSON): structured maintenance
findings, currently the output of the periodic contradiction scan
(``[{slug_a, slug_b, summary}, ...]``). Nullable — pages without
findings carry no row payload.

SQLite (tests) is a no-op — ORM + create_all carry the schema.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op

revision: str = "0016_phase17_wiki_notes"
down_revision: Union[str, None] = "0015_phase15_wiki"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.add_column("wiki_pages", sa.Column("notes", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.drop_column("wiki_pages", "notes")
