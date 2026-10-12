"""phase18 wiki boundary routing — page meta column

Revision ID: 0017_phase18_wiki_meta
Revises: 0016_phase17_wiki_notes
Create Date: 2026-10-12

Adds ``wiki_pages.meta`` (nullable JSON): the page's declared boundary,
``{"status", "version", "covers", "does_not_cover", "generated_at"}``.
``covers``/``does_not_cover`` are routing hints emitted by the summary
LLM; ``status`` stored here is the generation-time snapshot (readers
derive the live status from ``stale`` + ``notes``). Nullable — pages
generated before Phase 18 carry no meta and route on score + hard
checks only.

SQLite (tests) is a no-op — ORM + create_all carry the schema.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op

revision: str = "0017_phase18_wiki_meta"
down_revision: Union[str, None] = "0016_phase17_wiki_notes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.add_column("wiki_pages", sa.Column("meta", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.drop_column("wiki_pages", "meta")
