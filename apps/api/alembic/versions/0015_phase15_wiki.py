"""phase15 wiki layer — persistent LLM-maintained wiki pages

Revision ID: 0015_phase15_wiki
Revises: 0014_phase14_md_file_type
Create Date: 2026-09-25

One table for the wiki layer (Karpathy-style persistent knowledge):

* ``wiki_pages`` — LLM-generated markdown pages scoped to (rag_id),
  each compiled from a set of source documents at a specific content
  state. ``source_refs`` records the frontier it was built against:
  when any source document's content_hash changes, pages referencing
  it become stale (routing downgrade) until regenerated.

SQLite (tests) is a no-op — ORM + create_all carry the schema.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op

revision: str = "0015_phase15_wiki"
down_revision: Union[str, None] = "0014_phase14_md_file_type"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.create_table(
        "wiki_pages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "rag_id",
            sa.String(36),
            sa.ForeignKey("rags.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_refs", sa.JSON(), nullable=False),
        sa.Column("stale", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_wiki_pages_rag_kind", "wiki_pages", ["rag_id", "kind"])
    op.create_index(
        "ix_wiki_pages_rag_slug", "wiki_pages", ["rag_id", "slug"], unique=True
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.drop_index("ix_wiki_pages_rag_slug", table_name="wiki_pages")
    op.drop_index("ix_wiki_pages_rag_kind", table_name="wiki_pages")
    op.drop_table("wiki_pages")
