"""phase12 hybrid retrieval + conversation memory

Revision ID: 0012_phase12_hybrid_memory
Revises: 0011_phase11_embedding_bge_m3
Create Date: 2026-09-19

Three changes for hybrid retrieval (keyword leg over tokenized text)
and conversation memory (rolling summary + long-term facts):

* ``document_chunks.tokenized`` — space-separated tokens (jieba for
  CJK, lowercase words for ASCII) with a GIN index on
  ``to_tsvector('simple', tokenized)``. Existing rows are backfilled
  inline with an embedded copy of the tokenizer (migrations must be
  self-contained; the canonical implementation lives in
  ``app.services.tokenization``).
* ``conversations.summary`` / ``summary_until_message_id`` — rolling
  summary of the turns that fell out of the prompt window.
* ``memory_facts`` — long-term, per (user, rag) facts extracted from
  conversations.

SQLite (tests) is a no-op: the ORM models carry the new columns and
``Base.metadata.create_all`` builds them.
"""

from __future__ import annotations

import re
from typing import Sequence, Union

from alembic import context, op
import sqlalchemy as sa

revision: str = "0012_phase12_hybrid_memory"
down_revision: Union[str, None] = "0011_phase11_embedding_bge_m3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _tokenize(text: str) -> str:
    """Embedded copy of app.services.tokenization (keep in sync)."""
    lowered = (text or "").lower()
    tokens: list[str] = []
    try:
        import jieba  # type: ignore

        jieba.setLogLevel(60)
        for match in _WORD_RE.finditer(lowered):
            tokens.append(match.group())
        for match in _CJK_RE.finditer(lowered):
            tokens.extend(
                t.strip() for t in jieba.lcut(match.group()) if t.strip()
            )
        return " ".join(tokens)
    except ImportError:
        for match in _WORD_RE.finditer(lowered):
            tokens.append(match.group())
        for match in _CJK_RE.finditer(lowered):
            run = match.group()
            if len(run) < 2:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        return " ".join(tokens)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite (tests): ORM models + create_all carry the schema.
        return

    op.add_column(
        "document_chunks",
        sa.Column("tokenized", sa.Text(), nullable=True),
    )
    # Backfill existing rows with the tokenizer, then index.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, chunk_text FROM document_chunks WHERE tokenized IS NULL")
    ).fetchall()
    for chunk_id, chunk_text in rows:
        conn.execute(
            sa.text(
                "UPDATE document_chunks SET tokenized = :tok WHERE id = :id"
            ),
            {"tok": _tokenize(chunk_text), "id": chunk_id},
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_tokenized_tsv "
        "ON document_chunks USING GIN (to_tsvector('simple', tokenized))"
    )

    op.add_column(
        "conversations",
        sa.Column("summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("summary_until_message_id", sa.String(36), nullable=True),
    )

    op.create_table(
        "memory_facts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "rag_id",
            sa.String(36),
            sa.ForeignKey("rags.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_conversation_id", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_memory_facts_user_rag",
        "memory_facts",
        ["user_id", "rag_id"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_index("ix_memory_facts_user_rag", table_name="memory_facts")
    op.drop_table("memory_facts")
    op.drop_column("conversations", "summary_until_message_id")
    op.drop_column("conversations", "summary")
    op.execute(
        "DROP INDEX IF EXISTS ix_document_chunks_tokenized_tsv"
    )
    op.drop_column("document_chunks", "tokenized")
