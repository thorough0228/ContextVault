"""phase3 documents + chunks

Revision ID: 0002_phase3_documents_and_chunks
Revises: 0001_phase2_users_and_rags
Create Date: 2026-09-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM

revision: str = "0002_phase3_documents_and_chunks"
down_revision: Union[str, None] = "0001_phase2_users_and_rags"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # create_type=False: the explicit .create() calls below already emit
    # CREATE TYPE; without it create_table() re-emits them and aborts the
    # whole fresh-database migration with DuplicateObject.
    doc_file_type = ENUM(
        "pdf", "txt", name="document_file_type", create_type=False
    )
    doc_file_type.create(op.get_bind(), checkfirst=True)

    doc_status = ENUM(
        "CREATED",
        "PROCESSING",
        "READY",
        "FAILED",
        name="document_status",
        create_type=False,
    )
    doc_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("rag_id", sa.String(length=36), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("file_type", doc_file_type, nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("storage_key", sa.String(length=1024), nullable=False),
        sa.Column("status", doc_status, nullable=False, server_default="CREATED"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["rag_id"],
            ["rags.id"],
            name="fk_documents_rag_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_documents_rag_id", "documents", ["rag_id"])
    op.create_index(
        "ix_documents_storage_key", "documents", ["storage_key"], unique=True
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("rag_id", sa.String(length=36), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_chunks_document_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["rag_id"],
            ["rags.id"],
            name="fk_chunks_rag_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_rag_id", "chunks", ["rag_id"])
    op.create_index(
        "ix_chunks_doc_idx", "chunks", ["document_id", "chunk_index"], unique=True
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_doc_idx", table_name="chunks")
    op.drop_index("ix_chunks_rag_id", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_storage_key", table_name="documents")
    op.drop_index("ix_documents_rag_id", table_name="documents")
    op.drop_table("documents")
    sa.Enum(name="document_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="document_file_type").drop(op.get_bind(), checkfirst=True)