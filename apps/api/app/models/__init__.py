"""SQLAlchemy ORM models.

Phase 2 ships ``User`` and ``Rag``. Phase 3 adds ``Document`` and
``Chunk``. Phase 4 adds ``DocumentChunk`` — the search index target
that carries the embedding vector alongside the chunk text. Every
model uses a UUID primary key so that IDs aren't guessable across
tenants; ``Rag.user_id`` is a foreign key back to ``User.id`` and is
the sole boundary the routers trust for ownership.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    cast,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TEXT, TypeDecorator

from pgvector.sqlalchemy import Vector

from app.db import Base


class _PgVectorText(TypeDecorator):
    """PostgreSQL binding for ``DocumentChunk.embedding`` (``vector(N)``).

    asyncpg has no codec for the pgvector ``vector`` type, so neither a
    Python list nor a bare string can be bound directly. We bind the
    pgvector text literal (``'[0.1,0.2,...]'``) and let the rendered
    ``CAST(... AS vector)`` coerce it — works on every driver. On SQLite
    (tests) the base ``JSON`` type is used instead via ``with_variant``.
    """

    impl = TEXT
    cache_ok = True

    def bind_expression(self, bindvalue):
        return cast(bindvalue, Vector())

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            return "[" + ",".join(repr(float(x)) for x in value) + "]"

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            return [float(x) for x in str(value).strip("[]").split(",") if x]

        return process


def _new_uuid() -> str:
    """Stringified UUID4 — readable in logs, opaque to clients."""
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # Phase 6: admin flag. Default False; promoted manually in the
    # DB (or via a small CLI). The flag gates ``/api/v1/metrics``.
    is_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    rags: Mapped[list["Rag"]] = relationship(
        "Rag", back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"User(id={self.id!r}, email={self.email!r})"


class Rag(Base):
    __tablename__ = "rags"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        SAEnum("ACTIVE", "PROCESSING", "ERROR", name="rag_status"),
        nullable=False,
        default="ACTIVE",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    owner: Mapped["User"] = relationship("User", back_populates="rags")
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="rag",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Rag(id={self.id!r}, user_id={self.user_id!r}, name={self.name!r})"


class Document(Base):
    """One upload per row. The actual bytes live in object storage; the
    row only carries metadata + a status the worker updates."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    rag_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(
        SAEnum("pdf", "txt", "csv", "json", "md", name="document_file_type"),
        nullable=False,
    )
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(
            "CREATED", "PROCESSING", "READY", "FAILED",
            name="document_status",
        ),
        nullable=False,
        default="CREATED",
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Document-update support (migration 0013):
    # * content_hash — SHA-256 of the stored object; identical hash on
    #   PUT makes the update an idempotent no-op.
    # * superseded_by / supersedes — bluegreen chain: a superseded
    #   document stays in the DB (rollbackable) but is excluded from
    #   retrieval everywhere.
    content_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    superseded_by: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, index=True
    )
    supersedes: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    rag: Mapped["Rag"] = relationship("Rag", back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    document_chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Document(id={self.id!r}, rag_id={self.rag_id!r}, "
            f"status={self.status!r})"
        )


class Chunk(Base):
    """One chunk produced by the chunker from one document page.

    ``metadata`` is a free-form JSON blob (parser/chunker specific —
    e.g. token counts, page offsets). Indexes target the two common
    query paths: lookup by document (for delete-by-document) and by
    rag (for retrieval in Phase 4).
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rag_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Chunk(id={self.id!r}, document_id={self.document_id!r}, "
            f"page={self.page_number}, idx={self.chunk_index})"
        )


class DocumentChunk(Base):
    """The Phase 4 search index target.

    One row per (document, chunk_index) tuple. Carries the embedding
    vector alongside the chunk text so the search query can sort by
    cosine distance without joining back to ``chunks``.

    * ``embedding`` is stored as a JSON list of floats on SQLite
      (tests). On PostgreSQL, migration ``0004_phase4_pgvector``
      ALTERs the column to ``vector(N)`` and adds an HNSW index.
    * ``rag_id`` is denormalised for fast tenant filtering — every
      search query MUST filter on ``rag_id`` before distance.
    * ``chunk_index`` is per-document, monotonic, matches the value
      on the corresponding ``chunks`` row.
    """

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rag_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON, nullable=True
    )
    # Hybrid retrieval: space-separated tokens (jieba for CJK), indexed
    # with a GIN tsvector on PostgreSQL (migration 0012). NULL on
    # SQLite (tests) — the Python keyword scorer reads chunk_text.
    tokenized: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(_PgVectorText(), "postgresql"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(
        "Document", back_populates="document_chunks"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DocumentChunk(id={self.id!r}, document_id={self.document_id!r}, "
            f"rag_id={self.rag_id!r}, chunk_index={self.chunk_index})"
        )


class Conversation(Base):
    """A chat thread scoped to one (user, rag) pair.

    Conversations are the only place that owns the message log.
    ``rag_id`` is the boundary used by every chat request; the user
    can't quote a different rag's ``conversation_id`` to leak history
    because the chat router re-checks ownership on every turn.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rag_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    # Rolling summary of the turns that fell out of the prompt window
    # (migration 0012). summary_until_message_id marks how far the
    # summary covers; both NULL when nothing has been compressed yet.
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_until_message_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Conversation(id={self.id!r}, user_id={self.user_id!r}, "
            f"rag_id={self.rag_id!r}, title={self.title!r})"
        )


class Message(Base):
    """One turn inside a conversation.

    ``citations`` is only populated for assistant messages and is a
    JSON list of :class:`Citation` dicts (see
    ``app.schemas.chat.Citation``). It is denormalised onto the
    message so the UI can render the citation chips from a single
    query and so the user can audit exactly which chunks influenced
    a given answer."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        SAEnum("user", "assistant", "system", name="message_role"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Message(id={self.id!r}, conversation_id={self.conversation_id!r}, "
            f"role={self.role!r})"
        )


class MemoryFact(Base):
    """One long-term fact extracted from a conversation (migration 0012).

    Scope is (user, rag): facts are knowledge-base-specific context the
    user revealed across sessions (preferences, constraints, project
    background). Injected into chat prompts when non-empty; written by
    the ``app.tasks.extract_memory`` Celery task.
    """

    __tablename__ = "memory_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rag_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryFact(id={self.id!r}, user_id={self.user_id!r}, "
            f"rag_id={self.rag_id!r}, content={self.content[:40]!r})"
        )


class WikiPage(Base):
    """One LLM-maintained wiki page (Phase 15).

    Scoped to a RAG. Kinds: "doc_summary" (one per document, compiled
    by the LLM at ingest), "index" (per-RAG catalog of summaries).

    Provenance: ``source_refs`` is a JSON array of
    ``{"document_id", "content_hash", "filename"}`` — the content-hash
    frontier the page was compiled against. When a source document's
    current hash diverges, the page is marked ``stale`` (routing
    downgrade; regeneration pending).
    """

    __tablename__ = "wiki_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    rag_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_refs: Mapped[list] = mapped_column(JSON, nullable=False)
    stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Phase 17 maintenance findings (contradiction scan output), None
    # when the page carries no findings.
    notes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WikiPage(id={self.id!r}, rag_id={self.rag_id!r}, "
            f"kind={self.kind!r}, slug={self.slug!r}, stale={self.stale!r})"
        )