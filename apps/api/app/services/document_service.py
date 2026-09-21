"""Document service: upload + lifecycle + ownership.

Phase 3 adds document upload, listing, retrieval, and deletion. Every
operation is scoped by ``rag_id`` and that ``rag_id`` must already
have been verified to belong to the caller by the time we reach this
module — :func:`resolve_user_rag` enforces that.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import BadRequestError, ConflictError, NotFoundError
from app.models import Chunk, Document, Rag
from app.services import rag_service
from app.config import get_settings
from app.storage import build_storage_key, get_storage

logger = logging.getLogger(__name__)


def invalidate_bm25_for_rag(rag_id: str) -> None:
    """Hybrid search: drop the RAG's cached BM25 index after any
    document mutation (delete / replace / bluegreen switchover)."""
    from app.services import bm25_service

    bm25_service.invalidate(rag_id)


def compute_content_hash(body) -> str:
    """SHA-256 hex digest of a seekable binary stream; rewinds it."""

    body.seek(0)
    digest = hashlib.sha256()
    while True:
        piece = body.read(1024 * 1024)
        if not piece:
            break
        digest.update(piece)
    body.seek(0)
    return digest.hexdigest()


async def assert_filename_free(
    db: AsyncSession, *, rag_id: str, filename: str
) -> None:
    """409 when the rag already holds an active document with this name.

    Duplicate names are the classic "old and new content both
    retrievable" trap — the caller must use the PUT update endpoint
    instead. Superseded documents (bluegreen old versions) do not
    count: their replacement already exists.
    """

    row = await db.execute(
        select(Document.id).where(
            Document.rag_id == rag_id,
            Document.filename == filename,
            Document.superseded_by.is_(None),
        )
    )
    if row.first() is not None:
        raise ConflictError(
            f"document '{filename}' already exists in this RAG — "
            "use PUT /documents/{id}/content to update it"
        )


async def resolve_user_rag(
    db: AsyncSession, *, user_id: str, rag_id: str
) -> Rag:
    """Return the RAG only if it belongs to ``user_id``. 404 otherwise.

    Centralised here so the documents router doesn't have to repeat the
    pattern; same security boundary as :func:`rag_service.get_user_rag`.
    """
    return await rag_service.get_user_rag(db, user_id=user_id, rag_id=rag_id)


async def create_document_record(
    db: AsyncSession,
    *,
    user_id: str,
    rag_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    storage_key: str,
    content_hash: Optional[str] = None,
    supersedes: Optional[str] = None,
    status: str = "CREATED",
) -> Document:
    """Insert a Document row in CREATED state. Storage upload happens
    BEFORE this call so we never have a row pointing at a missing
    object."""
    doc = Document(
        rag_id=rag_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        storage_key=storage_key,
        content_hash=content_hash,
        supersedes=supersedes,
        status=status,
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    logger.info(
        "document.create.ok user_id=%s rag_id=%s document_id=%s file_type=%s",
        user_id, rag_id, doc.id, file_type,
    )
    return doc


async def list_user_documents(
    db: AsyncSession,
    *,
    user_id: str,
    rag_id: str,
    page: int = 1,
    page_size: int = 20,
) -> Tuple[List[Document], int]:
    """Page through documents inside a RAG. Caller has already
    validated ownership via :func:`resolve_user_rag`."""
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100

    offset = (page - 1) * page_size
    total = await db.scalar(
        select(Document.id).where(Document.rag_id == rag_id).order_by(None)
    )  # placeholder; replaced below
    # We need a proper count; do a separate count query.
    from sqlalchemy import func as sa_func

    total_q = await db.scalar(
        select(sa_func.count(Document.id)).where(Document.rag_id == rag_id)
    )
    total = int(total_q or 0)

    rows = await db.scalars(
        select(Document)
        .where(Document.rag_id == rag_id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    return list(rows.all()), total


async def get_user_document(
    db: AsyncSession,
    *,
    user_id: str,
    document_id: str,
) -> Document:
    """Return the Document if its RAG belongs to the caller, else 404.

    Crucial detail: ownership is checked through the parent Rag, not
    a ``Document.user_id`` column (we don't denormalise)."""
    # Resolve the document first; bail if missing.
    doc = await db.get(Document, document_id)
    if doc is None:
        raise NotFoundError(f"document {document_id} not found")
    # Then verify the parent RAG is owned by caller.
    rag = await db.get(Rag, doc.rag_id)
    if rag is None or rag.user_id != user_id:
        raise NotFoundError(f"document {document_id} not found")
    return doc


async def get_user_document_with_chunks(
    db: AsyncSession,
    *,
    user_id: str,
    document_id: str,
) -> Tuple[Document, List[Chunk]]:
    doc = await get_user_document(db, user_id=user_id, document_id=document_id)
    rows = await db.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.page_number, Chunk.chunk_index)
    )
    return doc, list(rows.all())


async def delete_user_document(
    db: AsyncSession,
    *,
    user_id: str,
    document_id: str,
) -> None:
    """Delete the row + best-effort delete the storage object.

    Storage failure does NOT block the DB delete; an orphaned object
    is cheaper to clean up than a dangling row.
    """
    doc = await get_user_document(db, user_id=user_id, document_id=document_id)
    storage_key = doc.storage_key

    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    invalidate_bm25_for_rag(doc.rag_id)
    await db.delete(doc)
    await db.flush()

    try:
        storage = get_storage()
        storage.delete(key=storage_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "document.delete.storage_failed document_id=%s err=%s",
            document_id, exc,
        )

    logger.info(
        "document.delete.ok user_id=%s document_id=%s", user_id, document_id
    )


async def mark_processing(
    db: AsyncSession, *, document_id: str
) -> Optional[Document]:
    doc = await db.get(Document, document_id)
    if doc is None:
        return None
    doc.status = "PROCESSING"
    doc.error_message = None
    await db.flush()
    return doc


async def mark_ready(
    db: AsyncSession,
    *,
    document_id: str,
    page_count: int,
    chunk_count: int,
) -> Optional[Document]:
    doc = await db.get(Document, document_id)
    if doc is None:
        return None
    doc.status = "READY"
    doc.page_count = page_count
    doc.error_message = None
    await db.flush()
    logger.info(
        "document.ready.ok document_id=%s pages=%d chunks=%d",
        document_id, page_count, chunk_count,
    )
    return doc


async def mark_failed(
    db: AsyncSession, *, document_id: str, error_message: str
) -> Optional[Document]:
    doc = await db.get(Document, document_id)
    if doc is None:
        return None
    doc.status = "FAILED"
    # Cap the error string — DB column is Text but we don't want an
    # unbounded stack trace blob.
    doc.error_message = error_message[:4096]
    await db.flush()
    logger.warning(
        "document.failed.ok document_id=%s err=%s", document_id, error_message,
    )
    return doc


async def replace_chunks(
    db: AsyncSession,
    *,
    document_id: str,
    rag_id: str,
    chunks: list,
) -> int:
    """Delete any pre-existing chunks for the document and insert the
    new list. Idempotency: re-running the task against the same
    document yields the same end state.

    Returns the number of rows inserted.
    """
    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await db.flush()

    if not chunks:
        return 0

    rows = [
        Chunk(
            document_id=document_id,
            rag_id=rag_id,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            text=c.text,
            metadata_json=c.metadata,
        )
        for c in chunks
    ]
    db.add_all(rows)
    await db.flush()
    return len(rows)

async def update_document_content(
    db: AsyncSession,
    *,
    user_id: str,
    document_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    new_hash: str,
    body,
    strategy: str,
) -> dict:
    """Document-update entry point (Phase 13, mirrors the production
    playbook: change detection by content hash, delete-then-reingest
    as the mutation path, bluegreen when validation before switchover
    matters).

    * identical hash  → ``{"updated": False}`` (idempotent no-op).
    * replace         → same document row: new object, row fields
      updated, status PROCESSING, existing ``process_document`` task
      re-ingests (its chunk replacement is a delete+insert inside the
      ingestion transaction — old data survives any failure).
    * bluegreen       → NEW document row ingests independently; once
      READY the old row is marked ``superseded_by`` (invisible to
      retrieval, rollbackable via ``rollback_document_update``).

    Returns a dict the router turns into the response.
    """

    doc = await get_user_document(
        db, user_id=user_id, document_id=document_id
    )
    if doc.superseded_by:
        raise ConflictError(
            "this document was superseded by a newer version and is "
            "hidden from retrieval — update the replacement instead"
        )
    if strategy not in ("replace", "bluegreen"):
        raise BadRequestError(
            f"unknown update strategy {strategy!r} (replace|bluegreen)"
        )

    if doc.content_hash and doc.content_hash == new_hash:
        logger.info(
            "document.update.unchanged document_id=%s", document_id
        )
        return {"updated": False, "reason": "content_unchanged", "document_id": doc.id}

    storage = get_storage()

    if strategy == "replace":
        old_key = doc.storage_key
        new_key = build_storage_key(
            user_id=user_id, rag_id=doc.rag_id, document_id=doc.id
        )
        body.seek(0)
        storage.upload(
            key=new_key, body=body,
            content_type="application/octet-stream",
        )
        doc.storage_key = new_key
        doc.filename = filename
        doc.file_type = file_type
        doc.file_size = file_size
        doc.content_hash = new_hash
        doc.status = "PROCESSING"
        doc.error_message = None
        await db.commit()
        if old_key and old_key != new_key:
            try:
                storage.delete(key=old_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "document.update.old_object_delete_failed key=%s err=%s",
                    old_key, str(exc)[:120],
                )
        from app.celery_client import celery_app

        celery_app.send_task(
            "app.tasks.process_document",
            kwargs={"document_id": doc.id},
            queue=get_settings().celery_default_queue,
        )
        logger.info(
            "document.update.replace document_id=%s strategy=replace",
            doc.id,
        )
        return {
            "updated": True,
            "strategy": "replace",
            "document_id": doc.id,
            "status": "PROCESSING",
        }

    # bluegreen: a NEW document row ingests independently; the old row
    # is superseded (hidden from retrieval) once ingestion succeeds —
    # enforced by process_document's completion callback below.
    import io

    pre = await create_document_record(
        db,
        user_id=user_id,
        rag_id=doc.rag_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        storage_key=build_storage_key(
            user_id=user_id, rag_id=doc.rag_id, document_id="__pending__"
        ),
    )
    real_key = build_storage_key(
        user_id=user_id, rag_id=doc.rag_id, document_id=pre.id
    )
    pre.storage_key = real_key
    pre.content_hash = new_hash
    pre.supersedes = doc.id
    pre.status = "PROCESSING"
    await db.commit()

    body.seek(0)
    storage.upload(
        key=real_key, body=body, content_type="application/octet-stream"
    )

    from app.celery_client import celery_app

    celery_app.send_task(
        "app.tasks.process_document",
        kwargs={
            "document_id": pre.id,
            "on_ready_supersede": doc.id,
        },
        queue=get_settings().celery_default_queue,
    )
    logger.info(
        "document.update.bluegreen old=%s new=%s", doc.id, pre.id
    )
    return {
        "updated": True,
        "strategy": "bluegreen",
        "document_id": doc.id,
        "new_document_id": pre.id,
        "status": "PROCESSING",
    }


async def rollback_document_update(
    db: AsyncSession,
    *,
    user_id: str,
    new_document_id: str,
) -> dict:
    """Undo a bluegreen update: delete the new document (with all of
    its chunks) and un-supersede the old one so retrieval sees it
    again."""

    new_doc = await get_user_document(
        db, user_id=user_id, document_id=new_document_id
    )
    old_id = new_doc.supersedes
    if not old_id:
        raise BadRequestError(
            "this document is not a bluegreen replacement — nothing to roll back"
        )
    old_doc = await db.get(Document, old_id)
    if old_doc is None:
        raise NotFoundError(f"superseded document {old_id} not found")
    if old_doc.superseded_by != new_doc.id:
        raise ConflictError(
            "supersede chain mismatch — the old document was re-superseded"
        )

    old_doc.superseded_by = None
    await delete_user_document(
        db, user_id=user_id, document_id=new_doc.id
    )
    await db.commit()
    logger.info(
        "document.update.rollback old=%s removed_new=%s", old_id, new_doc.id
    )
    return {
        "rolled_back": True,
        "restored_document_id": old_id,
        "removed_document_id": new_doc.id,
    }
