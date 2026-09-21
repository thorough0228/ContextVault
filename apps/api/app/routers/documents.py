"""Documents router: upload, list, get, delete.

Security: every endpoint resolves the parent RAG through
``document_service.resolve_user_rag`` first, so cross-tenant reads
look identical to missing resources. The actual bytes go to S3/MinIO
through the storage abstraction — no part of the path / filename
from the request influences the storage key.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Annotated, BinaryIO

from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.celery_client import celery_app
from app.config import Settings, get_settings
from app.db import get_db
from app.deps.auth import get_current_user
from app.exceptions import BadRequestError, NotFoundError, AppError
from app.ingest.file_type import FileTypeError, detect_file_type, safe_basename
from app.models import Document, User
from app.schemas.document import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services import document_service
from app.services.document_service import (
    assert_filename_free,
    compute_content_hash,
)
from app.storage import build_storage_key, get_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


class _PayloadTooLargeError(AppError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "payload_too_large"


class _UnsupportedFileTypeError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "unsupported_file_type"


async def _read_with_cap(
    upload: UploadFile, *, max_bytes: int
) -> tuple[BinaryIO, int]:
    """Stream the upload into a disk-backed spool, rejecting oversize
    early with 413.

    Returns ``(file, size)`` positioned at 0. Spooling to disk (past
    32 MiB) keeps memory flat for large uploads — a fully-buffered
    ``bytes`` body would hold the whole file in RAM twice over by the
    time it reaches object storage. The caller must ``close()`` the
    file when done.
    """
    spool = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
    total = 0
    try:
        while True:
            piece = await upload.read(1024 * 1024)
            if not piece:
                break
            total += len(piece)
            if total > max_bytes:
                raise _PayloadTooLargeError(
                    f"upload exceeds {max_bytes} bytes"
                )
            spool.write(piece)
        spool.seek(0)
        return spool, total
    except BaseException:
        spool.close()
        raise


@router.post(
    "/rags/{rag_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document to one of the caller's RAGs",
)
async def upload_document(
    rag_id: str,
    file: Annotated[UploadFile, File(description="PDF or TXT file")],
    current: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    # 1. Verify the parent RAG belongs to the caller (404 if not).
    rag = await document_service.resolve_user_rag(
        db, user_id=current.id, rag_id=rag_id
    )

    # 2. Stream the body into a disk-backed spool — memory stays flat
    #    for large uploads. Closed explicitly on every exit path below.
    body, file_size = await _read_with_cap(
        file, max_bytes=settings.upload_max_bytes
    )
    if not file_size:
        body.close()
        raise BadRequestError("empty upload")

    # 2b. Phase 13: content hash for update detection.
    content_hash = compute_content_hash(body)

    # 3. Sniff file type — never trust the client's Content-Type.
    try:
        file_type = detect_file_type(
            filename=file.filename, body=body
        )
    except FileTypeError as exc:
        body.close()
        raise _UnsupportedFileTypeError(str(exc))

    # 4. Pre-allocate the document row + storage key, upload, then
    #    commit. We upload to storage first because a half-uploaded
    #    object is cheaper to clean up than a row pointing at a
    #    missing key.
    await assert_filename_free(
        db, rag_id=rag.id,
        filename=safe_basename(file.filename, fallback=file_type),
    )
    pre_doc = await document_service.create_document_record(
        db,
        user_id=current.id,
        rag_id=rag.id,
        filename=safe_basename(file.filename, fallback=file_type),
        content_hash=content_hash,
        file_type=file_type,
        file_size=file_size,
        storage_key=build_storage_key(
            user_id=current.id,
            rag_id=rag.id,
            document_id="__pending__",  # replaced below
        ),
    )
    # Replace placeholder key with the real one (UUID-based).
    real_key = build_storage_key(
        user_id=current.id, rag_id=rag.id, document_id=pre_doc.id
    )
    pre_doc.storage_key = real_key
    await db.flush()

    storage = get_storage()
    body.seek(0)
    storage.upload(
        key=real_key,
        body=body,
        content_type=file.content_type or "application/octet-stream",
    )
    body.close()  # spooled to disk past 32 MiB — free it early

    # Commit before enqueueing so the worker (which opens a fresh
    # DB connection) can see the row. The router-level get_db will
    # commit again on success — that's a no-op once we've already
    # flushed + committed.
    await db.commit()
    await db.refresh(pre_doc)

    # 5. Enqueue async ingestion. send_task uses the registered name;
    #    the worker (separate process) picks it up.
    try:
        celery_app.send_task(
            "app.tasks.process_document",
            kwargs={"document_id": pre_doc.id},
            queue=settings.celery_default_queue,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "documents.upload.enqueue_failed document_id=%s err=%s",
            pre_doc.id, exc,
        )
        # Don't fail the upload — the row is in CREATED and a future
        # retry path can re-enqueue. Logged loudly so on-call notices.
    logger.info(
        "documents.upload.ok user_id=%s rag_id=%s document_id=%s",
        current.id, rag.id, pre_doc.id,
    )
    return DocumentResponse.model_validate(pre_doc)


@router.get(
    "/rags/{rag_id}/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List documents in one of the caller's RAGs",
)
async def list_documents(
    rag_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1, le=1000),
    page_size: int = Query(default=20, ge=1, le=100),
) -> DocumentListResponse:
    # 404 if not yours.
    await document_service.resolve_user_rag(
        db, user_id=current.id, rag_id=rag_id
    )
    rows, total = await document_service.list_user_documents(
        db,
        user_id=current.id,
        rag_id=rag_id,
        page=page,
        page_size=page_size,
    )
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch one document and its chunks",
)
async def get_document(
    document_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    # Phase 6: explicit router-level ownership check on top of the
    # service check. The service re-validates by loading the parent
    # Rag; doing the same lookup here means the boundary is visible
    # at the route, not buried in the service.
    doc = await db.get(Document, document_id)
    if doc is None:
        raise NotFoundError("document not found")
    await document_service.resolve_user_rag(
        db, user_id=current.id, rag_id=doc.rag_id
    )
    doc, chunks = await document_service.get_user_document_with_chunks(
        db, user_id=current.id, document_id=document_id
    )
    return DocumentDetailResponse(
        **DocumentResponse.model_validate(doc).model_dump(),
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete one of the caller's documents",
)
async def delete_document(
    document_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    # Phase 6: explicit router-level ownership check (see get_document
    # above for the rationale). Crucial for delete — the service
    # currently invokes ``delete(Chunk)`` *after* the check, but the
    # router pre-check closes the door on a future service refactor
    # that reorders those two steps.
    doc = await db.get(Document, document_id)
    if doc is None:
        raise NotFoundError("document not found")
    await document_service.resolve_user_rag(
        db, user_id=current.id, rag_id=doc.rag_id
    )
    await document_service.delete_user_document(
        db, user_id=current.id, document_id=document_id
    )
    return None

@router.put(
    "/documents/{document_id}/content",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Update a document's content (hash-detected, delete-and-reingest or bluegreen)",
)
async def update_document_content(
    document_id: str,
    file: Annotated[UploadFile, File(description="Replacement file")],
    strategy: Annotated[str, Query(description="replace|bluegreen")] = "",
    current: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
):
    """Phase 13 document update.

    * identical content hash → 200 ``updated: false`` (idempotent);
    * replace → the SAME document is re-ingested (delete + re-chunk +
      re-embed inside the worker's transaction; failure keeps old data);
    * bluegreen → a NEW document ingests, the old one is superseded
      (hidden from retrieval) once ingestion succeeds, rollbackable.
    """
    body, file_size = await _read_with_cap(
        file, max_bytes=settings.upload_max_bytes
    )
    if not file_size:
        body.close()
        raise BadRequestError("empty upload")
    try:
        file_type = detect_file_type(filename=file.filename, body=body)
    except FileTypeError as exc:
        body.close()
        raise _UnsupportedFileTypeError(str(exc))
    new_hash = compute_content_hash(body)

    from app.services.document_service import update_document_content

    result = await update_document_content(
        db,
        user_id=current.id,
        document_id=document_id,
        filename=safe_basename(file.filename, fallback=file_type),
        file_type=file_type,
        file_size=file_size,
        new_hash=new_hash,
        body=body,
        strategy=strategy or settings.document_update_strategy,
    )
    body.close()
    from fastapi.responses import JSONResponse

    if not result.get("updated"):
        # 200 — idempotent no-op (decorator default is 202)
        return JSONResponse(status_code=200, content=result)
    return JSONResponse(status_code=202, content=result)


@router.post(
    "/documents/{document_id}/rollback",
    status_code=status.HTTP_200_OK,
    summary="Roll back a bluegreen update (delete new, restore old)",
)
async def rollback_document_update(
    document_id: str,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.services.document_service import rollback_document_update

    return await rollback_document_update(
        db, user_id=current.id, new_document_id=document_id
    )
