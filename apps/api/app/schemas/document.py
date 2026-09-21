"""Document + chunk Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


DocumentStatusLiteral = Literal["CREATED", "PROCESSING", "READY", "FAILED"]
DocumentFileTypeLiteral = Literal["pdf", "txt", "csv", "json", "jsonl"]


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    rag_id: str
    filename: str
    file_type: DocumentFileTypeLiteral
    file_size: int
    storage_key: str
    status: DocumentStatusLiteral
    error_message: Optional[str] = None
    page_count: Optional[int] = None
    # Bluegreen update chain (Phase 13): non-null = superseded by a
    # newer version, hidden from retrieval, still rollbackable.
    superseded_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int = Field(description="Total items matching the query")
    page: int = Field(description="1-based page index")
    page_size: int = Field(description="Items per page")


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    rag_id: str
    page_number: int
    chunk_index: int
    text: str
    metadata_json: Optional[dict] = Field(
        default=None,
        validation_alias="metadata_json",
        serialization_alias="metadata",
    )


class DocumentDetailResponse(DocumentResponse):
    """Document + its chunks. Only returned by ``GET /documents/{id}``."""

    chunks: List[ChunkResponse] = Field(default_factory=list)