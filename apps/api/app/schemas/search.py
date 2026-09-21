"""Search request/response schemas."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    """POST /api/v1/rags/{rag_id}/search payload."""

    query: str = Field(
        min_length=1,
        max_length=4096,
        description="Free-form natural language query",
    )
    top_k: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Number of hits to return (0 → server default from settings)",
    )


class SearchHit(BaseModel):
    """One search hit. The route enriches the row with the document's
    ``filename`` so the client doesn't need a second lookup."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_text: str
    score: float = Field(description="Cosine similarity in [0, 1] (higher = better)")
    page_number: int
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    hits: List[SearchHit]
    query: str
    top_k: int
    embedding_model: str
    embedding_dimension: int

    model_config = ConfigDict(arbitrary_types_allowed=True)