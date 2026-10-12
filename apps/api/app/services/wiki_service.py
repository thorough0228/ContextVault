"""Wiki service: LLM-maintained persistent knowledge layer (Phase 15).

Implements the Karpathy-style wiki as a first-class store scoped to a
RAG:

* **Doc summary pages** — generated asynchronously after a document
  reaches READY; one page per document, compiled from its text by the
  configured LLM. ``source_refs`` records the provenance frontier
  (document id + content hash) so staleness is detectable.
* **Index page** — per-RAG catalog rebuilt from all doc summaries;
  the chat router reads it first (Karpathy: "the LLM reads the index
  before answering").
* **Staleness** — when a source document's content hash changes
  (update/bluegreen), pages referencing it are marked stale; the chat
  router excludes stale pages from prompts until regenerated.

All LLM calls go through the configured LLMProvider; failures degrade
to log-and-skip (a missing wiki page degrades gracefully — the chunk
RAG pipeline remains the authoritative source).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import LLMProvider
from app.models import Document, WikiPage

logger = logging.getLogger(__name__)

KIND_DOC_SUMMARY = "doc_summary"
KIND_INDEX = "index"

_SUMMARY_PROMPT = """You maintain the wiki of a knowledge base. Compress the
following document into a structured markdown summary page for future
retrieval. Keep: key facts, numbers, definitions, decisions, and open
questions. Drop filler. Use these markdown sections exactly:

## Key Points
- (bullets)

## Key Data
- (facts, numbers, names)

## Open Questions
- (or "None")

Document (filename: {filename}):
{body}
"""

_MAX_BODY_CHARS = 12000


def _wiki_llm():
    """LLM for wiki generation. ``WIKI_LLM_MODEL`` selects an OpenAI-
    compatible model at embedding_openai_base_url (SiliconFlow) using
    the embedding key; empty falls back to the main LLM provider."""

    from app.config import get_settings

    settings = get_settings()
    if settings.wiki_llm_model:
        from app.embedding.openai_provider import OpenAIEmbeddingProvider  # noqa: F401
        from app.llm.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(
            api_key=settings.embedding_openai_api_key,
            base_url=settings.embedding_openai_base_url,
            model=settings.wiki_llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    from app.llm import get_llm_provider

    return get_llm_provider()


def _summary_slug(filename: str) -> str:
    slug = filename.strip().lower()
    slug = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in slug)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:240] or "untitled"


async def generate_doc_summary_page(
    db: AsyncSession,
    *,
    doc: Document,
    llm: LLMProvider,
) -> Optional[WikiPage]:
    """Generate (or regenerate) the wiki summary page for one document.

    Reads the document's text from the chunks table, asks the LLM for
    a structured summary, and upserts the ``doc_summary`` wiki page
    with full provenance. Returns the page, or None on LLM/IO failure
    (caller logs; the document itself is unaffected).
    """

    from sqlalchemy import select

    from app.models import Chunk

    body_rows = await db.scalars(
        select(Chunk.text)
        .where(Chunk.document_id == doc.id)
        .order_by(Chunk.page_number, Chunk.chunk_index)
    )
    body = "\n".join(body_rows.all())
    if not body.strip():
        logger.info(
            "wiki.summary.skip document_id=%s reason=empty_body", doc.id
        )
        return None
    if len(body) > _MAX_BODY_CHARS:
        head = body[: int(_MAX_BODY_CHARS * 0.8)]
        tail = body[-int(_MAX_BODY_CHARS * 0.2):]
        body = head + "\n…[middle omitted]…\n" + tail

    prompt = _SUMMARY_PROMPT.format(filename=doc.filename, body=body)
    try:
        content = await llm.chat([{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001
        # Degradation is intentional (chunk RAG stays authoritative),
        # but this is the only place a broken wiki chain surfaces —
        # log at ERROR with the provider detail so billing/auth/model
        # failures are greppable without re-running the task.
        logger.error(
            "wiki.summary.llm_failed document_id=%s model=%s err=%s",
            doc.id, getattr(llm, "model", "?"), str(exc)[:500],
        )
        return None
    content = content.strip()
    if not content:
        logger.warning("wiki.summary.empty document_id=%s", doc.id)
        return None

    slug = _summary_slug(doc.filename)
    source_refs = [
        {
            "document_id": doc.id,
            "content_hash": doc.content_hash,
            "filename": doc.filename,
        }
    ]

    page = await db.scalars(
        select(WikiPage).where(
            WikiPage.rag_id == doc.rag_id,
            WikiPage.kind == KIND_DOC_SUMMARY,
            WikiPage.slug == slug,
        )
    )
    existing = page.first()
    if existing is not None:
        existing.title = f"Summary: {doc.filename}"
        existing.content = content
        existing.source_refs = source_refs
        existing.stale = False
        page_obj = existing
    else:
        page_obj = WikiPage(
            id=str(uuid.uuid4()),
            rag_id=doc.rag_id,
            kind=KIND_DOC_SUMMARY,
            slug=slug,
            title=f"Summary: {doc.filename}",
            content=content,
            source_refs=source_refs,
            stale=False,
        )
        db.add(page_obj)
    await db.flush()
    logger.info(
        "wiki.summary.upserted document_id=%s slug=%s", doc.id, slug
    )
    return page_obj


async def rebuild_index_page(db: AsyncSession, *, rag_id: str) -> Optional[WikiPage]:
    """Rebuild the RAG's index page from all doc_summary pages."""

    rows = await db.scalars(
        select(WikiPage)
        .where(
            WikiPage.rag_id == rag_id,
            WikiPage.kind == KIND_DOC_SUMMARY,
            WikiPage.stale.is_(False),
        )
        .order_by(WikiPage.title)
    )
    summaries = list(rows.all())

    lines = ["# Knowledge Base Index", ""]
    for s in summaries:
        first_section = s.content.split("\n## ")[0]
        one_line = " ".join(
            ln.lstrip("- ").strip()
            for ln in first_section.splitlines()
            if ln.strip() and not ln.startswith("#")
        )[:200]
        lines.append(f"## {s.title} ({'stale' if s.stale else 'current'})")
        lines.append(f"- Summary: {one_line or '(empty)'}")
        lines.append(f"- Slug: {s.slug}")
        lines.append("")

    content = "\n".join(lines).strip() + "\n" if len(lines) > 2 else ""
    if not content:
        return None

    page = await db.scalars(
        select(WikiPage).where(
            WikiPage.rag_id == rag_id, WikiPage.kind == KIND_INDEX
        )
    )
    existing = page.first()
    if existing is not None:
        existing.content = content
        page_obj = existing
    else:
        page_obj = WikiPage(
            id=str(uuid.uuid4()),
            rag_id=rag_id,
            kind=KIND_INDEX,
            slug="index",
            title="Knowledge Base Index",
            content=content,
            source_refs=[],
            stale=False,
        )
        db.add(page_obj)
    await db.flush()
    return page_obj


async def mark_stale_for_document(db: AsyncSession, *, document_id: str) -> int:
    """Mark doc_summary pages referencing ``document_id`` as stale.

    Called from the document-update path (content change / bluegreen
    switch). Returns the number of pages marked.
    """

    doc = await db.get(Document, document_id)
    if doc is None:
        return 0
    rows = await db.scalars(
        select(WikiPage).where(
            WikiPage.rag_id == doc.rag_id,
            WikiPage.kind == KIND_DOC_SUMMARY,
            WikiPage.slug == _summary_slug(doc.filename),
        )
    )
    marked = 0
    for page in rows.all():
        refs = page.source_refs or []
        ref = next((r for r in refs if r.get("document_id") == document_id), None)
        if ref is None or ref.get("content_hash") != doc.content_hash:
            page.stale = True
            marked += 1
    if marked:
        await db.flush()
        logger.info(
            "wiki.stale.marked document_id=%s pages=%d", document_id, marked
        )
    return marked


async def wiki_context_for_query(
    db: AsyncSession, *, rag_id: str, query: str, top_k: int = 3
) -> List[dict]:
    """Wiki-first retrieval: BM25-ish lexical scoring of wiki pages
    against the query. Non-stale pages only. Returns a list of
    ``{"slug", "title", "content", "score"}`` dicts (may be empty)."""

    from app.services.tokenization import tokenize

    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []
    rows = await db.scalars(
        select(WikiPage).where(
            WikiPage.rag_id == rag_id,
            WikiPage.stale.is_(False),
        )
    )
    scored: List[dict] = []
    for page in rows.all():
        doc_tokens = set(tokenize(page.title + "\n" + page.content))
        overlap = len(q_tokens & doc_tokens)
        if overlap == 0:
            continue
        scored.append(
            {
                "slug": page.slug,
                "title": page.title,
                "content": page.content,
                "score": overlap / len(q_tokens),
            }
        )
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]



