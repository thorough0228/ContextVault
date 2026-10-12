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

    page = await db.scalars(
        select(WikiPage).where(
            WikiPage.rag_id == rag_id, WikiPage.kind == KIND_INDEX
        )
    )
    existing = page.first()

    # No summaries left (all deleted/pruned) → the index page must go
    # too; an empty index listing ghosts would keep surfacing in chat.
    if not content:
        if existing is not None:
            await db.delete(existing)
            await db.flush()
            logger.info("wiki.index.removed rag_id=%s (no summaries)", rag_id)
        return None

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
        from app.services import wiki_cache

        await wiki_cache.invalidate(doc.rag_id)
        logger.info(
            "wiki.stale.marked document_id=%s pages=%d", document_id, marked
        )
    return marked


async def wiki_context_for_query(
    db: AsyncSession, *, rag_id: str, query: str, top_k: int = 3
) -> List[dict]:
    """Wiki-first retrieval: BM25-ish lexical scoring of wiki pages
    against the query. Non-stale pages only (served through the Phase
    17 hot cache when enabled). Returns a list of
    ``{"slug", "title", "content", "source_refs", "notes", "score"}``
    dicts (may be empty)."""

    from app.services.tokenization import tokenize
    from app.services import wiki_cache

    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []
    pages = await wiki_cache.load_pages(db, rag_id=rag_id)
    scored: List[dict] = []
    for page in pages:
        doc_tokens = set(tokenize(page["title"] + "\n" + page["content"]))
        overlap = len(q_tokens & doc_tokens)
        if overlap == 0:
            continue
        scored.append(
            {
                "slug": page["slug"],
                "title": page["title"],
                "content": page["content"],
                "source_refs": page.get("source_refs") or [],
                "notes": page.get("notes"),
                "score": overlap / len(q_tokens),
            }
        )
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]


def wiki_citations_from_pages(pages: List[dict]) -> List[dict]:
    """Synthesise document-level citation dicts from wiki pages.

    A wiki page's ``source_refs`` carry the provenance frontier
    ({document_id, content_hash, filename}); the routed answer cites
    documents, not chunks. ``chunk_id``/``page_number`` are synthesised
    because the Citation schema requires them.
    """

    out: List[dict] = []
    for p in pages:
        for ref in p.get("source_refs") or []:
            out.append(
                {
                    "chunk_id": f"wiki:{p['slug']}:{ref.get('document_id')}",
                    "document_id": ref.get("document_id"),
                    "filename": ref.get("filename") or p["slug"],
                    "page_number": 0,
                    "chunk_text": p["content"],
                    "retrieval_score": float(p["score"]),
                    "metadata": {
                        "source": "wiki",
                        "slug": p["slug"],
                        "content_hash": ref.get("content_hash"),
                    },
                }
            )
    return out


async def prune_pages_for_document(
    db: AsyncSession, *, rag_id: str, document_id: str
) -> dict:
    """Immediate wiki consistency after a document DELETE (Phase 17).

    The ``source_refs`` provenance is a JSON column with no FK, so a
    deleted document would otherwise leave a routable page citing a
    ghost document_id. Prunes the ref from every doc_summary page;
    pages left without refs are deleted; the index page is rebuilt and
    the hot cache evicted. Returns cleanup counts.
    """

    rows = await db.scalars(
        select(WikiPage).where(
            WikiPage.rag_id == rag_id,
            WikiPage.kind == KIND_DOC_SUMMARY,
        )
    )
    pruned = deleted = 0
    touched = False
    for page in rows.all():
        refs = page.source_refs or []
        live = [r for r in refs if r.get("document_id") != document_id]
        if len(live) == len(refs):
            continue
        if live:
            page.source_refs = live
            pruned += 1
            touched = True
        else:
            await db.delete(page)
            deleted += 1
            touched = True
    if touched:
        await rebuild_index_page(db, rag_id=rag_id)
        await db.flush()
        from app.services import wiki_cache

        await wiki_cache.invalidate(rag_id)
        logger.info(
            "wiki.prune.ok rag_id=%s document_id=%s pruned=%d deleted=%d",
            rag_id, document_id, pruned, deleted,
        )
    return {"pruned": pruned, "deleted": deleted}


async def rescan_wiki_pages(db: AsyncSession, *, enqueue) -> dict:
    """Maintenance pass 1 (Phase 17): find READY documents whose wiki
    page is missing, stale, or built against a diverged content hash
    (a rename leaves hash-diverged pages that the update-path stale
    marker misses), mark them stale, and re-enqueue generation.

    ``enqueue`` is ``_enqueue_wiki_generation``-shaped (injected so
    tests can spy on it). Pure DB — safe on every beat tick.
    """

    from app.models import Chunk

    docs = (
        (await db.scalars(select(Document).where(Document.status == "READY")))
        .all()
    )
    docs_by_id = {d.id: d for d in docs}
    pages = (
        await db.scalars(
            select(WikiPage).where(WikiPage.kind == KIND_DOC_SUMMARY)
        )
    ).all()

    marked = 0
    enqueued_ids: List[str] = []
    for page in pages:
        diverged = False
        for ref in page.source_refs or []:
            doc = docs_by_id.get(ref.get("document_id"))
            if doc is not None and ref.get("content_hash") != doc.content_hash:
                diverged = True
        if diverged and not page.stale:
            page.stale = True
            marked += 1
            for ref in page.source_refs or []:
                if ref.get("document_id") in docs_by_id:
                    enqueued_ids.append(ref["document_id"])

    existing_slugs = {(p.rag_id, p.slug) for p in pages}
    for doc in docs:
        slug = _summary_slug(doc.filename)
        if (doc.rag_id, slug) in existing_slugs:
            continue
        # a page-less doc with no chunk body would skip regeneration
        # forever (empty_body) — don't re-enqueue it every tick
        has_body = await db.scalar(
            select(Chunk.id).where(Chunk.document_id == doc.id).limit(1)
        )
        if has_body:
            enqueued_ids.append(doc.id)

    enqueued = 0
    for document_id in dict.fromkeys(enqueued_ids):  # dedupe, keep order
        enqueue(document_id=document_id)
        enqueued += 1
    await db.commit()
    if marked or enqueued:
        logger.info("wiki.rescan.ok marked=%d enqueued=%d", marked, enqueued)
    return {"marked": marked, "enqueued": enqueued}


async def lint_orphan_pages(db: AsyncSession) -> dict:
    """Maintenance pass 2 (Phase 17): drop wiki pages whose provenance
    is gone — every source_refs entry points at a deleted document, or
    the slug no longer matches any live document filename (rename
    leftovers). Prunes partially-dead refs, rebuilds the affected
    RAGs' index pages, evicts their caches. Pure DB."""

    docs = (await db.scalars(select(Document))).all()
    docs_by_id = {d.id: d for d in docs}
    rag_slugs: dict = {}
    for d in docs:
        rag_slugs.setdefault(d.rag_id, set()).add(_summary_slug(d.filename))

    pages = (
        await db.scalars(
            select(WikiPage).where(WikiPage.kind == KIND_DOC_SUMMARY)
        )
    ).all()

    deleted = pruned = 0
    touched_rags: List[str] = []
    for page in pages:
        refs = page.source_refs or []
        live_refs = [r for r in refs if r.get("document_id") in docs_by_id]
        slug_live = page.slug in rag_slugs.get(page.rag_id, set())
        if not live_refs or not slug_live:
            await db.delete(page)
            deleted += 1
            touched_rags.append(page.rag_id)
            continue
        if len(live_refs) != len(refs):
            page.source_refs = live_refs
            pruned += 1
            touched_rags.append(page.rag_id)

    for rag_id in dict.fromkeys(touched_rags):
        await rebuild_index_page(db, rag_id=rag_id)
    await db.commit()
    if touched_rags:
        from app.services import wiki_cache

        for rag_id in dict.fromkeys(touched_rags):
            await wiki_cache.invalidate(rag_id)
        logger.info(
            "wiki.lint.ok deleted=%d pruned=%d rags=%d",
            deleted, pruned, len(set(touched_rags)),
        )
    return {"deleted": deleted, "pruned": pruned}


_CONTRADICTION_PROMPT = """You audit the wiki of a knowledge base. Below are
its distilled pages. Identify pairs of pages that CONTRADICT each other —
conflicting facts, numbers, dates, or claims about the same subject.

Answer with ONLY a JSON array (possibly empty):
[{{"slug_a": "...", "slug_b": "...", "summary": "one-line description"}}]

Pages:
{pages}
"""


def _parse_contradiction_array(raw: str) -> Optional[List[dict]]:
    """Best-effort strict-JSON object-array extraction (same posture as
    the memory fact parser: fence strip, bracket slice, strict loads).
    Returns ``None`` when the answer isn't a JSON array at all (parse
    failure — distinct from a genuine empty ``[]``)."""

    import json as _json
    import re as _re

    text = (raw or "").strip()
    fence = _re.search(r"```(?:json)?\s*(.+?)```", text, _re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = _json.loads(text[start : end + 1])
    except _json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [item for item in data if isinstance(item, dict)]


async def detect_contradictions(
    db: AsyncSession, *, rag_id: str, llm: LLMProvider
) -> List[dict]:
    """Maintenance pass 3 (Phase 17, LLM): compare a RAG's doc_summary
    pages and record contradictions in each involved page's ``notes``
    column ({"contradictions": [...]}); pages with no findings get
    their notes cleared. An unparseable LLM answer leaves notes
    untouched (degradation — never wipe findings on a bad answer)."""

    pages = (
        await db.scalars(
            select(WikiPage).where(
                WikiPage.rag_id == rag_id,
                WikiPage.kind == KIND_DOC_SUMMARY,
                WikiPage.stale.is_(False),
            )
        )
    ).all()
    if len(pages) < 2:
        return []

    body = "\n\n".join(f"### {p.slug}\n{p.content[:1500]}" for p in pages)
    try:
        raw = await llm.chat(
            [{"role": "user", "content": _CONTRADICTION_PROMPT.format(pages=body)}]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "wiki.contradictions.failed rag_id=%s err=%s",
            rag_id, str(exc)[:160],
        )
        return []
    findings = _parse_contradiction_array(raw)
    if findings is None:
        logger.warning(
            "wiki.contradictions.parse_failed rag_id=%s — notes untouched",
            rag_id,
        )
        return []

    findings = [
        f for f in findings
        if f.get("slug_a") and f.get("slug_b") and f.get("summary")
    ]
    by_slug = {p.slug: p for p in pages}
    involved: dict = {}
    for f in findings:
        for slug in (f["slug_a"], f["slug_b"]):
            if slug in by_slug:
                involved.setdefault(slug, []).append(f)
    for page in pages:
        page.notes = (
            {"contradictions": involved[page.slug]}
            if page.slug in involved
            else None
        )
    await db.commit()
    if findings:
        logger.info(
            "wiki.contradictions.ok rag_id=%s findings=%d pages=%d",
            rag_id, len(findings), len(involved),
        )
    return findings



