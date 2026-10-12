"""Wiki boundary router (Phase 18) — the four-way query router.

A wiki page is a bounded knowledge unit, not an authoritative fact
store. ``WIKI_ONLY`` is allowed only when the matched pages pass every
checkable boundary condition together:

* relevant — lexical score at or above ``WIKI_ROUTE_MIN_SCORE``;
* fresh    — no stale flag and every ``source_refs`` entry still
  matches the live document's content hash;
* trusted  — no unresolved contradiction findings (``notes``);
* sufficient for the question — the query does not demand exact
  source verification (verbatim wording, page numbers, clause
  numbers) and does not touch a ``does_not_cover`` tag.

Any failed check degrades one step: ``WIKI_PLUS_RAG`` (wiki as leads,
chunks as the citable evidence), ``RAG_ONLY`` (no relevant page), or
``ABSTAIN`` (every matched page is conflicted — refuse instead of
guessing). Every decision is deterministic and carries a machine
reason string; coverage hints are treated as hints, never guarantees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document

ROUTE_WIKI_ONLY = "wiki_only"
ROUTE_WIKI_PLUS_RAG = "wiki_plus_rag"
ROUTE_RAG_ONLY = "rag_only"
ROUTE_ABSTAIN = "abstain"

STATUS_VERIFIED = "VERIFIED"
STATUS_STALE = "STALE"
STATUS_CONFLICTED = "CONFLICTED"


@dataclass
class RouteDecision:
    route: str
    reason: str
    score: float
    pages: List[dict] = field(default_factory=list)
    slugs: List[str] = field(default_factory=list)


# Exactness: the question asks to verify/quote the source itself — a
# distilled page may have rewritten or dropped the original wording.
_EXACT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"逐字", "verbatim_zh"),
    (r"原文", "original_text_zh"),
    (r"原句|原话", "original_wording_zh"),
    (r"页码|第\s*\d+\s*页", "page_number_zh"),
    (r"第\s*[\d一二三四五六七八九十百]+\s*[条款章章节]", "clause_number_zh"),
    (r"具体条款|准确条款|条款原文", "clause_zh"),
    (r"verbatim", "verbatim"),
    (r"exact\s+(wording|quote|text|figure|number)", "exact_match"),
    (r"quote\s+(the|from)", "quote_request"),
    (r"page\s+number", "page_number"),
    (r"section\s+\d+", "section_number"),
    (r"clause\s+\d+", "clause_number"),
)
_EXACT_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for pat, name in _EXACT_PATTERNS),
    re.IGNORECASE,
)

# Comparison / multi-hop: needs several sources jointly — a single
# distilled page may have dropped the relations between them.
_COMPARE_RE = re.compile(
    r"对比|比较|差异|分别|各?自的|compare|difference|versus|\bvs\b|respectively",
    re.IGNORECASE,
)


def requires_exact_verification(query: str) -> Tuple[bool, str]:
    """Heuristic: does the question demand the raw source itself?"""

    match = _EXACT_RE.search(query)
    if match is not None:
        return True, match.lastgroup or "exact"
    if _COMPARE_RE.search(query):
        return True, "comparison"
    return False, ""


def _derive_status(page: dict, valid_hashes: dict) -> str:
    """Live page status: STALE (source diverged/missing) beats
    CONFLICTED (unresolved findings) beats VERIFIED. The index page is
    a catalog rebuilt on every write — it has no per-document
    provenance to diverge, so it never grades STALE here."""

    if page.get("kind") == "index":
        return STATUS_VERIFIED
    refs = page.get("source_refs") or []
    if not refs:
        return STATUS_STALE  # no provenance → cannot be trusted
    for ref in refs:
        doc_id = ref.get("document_id")
        if doc_id not in valid_hashes:
            return STATUS_STALE  # source deleted
        if ref.get("content_hash") != valid_hashes[doc_id]:
            return STATUS_STALE  # source updated, page not regenerated
    notes = page.get("notes") or {}
    if notes.get("contradictions"):
        return STATUS_CONFLICTED
    return STATUS_VERIFIED


async def _live_source_hashes(db: AsyncSession, page_list: List[dict]) -> dict:
    """document_id → content_hash for every ref in the matched pages."""

    ref_ids = {
        ref.get("document_id")
        for page in page_list
        for ref in (page.get("source_refs") or [])
        if ref.get("document_id")
    }
    if not ref_ids:
        return {}
    rows = await db.execute(
        select(Document.id, Document.content_hash).where(
            Document.id.in_(ref_ids)
        )
    )
    return {doc_id: content_hash for doc_id, content_hash in rows.all()}


# Stopword handling is shared with wiki-page scoring (tokenization
# module) — a tag like "Exact Wording of Plots and Taglines" must not
# fire on "president of France" just because both contain "of".


def _does_not_cover_hit(page: dict, query_tokens: set, tokenize) -> Optional[str]:
    """First declared does_not_cover tag all of whose content tokens
    appear in the query. Tags are hints, not guarantees — a hit only
    downgrades the route."""

    from app.services.tokenization import STOPWORDS

    meta = page.get("meta") or {}
    for tag in meta.get("does_not_cover") or []:
        tag_tokens = {
            t for t in tokenize(str(tag)) if t.lower() not in STOPWORDS
        }
        if tag_tokens and tag_tokens <= query_tokens:
            return str(tag)
    return None


async def route_for_query(
    db: AsyncSession, *, rag_id: str, query: str, top_k: int = 3
) -> RouteDecision:
    """Decide how this turn may use the wiki layer. Never raises."""

    from app.config import get_settings
    from app.services.tokenization import tokenize
    from app.services import wiki_service

    settings = get_settings()
    pages = await wiki_service.wiki_context_for_query(
        db, rag_id=rag_id, query=query, top_k=top_k
    )
    slugs = [p["slug"] for p in pages]
    top_score = float(pages[0]["score"]) if pages else 0.0

    if not settings.wiki_route_enabled:
        return RouteDecision(ROUTE_RAG_ONLY, "routing_disabled", top_score, pages, slugs)

    if not pages or top_score < settings.wiki_route_min_score:
        return RouteDecision(ROUTE_RAG_ONLY, "no_relevant_page", top_score, pages, slugs)

    valid_hashes = await _live_source_hashes(db, pages)
    statuses = [_derive_status(p, valid_hashes) for p in pages]

    if STATUS_STALE in statuses:
        return RouteDecision(
            ROUTE_WIKI_PLUS_RAG, "source_stale_or_diverged", top_score, pages, slugs
        )

    # Only distilled knowledge pages carry contradictions; the index is
    # a catalog and always grades VERIFIED, so it must not dilute the
    # all-conflicted decision.
    summary_statuses = [
        s for p, s in zip(pages, statuses) if p.get("kind") != "index"
    ]
    conflicted = [s for s in summary_statuses if s == STATUS_CONFLICTED]
    if conflicted:
        if len(conflicted) == len(summary_statuses):
            return RouteDecision(ROUTE_ABSTAIN, "unresolved_conflict", top_score, pages, slugs)
        return RouteDecision(
            ROUTE_WIKI_PLUS_RAG, "partial_conflict", top_score, pages, slugs
        )

    exact, pattern = requires_exact_verification(query)
    if exact:
        return RouteDecision(
            ROUTE_WIKI_PLUS_RAG, f"exact_verification:{pattern}", top_score, pages, slugs
        )

    # WIKI_ONLY needs at least TWO overlapping content tokens — a
    # single-token lexical collision (e.g. "president" inside a movie
    # summary) is not evidence the page covers the question.
    top_overlap = int(pages[0].get("overlap") or 0)
    if top_overlap < 2:
        return RouteDecision(
            ROUTE_WIKI_PLUS_RAG, "weak_lexical_evidence", top_score, pages, slugs
        )

    query_tokens = set(tokenize(query))
    for page in pages:
        tag = _does_not_cover_hit(page, query_tokens, tokenize)
        if tag is not None:
            return RouteDecision(
                ROUTE_WIKI_PLUS_RAG, f"declared_not_covered:{tag[:60]}",
                top_score, pages, slugs,
            )

    return RouteDecision(ROUTE_WIKI_ONLY, "boundary_pass", top_score, pages, slugs)


def abstain_message(query: str) -> str:
    """Deterministic refusal for the ABSTAIN route — no LLM in the
    loop, so an unresolved conflict can never be papered over."""

    def _cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    if _cjk(query):
        return (
            "无法安全回答：与该问题相关的 Wiki 页面目前存在**尚未解决的冲突**"
            "（与其他页面的结论不一致）。为避免基于矛盾知识作答，本次不给出结论。"
            "建议直接检索原文核对，或在维护任务解决冲突后重新提问。"
        )
    return (
        "I can't answer this safely: the wiki pages relevant to this question "
        "currently carry **unresolved contradictions**. Rather than guess, "
        "please consult the source documents directly, or retry after the "
        "maintenance pass has resolved the conflict."
    )
