"""Query/document tokenization for hybrid (keyword) retrieval.

Used by the ingestion writer to fill ``document_chunks.tokenized`` and
by the search service to build keyword queries and score the SQLite
fallback path. English terms are matched as whole words; Chinese text
is segmented with jieba (imported lazily so a missing dependency only
fails when Chinese tokenization is actually requested).

The output is a flat list of lowercase tokens with whitespace and
punctuation removed — the exact string form written to the database is
``" ".join(tokens)``, which feeds ``to_tsvector('simple', ...)``.
"""

from __future__ import annotations

import re
from typing import List

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

_jieba = None
_jieba_failed = False


def _get_jieba():
    global _jieba, _jieba_failed
    if _jieba is None and not _jieba_failed:
        try:
            import jieba  # type: ignore

            jieba.setLogLevel(60)  # silence the build-prefix log spam
            _jieba = jieba
        except ImportError:
            _jieba_failed = True
    return _jieba


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens: ASCII words + jieba-segmented CJK runs.

    Chinese output relies on jieba; when it is unavailable, CJK runs
    are emitted as 2-grams (with a trailing single character for odd
    lengths) so the keyword leg still works without the dependency.
    """

    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = []
    for match in _WORD_RE.finditer(lowered):
        if len(match.group()) >= 1:
            tokens.append(match.group())
    for match in _CJK_RE.finditer(lowered):
        run = match.group()
        jieba = _get_jieba()
        if jieba is not None:
            tokens.extend(t.strip() for t in jieba.lcut(run) if t.strip())
        else:
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
                if len(run) % 2 == 1:
                    tokens.append(run[-1])
    return tokens


def tokenize_to_string(text: str) -> str:
    return " ".join(tokenize(text))


# Function words that carry no topical signal. Wiki-page scoring
# (Phase 18) filters these from BOTH sides of the overlap so an
# out-of-kb question cannot ride "who is the" into a wiki-only route.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "what", "which", "who", "will",
    "with", "的", "了", "是", "在", "有", "和", "与", "及", "对", "把",
    "被", "将", "这", "那",
})


def content_tokens(text: str) -> set:
    """Token set with stopwords removed — topical signal only."""

    return {t for t in tokenize(text) if t not in STOPWORDS}
