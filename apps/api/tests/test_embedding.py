"""Tests for the embedding provider abstractions.

The :class:`HashEmbeddingProvider` is what production falls back to
when no API key is configured; it's also the deterministic default in
tests. Provider-mock coverage lives in the search/router tests so
this file focuses on the production-path guarantees:

* empty / blank text is a permanent error
* non-string input is a permanent error
* dimension is exactly what was configured
* output is deterministic across calls
* identical text yields identical vectors
"""

from __future__ import annotations

import httpx
import pytest

from app.embedding import (
    EmbeddingPermanentError,
    EmbeddingTransientError,
    get_embedding_provider,
    set_provider_for_tests,
    reset_provider_for_tests,
)
from app.embedding.hash_provider import HashEmbeddingProvider


def test_hash_provider_normalises_output() -> None:
    p = HashEmbeddingProvider(dimension=8)
    vec = p.embed_text("hello world")
    assert len(vec) == 8
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_hash_provider_is_deterministic() -> None:
    p = HashEmbeddingProvider(dimension=16)
    v1 = p.embed_text("consistent input")
    v2 = p.embed_text("consistent input")
    assert v1 == v2


def test_hash_provider_rejects_empty_text() -> None:
    p = HashEmbeddingProvider(dimension=4)
    with pytest.raises(EmbeddingPermanentError):
        p.embed_text("")
    with pytest.raises(EmbeddingPermanentError):
        p.embed_text("   \n  ")


def test_hash_provider_rejects_non_string() -> None:
    p = HashEmbeddingProvider(dimension=4)
    with pytest.raises(EmbeddingPermanentError):
        p.embed_text(None)  # type: ignore[arg-type]


def test_hash_provider_batches() -> None:
    p = HashEmbeddingProvider(dimension=8)
    out = p.embed_texts(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 8 for v in out)


def test_hash_provider_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError):
        HashEmbeddingProvider(dimension=0)


def test_factory_returns_hash_provider_by_default() -> None:
    reset_provider_for_tests()
    p = get_embedding_provider()
    assert p.name == "hash"
    assert p.model == "hash-256"


def test_factory_set_for_tests_swaps_provider() -> None:
    class _Stub:
        name = "stub"
        model = "stub-1"
        dimension = 4

        def embed_text(self, text: str):
            return [0.0] * 4

        def embed_texts(self, texts):
            return [[0.0] * 4 for _ in texts]

    try:
        set_provider_for_tests(_Stub())  # type: ignore[arg-type]
        assert get_embedding_provider().name == "stub"
    finally:
        reset_provider_for_tests()
    assert get_embedding_provider().name == "hash"

# ---------------------------------------------------------------------------
# OpenAI-compatible provider: dimensions parameter gating
# ---------------------------------------------------------------------------


def _capture_post(monkeypatch: pytest.MonkeyPatch, dimension: int = 4) -> dict:
    """Replace httpx.post with a stub; return the captured payload box."""

    import app.embedding.openai_provider as mod

    box: dict = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        box["last"] = json
        n = len((json or {}).get("input", []))
        return httpx.Response(
            200,
            json={"data": [
                {"embedding": [0.1] * dimension, "index": i} for i in range(n)
            ]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(mod.httpx, "post", _fake_post)
    return box


def test_openai_provider_sends_dimensions_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.embedding.openai_provider import OpenAIEmbeddingProvider

    box = _capture_post(monkeypatch, dimension=4)
    p = OpenAIEmbeddingProvider(
        api_key="k",
        base_url="http://mock",
        model="m",
        dimension=4,
        dimensions=4,
    )
    p.embed_text("hello")
    assert box["last"]["dimensions"] == 4


def test_openai_provider_can_omit_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.embedding.openai_provider import OpenAIEmbeddingProvider

    box = _capture_post(monkeypatch, dimension=4)
    p = OpenAIEmbeddingProvider(
        api_key="k",
        base_url="http://mock",
        model="m",
        dimension=4,
        dimensions=None,
    )
    p.embed_text("hello")
    assert "dimensions" not in box["last"]


def test_factory_omits_dimensions_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.embedding.factory import reset_provider_for_tests
    from app.config import get_settings

    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "http://mock")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "4")
    monkeypatch.setenv("EMBEDDING_OPENAI_SEND_DIMENSIONS", "0")
    get_settings.cache_clear()
    reset_provider_for_tests()
    try:
        p = get_embedding_provider()
        assert p._dimensions is None
    finally:
        reset_provider_for_tests()
        get_settings.cache_clear()


def test_openai_provider_throttles_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.embedding.openai_provider import OpenAIEmbeddingProvider

    sleeps: list[float] = []

    class _FakeTime:
        def __init__(self):
            self.now = 1000.0

        def monotonic(self):
            return self.now

    fake = _FakeTime()
    monkeypatch.setattr(
        "app.embedding.openai_provider.time.monotonic", fake.monotonic
    )
    monkeypatch.setattr(
        "app.embedding.openai_provider.time.sleep",
        lambda s: sleeps.append(s),
    )

    box = _capture_post(monkeypatch, dimension=4)
    p = OpenAIEmbeddingProvider(
        api_key="k",
        base_url="http://mock",
        model="m",
        dimension=4,
        request_interval_seconds=0.6,
    )
    p.embed_texts(["a", "b"])  # one request
    p.embed_texts(["c"])       # second request — now later than interval
    fake.now += 1.0
    p.embed_texts(["d"])       # interval already elapsed — no sleep

    assert len(box["last"]["input"]) == 1  # last payload only
    assert sleeps == [0.6]  # only the too-soon second call slept


# ---------------------------------------------------------------------------
# RPM/TPM token-bucket limiter (Phase 12)
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


def test_limiter_unlimited_never_sleeps(monkeypatch: pytest.MonkeyPatch):
    from app.embedding.openai_provider import _RpmTpmLimiter

    limiter = _RpmTpmLimiter(0, 0)
    assert limiter.estimate_tokens(["abc"]) > 0
    limiter.acquire(100)  # must be a no-op without a clock


def test_limiter_rpm_forces_sleep_at_window_edge(monkeypatch: pytest.MonkeyPatch):
    from app.embedding.openai_provider import _RpmTpmLimiter

    clock = _FakeClock()
    limiter = _RpmTpmLimiter(rpm=2, tpm=0)
    sleeps: list[float] = []
    monkeypatch.setattr(
        "app.embedding.openai_provider.time.monotonic",
        lambda: clock.now,
    )
    monkeypatch.setattr(
        "app.embedding.openai_provider.time.sleep",
        lambda s: sleeps.append(s),
    )

    limiter.acquire(10)   # request 1
    limiter.acquire(10)   # request 2 — window full (rpm=2)
    clock.now += 61       # window slides — old events expire
    limiter.acquire(10)
    assert sleeps == []   # expiry alone freed the slot, no sleep needed


def test_limiter_tpm_blocks_over_budget(monkeypatch: pytest.MonkeyPatch):
    from app.embedding.openai_provider import _RpmTpmLimiter

    clock = _FakeClock()
    limiter = _RpmTpmLimiter(rpm=0, tpm=100)
    sleeps: list[float] = []
    monkeypatch.setattr(
        "app.embedding.openai_provider.time.monotonic",
        lambda: clock.now,
    )
    monkeypatch.setattr(
        "app.embedding.openai_provider.time.sleep",
        lambda s: (sleeps.append(s), setattr(clock, "now", clock.now + 60)),
    )

    limiter.acquire(90)   # near the budget
    limiter.acquire(90)   # would exceed 100 → must sleep a full window
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(60.0)


def test_limiter_estimate_is_conservative():
    from app.embedding.openai_provider import _RpmTpmLimiter

    est = _RpmTpmLimiter.estimate_tokens(["x" * 500])
    assert est >= 250  # ~0.5 token/char plus framing — never under-counts
