"""Tests for the MiniMax embedding provider (HTTP-level, mocked)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.embedding.base import (
    EmbeddingPermanentError,
    EmbeddingTransientError,
)
from app.embedding.minimax_provider import MiniMaxEmbeddingProvider


def _make_provider(handler) -> MiniMaxEmbeddingProvider:
    transport = httpx.MockTransport(handler)
    return MiniMaxEmbeddingProvider(
        api_key="test-key",
        group_id="test-group",
        base_url="https://minimax.test",
        dimension=4,
        transport=transport,
    )


def _response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def test_embed_text_uses_query_purpose_and_parses_vector() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return _response(
            {"vectors": [[0.1, 0.2, 0.3, 0.4]], "base_resp": {"status_code": 0}}
        )

    out = _make_provider(handler).embed_text("hello")

    assert out == [0.1, 0.2, 0.3, 0.4]
    assert captured["url"].startswith("https://minimax.test/v1/embeddings")
    assert "GroupId=test-group" in captured["url"]
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["type"] == "query"
    assert captured["body"]["texts"] == ["hello"]


def test_embed_texts_uses_db_purpose() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return _response(
            {
                "vectors": [[0.0, 0.0, 0.0, 0.0] for _ in body["texts"]],
                "base_resp": {"status_code": 0},
            }
        )

    out = _make_provider(handler).embed_texts(["a", "b"])
    assert len(out) == 2
    # batch call must target the indexing space, not the query space
    assert out == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_dimension_mismatch_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(
            {"vectors": [[0.1, 0.2]], "base_resp": {"status_code": 0}}
        )

    with pytest.raises(EmbeddingPermanentError, match="dims"):
        _make_provider(handler).embed_text("hello")


def test_rate_limit_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(
            {"base_resp": {"status_code": 1002, "status_msg": "rate limited"}}
        )

    with pytest.raises(EmbeddingTransientError):
        _make_provider(handler).embed_text("hello")


def test_auth_failure_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response({}, status=401)

    with pytest.raises(EmbeddingPermanentError, match="auth"):
        _make_provider(handler).embed_text("hello")


def test_http_500_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(EmbeddingTransientError):
        _make_provider(handler).embed_text("hello")


def test_missing_key_or_group_is_permanent() -> None:
    with pytest.raises(EmbeddingPermanentError):
        MiniMaxEmbeddingProvider(api_key="", group_id="g", dimension=4)
    with pytest.raises(EmbeddingPermanentError):
        MiniMaxEmbeddingProvider(api_key="k", group_id="", dimension=4)


def test_empty_batch_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not hit the network for an empty batch")

    assert _make_provider(handler).embed_texts([]) == []
