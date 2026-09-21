"""End-to-end smoke of the sqlite eval environment itself.

Uses the hash embedding + a tiny marvel corpus so the whole chain —
env vars -> cached singletons reset -> app boot -> eager worker ->
upload -> READY -> search -> NDJSON chat — is exercised in seconds
without loading the bge model or touching the network.
"""

from __future__ import annotations

import pytest

from evals.harness.environment import SqliteEvalEnv, chat_turn, run_search

pytestmark = pytest.mark.offline


async def test_sqlite_env_boots_ingests_and_serves():
    async with SqliteEvalEnv(
        corpora=["marvel"],
        embedding="hash",
        llm="hash",
        corpus_size=8,
        seed=11,
        ingest_timeout_s=60.0,
    ) as env:
        assert set(env.rag_ids) == {"marvel"}
        spec = env.corpora["marvel"]
        assert spec.meta["records"] == 8

        rag_id = env.rag_ids["marvel"]

        # Search returns real hits from the ingested corpus.
        hits = await run_search(
            env.client, env.auth_headers, rag_id=rag_id,
            query="captain america", top_k=5,
        )
        assert 1 <= len(hits) <= 5
        assert {"chunk_id", "chunk_text", "score", "page_number"} <= set(hits[0])

        # Chat runs the full streaming pipeline with the hash LLM.
        turn = await chat_turn(
            env.client, env.auth_headers, rag_id=rag_id, message="hello?"
        )
        assert turn["terminal"] == "done"
        assert turn["answer"]  # hash LLM still emits deterministic text
        assert turn["citations"], "citation event must precede tokens"

        types = [e["type"] for e in turn["events"]]
        assert types[0] == "citation"
        assert types[-1] == "done"
        assert "token" in types


async def test_sqlite_env_chunk_guard_fires():
    with pytest.raises(AssertionError, match="sqlite guard"):
        async with SqliteEvalEnv(
            corpora=["marvel"],
            embedding="hash",
            llm="hash",
            corpus_size=150,          # would re-chunk well past the cap
            chunk_guard=10,           # deliberately tiny to force the guard
            ingest_timeout_s=60.0,
        ):
            pass
