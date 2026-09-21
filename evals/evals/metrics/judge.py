"""DeepEval judge adapters over the app's provider abstractions.

AGENTS.md rule 8 (all external AI providers behind the abstract
interface) applies to the judge too: ``ProviderJudgeLLM`` delegates to
``get_llm_provider()`` (MiniMax-M3 via the OpenAI-compatible provider
in the default .env setup) and ``ProviderEmbeddingModel`` delegates to
``get_embedding_provider()`` (local bge). Swapping the judge to a
stronger model later is a pure .env change — no code edits.

DeepEval's sync ``generate`` may be called from inside a running event
loop (our suites are async); calling ``asyncio.run`` there would raise.
The wrapper runs the provider call on a worker thread with its own
loop instead, which is safe from any context.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

_JUDGE_THREAD = ThreadPoolExecutor(max_workers=2, thread_name_prefix="eval-judge")


def _chat_on_thread(coro_factory, prompt: str) -> str:
    """Run the async call in a fresh loop on a worker thread."""

    return _JUDGE_THREAD.submit(asyncio.run, coro_factory(prompt)).result()


def _build_judge_provider():
    """Judge-side provider instance: same OpenAILLMProvider abstraction,
    but a doubled output budget.

    DeepEval templates make MiniMax-M3 reason at length; with the chat
    default (8192 output tokens) the <think> block can exhaust the
    budget and the visible content comes back empty, which DeepEval
    rejects as invalid JSON. 16384 stays far below MiniMax's hard cap.
    """

    from app.config import get_settings
    from app.llm import get_llm_provider
    from app.llm.openai_provider import OpenAILLMProvider

    settings = get_settings()
    if settings.llm_provider != "openai":
        return get_llm_provider()  # hash / scripted providers: as-is
    return OpenAILLMProvider(
        api_key=settings.llm_openai_api_key,
        base_url=settings.llm_openai_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_context_tokens=max(16384, settings.llm_max_context_tokens * 2),
        strip_think=True,
    )


def build_judge_llm():
    """DeepEval-compatible judge LLM backed by the app's LLMProvider."""

    from deepeval.models import DeepEvalBaseLLM

    provider = _build_judge_provider()

    async def _call(prompt: str) -> str:
        # Empty visible content = think-block ate the budget; retrying
        # usually lands a complete answer on the second attempt.
        JUDGE_CALLS["count"] = JUDGE_CALLS.get("count", 0) + 1
        out = ""
        for _attempt in range(3):
            out = await provider.chat([{"role": "user", "content": prompt}])
            if out.strip():
                return out
            JUDGE_CALLS["empty"] = JUDGE_CALLS.get("empty", 0) + 1
        return out

    class ProviderJudgeLLM(DeepEvalBaseLLM):
        def load_model(self):
            return provider

        def generate(self, prompt: str, schema=None, **kwargs) -> str:  # noqa: ARG002
            return _chat_on_thread(_call, prompt)

        async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:  # noqa: ARG002
            return await _call(prompt)

        def get_model_name(self) -> str:
            return provider.model

    return ProviderJudgeLLM()


def build_judge_embedding():
    """DeepEval-compatible embedding model backed by EmbeddingProvider."""

    from deepeval.models import DeepEvalBaseEmbeddingModel

    from app.embedding import get_embedding_provider

    provider = get_embedding_provider()

    class ProviderEmbeddingModel(DeepEvalBaseEmbeddingModel):
        def load_model(self):
            return provider

        def embed_text(self, text: str) -> list[float]:
            return provider.embed_text(text)

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return provider.embed_texts(texts)

        async def a_embed_text(self, text: str) -> list[float]:
            return provider.embed_text(text)

        async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
            return provider.embed_texts(texts)

        def get_model_name(self) -> str:
            return provider.model

    return ProviderEmbeddingModel()


# Judge call budget bookkeeping — the suites report it so token spend
# stays visible without instrumenting the provider.
JUDGE_CALLS = {"count": 0}
