"""Tests for the in-process local embedding provider."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from app.embedding.base import EmbeddingPermanentError
from app.embedding.local_provider import LocalEmbeddingProvider


class _StubModel:
    """Mimics SentenceTransformer without loading any real weights."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.sentences: list[str] = []
        self.last_normalize: bool | None = None

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, sentences, normalize_embeddings=False, convert_to_numpy=True):
        self.sentences.extend(sentences)
        self.last_normalize = normalize_embeddings
        base = np.tile(
            np.arange(self.dim, dtype="float32") + 1.0, (len(sentences), 1)
        )
        if normalize_embeddings:
            base = base / np.linalg.norm(base, axis=1, keepdims=True)
        return base


def _provider(**overrides) -> LocalEmbeddingProvider:
    model = overrides.pop("model", None) or _StubModel(dim=4)
    kwargs = {
        "model_name": "stub-model",
        "dimension": model.dim,
        "query_instruction": "",
        "model": model,
    }
    kwargs.update(overrides)
    return LocalEmbeddingProvider(**kwargs)


def test_query_gets_instruction_prefix() -> None:
    model = _StubModel(dim=4)
    p = _provider(model=model, query_instruction="指令：")
    p.embed_text("hello")
    assert model.sentences == ["指令：hello"]


def test_indexed_texts_get_no_prefix() -> None:
    model = _StubModel(dim=4)
    p = _provider(model=model, query_instruction="指令：")
    p.embed_texts(["a", "b"])
    assert model.sentences == ["a", "b"]


def test_output_is_normalized() -> None:
    model = _StubModel(dim=4)
    p = _provider(model=model)
    vec = p.embed_text("hello")
    assert len(vec) == 4
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-5
    assert model.last_normalize is True


def test_empty_batch_short_circuits() -> None:
    model = _StubModel(dim=4)
    p = _provider(model=model)
    assert p.embed_texts([]) == []
    assert model.sentences == []


def test_dimension_mismatch_fails_fast() -> None:
    with pytest.raises(EmbeddingPermanentError, match="dims"):
        LocalEmbeddingProvider(
            model_name="stub-model", dimension=8, model=_StubModel(dim=4)
        )


def test_non_positive_dimension_fails_fast() -> None:
    with pytest.raises(EmbeddingPermanentError):
        LocalEmbeddingProvider(model_name="stub-model", dimension=0, model=_StubModel(dim=4))


class _HalfRecordingStub(_StubModel):
    """Stub that records half() calls and simulates a device."""

    def __init__(self, dim: int = 4, device_type: str = "cpu"):
        super().__init__(dim=dim)
        self.device_type = device_type
        self.half_called = False

    @property
    def device(self):
        class _Dev:
            def __init__(self, t: str):
                self.type = t

        return _Dev(self.device_type)

    def half(self):
        self.half_called = True
        return self


def test_fp16_on_cuda_casts_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(get_device_capability=lambda: (7, 5))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    model = _HalfRecordingStub(dim=4, device_type="cuda")
    LocalEmbeddingProvider(
        model_name="stub-model", dimension=4, fp16=True, model=model
    )
    assert model.half_called is True


def test_fp16_on_cpu_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _HalfRecordingStub(dim=4, device_type="cpu")
    LocalEmbeddingProvider(
        model_name="stub-model", dimension=4, fp16=True, model=model
    )
    assert model.half_called is False


def test_fp16_on_old_gpu_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(get_device_capability=lambda: (6, 1))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    model = _HalfRecordingStub(dim=4, device_type="cuda")
    LocalEmbeddingProvider(
        model_name="stub-model", dimension=4, fp16=True, model=model
    )
    assert model.half_called is False


def test_fp16_defaults_off() -> None:
    model = _HalfRecordingStub(dim=4, device_type="cuda")
    LocalEmbeddingProvider(model_name="stub-model", dimension=4, model=model)
    assert model.half_called is False


def test_missing_dependency_gives_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate sentence-transformers not being installed: importing a
    # module whose sys.modules entry is None raises ImportError.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingPermanentError, match="pip install"):
        LocalEmbeddingProvider(model_name="any", dimension=4)
