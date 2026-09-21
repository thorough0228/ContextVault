"""Unit tests for corpus building + golden dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.datasets.loader import (
    DatasetError,
    QACase,
    RetrievalCase,
    load_qa_cases,
    load_retrieval_cases,
    sample_cases,
)
from evals.harness.alignment import CorpusIndex
from evals.harness.corpus import CORPUS_SOURCES, build_corpus, sample_records

pytestmark = pytest.mark.offline


class TestSourceLoaders:
    def test_all_three_sources_load_nonempty(self):
        for name, loader in CORPUS_SOURCES.items():
            records = loader()
            assert records, f"{name} loaded zero records"
            assert all(set(r) >= {"id", "text"} for r in records[:5])

    def test_table_tennis_ids_are_stable_chunk_ids(self):
        records = CORPUS_SOURCES["table_tennis"]()
        assert records[0]["id"] == "chunk_0"
        assert len(records) > 1000  # full corpus available for sampling


class TestSampling:
    def test_seed_is_reproducible_and_order_preserved(self):
        records = CORPUS_SOURCES["table_tennis"]()
        a = sample_records(records, 50, seed=7)
        b = sample_records(records, 50, seed=7)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        # Order follows the source corpus order.
        positions = [records.index(r) for r in a]
        assert positions == sorted(positions)

    def test_size_capped_at_source(self):
        records = CORPUS_SOURCES["marvel"]()
        assert len(sample_records(records, 10_000, seed=1)) == len(records)


class TestBuildCorpus:
    def test_jsonl_serialization_matches_alignment_index(self, tmp_path):
        spec = build_corpus("marvel", size=10, seed=3, out_dir=tmp_path)
        assert spec.jsonl_path.is_file()
        lines = spec.jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 10
        # The index over the spec's records reproduces the uploaded file.
        on_disk = [json.loads(line) for line in lines]
        assert on_disk == spec.records
        index = CorpusIndex(spec.records)
        assert index.record_ids == {r["id"] for r in spec.records}

    def test_unknown_corpus_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown corpus"):
            build_corpus("nope", size=5, seed=1, out_dir=tmp_path)

    def test_chunk_estimate_positive_and_grows_with_corpus(self, tmp_path):
        small = build_corpus("marvel", size=5, seed=3, out_dir=tmp_path / "a")
        bigger = build_corpus("marvel", size=30, seed=3, out_dir=tmp_path / "b")
        assert small.meta["estimated_chunks"] > 0
        assert (
            bigger.meta["estimated_chunks"] >= small.meta["estimated_chunks"]
        )


class TestLoader:
    def _write(self, path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_retrieval_roundtrip(self, tmp_path):
        self._write(
            tmp_path / "tt" / "retrieval.jsonl",
            [
                {"id": "q1", "query": "胶皮怎么选", "positive_ids": ["chunk_0"]},
                {"id": "q2", "query": "底板推荐", "positive_ids": ["chunk_1", "chunk_2"], "notes": "x"},
            ],
        )
        cases = load_retrieval_cases("tt", datasets_dir=tmp_path)
        assert cases[0] == RetrievalCase("q1", "胶皮怎么选", frozenset({"chunk_0"}), "")
        assert cases[1].positive_ids == frozenset({"chunk_1", "chunk_2"})

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="missing"):
            load_retrieval_cases("ghost", datasets_dir=tmp_path)

    def test_empty_positive_ids_rejected(self, tmp_path):
        self._write(
            tmp_path / "tt" / "retrieval.jsonl",
            [{"id": "q1", "query": "q", "positive_ids": []}],
        )
        with pytest.raises(DatasetError, match="positive_ids"):
            load_retrieval_cases("tt", datasets_dir=tmp_path)

    def test_duplicate_id_rejected(self, tmp_path):
        self._write(
            tmp_path / "tt" / "retrieval.jsonl",
            [
                {"id": "q1", "query": "a", "positive_ids": ["x"]},
                {"id": "q1", "query": "b", "positive_ids": ["y"]},
            ],
        )
        with pytest.raises(DatasetError, match="duplicate"):
            load_retrieval_cases("tt", datasets_dir=tmp_path)

    def test_qa_roundtrip(self, tmp_path):
        self._write(
            tmp_path / "tt" / "qa.jsonl",
            [{"id": "qa1", "question": "Q", "ground_truth": "A", "source_ids": ["chunk_0"]}],
        )
        cases = load_qa_cases("tt", datasets_dir=tmp_path)
        assert isinstance(cases[0], QACase)
        assert cases[0].category == "factual"


class TestSamplingCases:
    def test_sample_cases_stable_and_capped(self):
        cases = [RetrievalCase(f"q{i}", f"query {i}", frozenset({f"c{i}"})) for i in range(10)]
        picked = sample_cases(cases, 4, seed=5)
        again = sample_cases(cases, 4, seed=5)
        assert [c.case_id for c in picked] == [c.case_id for c in again]
        assert len(picked) == 4
        assert len(sample_cases(cases, 99, seed=1)) == 10
