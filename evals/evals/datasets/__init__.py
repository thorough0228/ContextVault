"""Dataset package: loading golden labels; synthesis lives in synthesize.py."""

from evals.datasets.loader import (
    DatasetError,
    QACase,
    RetrievalCase,
    audit_against_corpus,
    load_out_of_kb,
    load_qa_cases,
    load_retrieval_cases,
    sample_cases,
)

__all__ = [
    "DatasetError",
    "QACase",
    "RetrievalCase",
    "audit_against_corpus",
    "load_out_of_kb",
    "load_qa_cases",
    "load_retrieval_cases",
    "sample_cases",
]
