"""Full-corpus rebuild OUTSIDE the Celery worker (Phase 14 experiment).

The Windows solo worker stalls silently on semantic chunking (thousands
of sentence-embedding calls per document — see problems/), so this
script performs the whole pipeline in-process, one document at a time,
with per-batch retries and resumability:

    parse → chunk (semantic | recursive | fixed) →
    chunk embeddings (throttled, batched) → replace chunks + vectors.

Usage:
    python scripts/rebuild_embeddings.py --chunker semantic
    python scripts/rebuild_embeddings.py --chunker fixed
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

import psycopg2
import psycopg2.extras

from app.config import get_settings
from app.embedding.factory import get_embedding_provider
from app.ingest import parse_document
from app.ingest.chunker import FixedWindowChunker, RecursiveChunker
from app.ingest.semantic_chunker import SemanticChunker
from app.services.tokenization import tokenize_to_string


def build_chunker(kind: str, settings):
    if kind == "semantic":
        return SemanticChunker(
            size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
            percentile=settings.chunker_semantic_percentile,
            min_chars=settings.chunker_semantic_min_chars,
        )
    if kind == "recursive":
        return RecursiveChunker(
            size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
    return FixedWindowChunker(
        size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )


def embed_all(provider, texts) -> list:
    out: list = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk_texts = texts[i : i + batch]
        for attempt in range(5):
            try:
                out.extend(provider.embed_texts(chunk_texts))
                break
            except Exception as exc:  # noqa: BLE001
                wait = 3 * (attempt + 1)
                print(f"    embed retry {attempt+1} in {wait}s: {str(exc)[:80]}")
                time.sleep(wait)
        else:
            raise RuntimeError("embedding permanently failed")
        time.sleep(0.2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunker", default="semantic",
                    choices=["semantic", "recursive", "fixed"])
    args = ap.parse_args()

    settings = get_settings()
    provider = get_embedding_provider()
    conn = psycopg2.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        dbname=settings.postgres_db, user=settings.postgres_user,
        password=settings.postgres_password,
    )
    chunker = build_chunker(args.chunker, settings)
    storage = __import__("app.storage", fromlist=["get_storage"]).get_storage()

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT id, filename FROM documents WHERE status = 'READY'")
        docs = cur.fetchall()
        print(f"{len(docs)} documents to rebuild with chunker={args.chunker}")

        for doc_id, filename in docs:
            t0 = time.time()
            cur.execute(
                "SELECT storage_key, file_type, rag_id FROM documents WHERE id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
            if row is None:
                continue
            storage_key, file_type, rag_id = row

            try:
                downloaded = storage.download(key=storage_key)
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP {filename}: storage {str(exc)[:80]}")
                continue

            try:
                pages = parse_document(
                    file_type=file_type, stream=io.BytesIO(downloaded.body)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP {filename}: parse {str(exc)[:80]}")
                continue

            chunks = chunker.chunk_pages(
                pages, document_id=doc_id, rag_id=rag_id
            )
            if not chunks:
                print(f"  SKIP {filename}: zero chunks")
                continue

            texts = [c.text for c in chunks]
            vectors = embed_all(provider, texts)

            cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %s", (doc_id,)
            )
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            conn.commit()

            for c, vec in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO chunks (id, document_id, rag_id, page_number,"
                    " chunk_index, text) VALUES (gen_random_uuid(), %s, %s,"
                    " %s, %s, %s)",
                    (doc_id, rag_id, c.page_number, c.chunk_index, c.text),
                )
                cur.execute(
                    "INSERT INTO document_chunks (id, document_id, rag_id,"
                    " chunk_index, chunk_text, page_number, tokenized,"
                    " embedding) VALUES (gen_random_uuid(), %s, %s, %s, %s,"
                    " %s, %s, %s::vector)",
                    (
                        doc_id, rag_id, c.chunk_index, c.text, c.page_number,
                        tokenize_to_string(c.text),
                        "[" + ",".join(repr(float(x)) for x in vec) + "]",
                    ),
                )
            conn.commit()
            print(
                f"  {filename}: {len(chunks)} chunks ({time.time() - t0:.0f}s)",
                flush=True,
            )
    conn.close()
    print("REBUILD DONE")


if __name__ == "__main__":
    main()
