"use client";

import { useCallback, useState } from "react";
import { apiClient, ApiError } from "@/lib/api-client";
import type { SearchHit, SearchResponse } from "@contextvault/shared";

interface Props {
  ragId: string;
}

function formatScore(score: number): string {
  return score.toFixed(3);
}

export default function SearchPanel({ ragId }: Props) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);

  const onSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!query.trim()) {
        setError("query must not be empty");
        return;
      }
      setBusy(true);
      setError(null);
      try {
        const res = await apiClient.searchRags(ragId, {
          query: query.trim(),
          top_k: topK,
        });
        setResult(res);
      } catch (err) {
        const msg = err instanceof ApiError ? err.message : "search failed";
        setError(msg);
      } finally {
        setBusy(false);
      }
    },
    [query, topK, ragId],
  );

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">Search</h2>
        <p className="text-xs text-slate-500">
          The RAG is isolated — the server only sees chunks owned by you
          and this RAG.
          {result ? (
            <>
              {" "}Embedded with{" "}
              <span className="font-mono">{result.embedding_model}</span> (
              {result.embedding_dimension} dimensions).
            </>
          ) : null}
        </p>
      </div>

      <form className="flex flex-wrap items-end gap-3" onSubmit={onSubmit}>
        <label className="flex-1 min-w-[12rem]">
          <span className="text-sm text-slate-700 dark:text-slate-200">
            Query
          </span>
          <input
            type="text"
            required
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What does this RAG say about ...?"
            className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
          />
        </label>
        <label className="w-24">
          <span className="text-sm text-slate-700 dark:text-slate-200">top_k</span>
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value) || 5)}
            className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
          />
        </label>
        <button
          type="submit"
          disabled={busy}
          className="rounded bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      {error ? (
        <p className="rounded border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
          {error}
        </p>
      ) : null}

      {result ? (
        result.hits.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900">
            No matching chunks in this RAG.
          </div>
        ) : (
          <ul className="space-y-3">
            {result.hits.map((hit) => (
              <Hit key={hit.chunk_id} hit={hit} />
            ))}
          </ul>
        )
      ) : null}
    </section>
  );
}

function Hit({ hit }: { hit: SearchHit }) {
  return (
    <li className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span className="font-mono">{hit.filename} · page {hit.page_number}</span>
        <span className="rounded border border-slate-300 px-2 py-0.5 font-mono text-[10px] uppercase dark:border-slate-700">
          score {formatScore(hit.score)}
        </span>
      </div>
      <p className="mt-2 whitespace-pre-wrap text-sm">{hit.chunk_text}</p>
    </li>
  );
}