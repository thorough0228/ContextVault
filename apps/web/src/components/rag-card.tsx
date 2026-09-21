"use client";

import Link from "next/link";
import { useState } from "react";
import { apiClient, ApiError } from "@/lib/api-client";
import type { Rag } from "@contextvault/shared";

interface Props {
  rag: Rag;
  onChanged: () => void;
  onDeleted: () => void;
}

const STATUS_STYLES: Record<Rag["status"], string> = {
  ACTIVE: "bg-emerald-100 text-emerald-800 border-emerald-300",
  PROCESSING: "bg-amber-100 text-amber-800 border-amber-300",
  ERROR: "bg-rose-100 text-rose-800 border-rose-300",
};

export default function RagCard({ rag, onChanged, onDeleted }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onDelete() {
    if (!confirm(`Delete "${rag.name}"? This cannot be undone.`)) return;
    setBusy(true);
    setError(null);
    try {
      await apiClient.deleteRag(rag.id);
      onDeleted();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "delete failed";
      setError(msg);
      setBusy(false);
    }
  }

  const created = new Date(rag.created_at);
  const createdLabel = Number.isFinite(created.getTime())
    ? created.toLocaleString()
    : rag.created_at;

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-lg font-semibold">{rag.name}</h3>
          {rag.description ? (
            <p className="mt-1 text-sm text-slate-500">{rag.description}</p>
          ) : null}
        </div>
        <span
          className={`shrink-0 rounded border px-2 py-0.5 text-xs font-medium uppercase tracking-wide ${STATUS_STYLES[rag.status]}`}
        >
          {rag.status}
        </span>
      </div>
      <p className="mt-3 text-xs text-slate-500">
        Created {createdLabel}
      </p>
      {error ? (
        <p className="mt-2 rounded border border-rose-300 bg-rose-50 px-2 py-1 text-xs text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
          {error}
        </p>
      ) : null}
      <div className="mt-4 flex flex-wrap gap-2">
        <Link
          href={`/rags/${rag.id}`}
          className="rounded bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-700"
        >
          Enter
        </Link>
        <button
          type="button"
          onClick={onDelete}
          disabled={busy}
          className="rounded border border-rose-300 px-3 py-1.5 text-xs text-rose-700 hover:bg-rose-50 disabled:opacity-50 dark:text-rose-200 dark:hover:bg-rose-950/40"
        >
          {busy ? "Deleting…" : "Delete"}
        </button>
      </div>
    </article>
  );
}