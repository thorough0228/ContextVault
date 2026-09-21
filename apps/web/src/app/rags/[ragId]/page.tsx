"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { apiClient, ApiError } from "@/lib/api-client";
import { useSession } from "@/lib/auth-store";
import DocumentsPanel from "@/components/documents-panel";
import SearchPanel from "@/components/search-panel";
import type { Rag, RagStatus } from "@contextvault/shared";

const STATUS_OPTIONS: RagStatus[] = ["ACTIVE", "PROCESSING", "ERROR"];

export default function RagDetailPage() {
  const router = useRouter();
  const params = useParams<{ ragId: string }>();
  const ragId = params?.ragId ?? "";
  const { hydrated, token } = useSession();

  const [rag, setRag] = useState<Rag | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [status, setStatus] = useState<RagStatus>("ACTIVE");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [notFound, setNotFound] = useState(false);

  const load = useCallback(async () => {
    if (!ragId) return;
    setError(null);
    setNotFound(false);
    try {
      const r = await apiClient.getRag(ragId);
      setRag(r);
      setName(r.name);
      setDescription(r.description);
      setStatus(r.status);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setNotFound(true);
        return;
      }
      const msg = err instanceof ApiError ? err.message : "failed to load RAG";
      setError(msg);
    }
  }, [ragId]);

  useEffect(() => {
    if (!hydrated) return;
    if (!token) {
      router.replace("/login");
      return;
    }
    load();
  }, [hydrated, token, load, router]);

  async function onSave(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setSaving(true);
    try {
      const updated = await apiClient.patchRag(ragId, {
        name: name.trim(),
        description: description.trim(),
        status,
      });
      setRag(updated);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "save failed";
      setError(msg);
    } finally {
      setSaving(false);
    }
  }

  async function onDelete() {
    if (!rag) return;
    if (!confirm(`Delete "${rag.name}"? This cannot be undone.`)) return;
    try {
      await apiClient.deleteRag(rag.id);
      router.push("/dashboard");
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "delete failed";
      setError(msg);
    }
  }

  if (!hydrated || !token) {
    return <p className="text-sm text-slate-500">Redirecting to login…</p>;
  }
  if (notFound) {
    return (
      <section className="space-y-4">
        <h1 className="text-2xl font-semibold">RAG not found</h1>
        <p className="text-sm text-slate-500">
          This knowledge base may have been deleted, or it does not belong to
          your account.
        </p>
        <Link
          href="/dashboard"
          className="inline-block rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          ← Back to dashboard
        </Link>
      </section>
    );
  }
  if (!rag) {
    return <p className="text-sm text-slate-500">Loading…</p>;
  }

  return (
    <section className="space-y-6">
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Link href="/dashboard" className="hover:text-brand-600">
          ← Dashboard
        </Link>
      </div>

      <header>
        <h1 className="text-2xl font-semibold">{rag.name}</h1>
        <p className="text-xs text-slate-500">id: {rag.id}</p>
      </header>

      <form
        className="space-y-3 rounded-lg border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900"
        onSubmit={onSave}
      >
        <label className="block">
          <span className="text-sm text-slate-700 dark:text-slate-200">
            Name
          </span>
          <input
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
          />
        </label>
        <label className="block">
          <span className="text-sm text-slate-700 dark:text-slate-200">
            Description
          </span>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
          />
        </label>
        <label className="block">
          <span className="text-sm text-slate-700 dark:text-slate-200">
            Status
          </span>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as RagStatus)}
            className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        {error ? (
          <p className="rounded border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
            {error}
          </p>
        ) : null}

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={saving}
            className="rounded bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {saving ? "Saving…" : "Save"}
          </button>
          <button
            type="button"
            onClick={onDelete}
            className="rounded border border-rose-300 px-4 py-2 text-sm text-rose-700 hover:bg-rose-50 dark:text-rose-200 dark:hover:bg-rose-950/40"
          >
            Delete
          </button>
        </div>
      </form>

      <div className="rounded-lg border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900">
        <DocumentsPanel ragId={rag.id} />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900">
        <SearchPanel ragId={rag.id} />
      </div>

      <div className="flex gap-3">
        <Link
          href={`/chat/${rag.id}`}
          className="rounded bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          Open chat →
        </Link>
      </div>
    </section>
  );
}