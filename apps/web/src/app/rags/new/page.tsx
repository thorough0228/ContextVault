"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { apiClient, ApiError } from "@/lib/api-client";
import { useSession } from "@/lib/auth-store";

export default function NewRagPage() {
  const router = useRouter();
  const { hydrated, token } = useSession();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!hydrated) return;
    if (!token) router.replace("/login");
  }, [hydrated, token, router]);

  if (!hydrated || !token) {
    return <p className="text-sm text-slate-500">Redirecting to login…</p>;
  }

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const created = await apiClient.createRag({
        name: name.trim(),
        description: description.trim(),
      });
      router.push(`/rags/${created.id}`);
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "create RAG failed";
      setError(msg);
      setBusy(false);
    }
  }

  return (
    <section className="mx-auto max-w-md space-y-4">
      <h1 className="text-2xl font-semibold">New knowledge base</h1>
      <p className="text-sm text-slate-500">
        Give it a name and an optional description. You can edit both later.
      </p>
      <form className="space-y-3" onSubmit={onSubmit}>
        <label className="block">
          <span className="text-sm text-slate-700 dark:text-slate-200">Name</span>
          <input
            type="text"
            required
            minLength={1}
            maxLength={128}
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
            placeholder="Customer support docs"
          />
        </label>
        <label className="block">
          <span className="text-sm text-slate-700 dark:text-slate-200">
            Description
          </span>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            maxLength={4096}
            rows={4}
            className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
            placeholder="What kind of documents will live here?"
          />
        </label>
        {error ? (
          <p className="rounded border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
            {error}
          </p>
        ) : null}
        <div className="flex gap-2">
          <button
            type="submit"
            disabled={busy || name.trim().length === 0}
            className="rounded bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {busy ? "Creating…" : "Create"}
          </button>
          <button
            type="button"
            onClick={() => router.back()}
            className="rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Cancel
          </button>
        </div>
      </form>
    </section>
  );
}