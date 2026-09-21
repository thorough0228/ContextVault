"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { apiClient, ApiError } from "@/lib/api-client";
import { useSession } from "@/lib/auth-store";
import HealthBadge from "@/components/health-badge";
import RagCard from "@/components/rag-card";
import type { Rag, RagStatus } from "@contextvault/shared";

export default function DashboardPage() {
  const router = useRouter();
  const { token, hydrated, user } = useSession();
  const [rags, setRags] = useState<Rag[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await apiClient.listRags();
      setRags(list.items);
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "failed to load RAGs";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    if (!token) {
      router.replace("/login");
      return;
    }
    refresh();
  }, [hydrated, token, refresh, router]);

  if (!hydrated || !user) {
    return (
      <p className="text-sm text-slate-500">Redirecting to login…</p>
    );
  }

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <p className="text-sm text-slate-500">
            Welcome, <span className="font-mono">{user.email}</span>
          </p>
        </div>
        <Link
          href="/rags/new"
          className="rounded bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          + New RAG
        </Link>
      </header>

      <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
          API Health
        </h2>
        <HealthBadge />
      </div>

      <div>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          My Knowledge Bases
        </h2>

        {loading ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : error ? (
          <p className="rounded border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
            {error}
          </p>
        ) : rags.length === 0 ? (
          <div
            className="rounded-lg border border-dashed border-slate-300 bg-white p-10 text-center dark:border-slate-700 dark:bg-slate-900"
            data-testid="empty-state"
          >
            <p className="text-lg font-medium">你还没有知识库，请先创建一个知识库。</p>
            <p className="mt-1 text-sm text-slate-500">
              A knowledge base lets you upload documents and query them with
              a retrieval-augmented chat.
            </p>
            <Link
              href="/rags/new"
              className="mt-4 inline-block rounded bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
            >
              + Create your first RAG
            </Link>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            {rags.map((rag) => (
              <RagCard
                key={rag.id}
                rag={rag}
                onChanged={refresh}
                onDeleted={refresh}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

export type { RagStatus };