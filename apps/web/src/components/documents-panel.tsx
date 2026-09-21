"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient, ApiError } from "@/lib/api-client";
import type { Document, DocumentStatus } from "@contextvault/shared";

const STATUS_STYLES: Record<DocumentStatus, string> = {
  CREATED: "bg-slate-100 text-slate-700 border-slate-300",
  PROCESSING: "bg-amber-100 text-amber-800 border-amber-300",
  READY: "bg-emerald-100 text-emerald-800 border-emerald-300",
  FAILED: "bg-rose-100 text-rose-800 border-rose-300",
};

const STATUS_LABEL: Record<DocumentStatus, string> = {
  CREATED: "queued",
  PROCESSING: "processing…",
  READY: "ready",
  FAILED: "failed",
};

interface Props {
  ragId: string;
}

type QueueStatus = "waiting" | "uploading" | "queued" | "failed";

interface UploadQueueItem {
  name: string;
  size: number;
  sent: number;
  total: number;
  status: QueueStatus;
  error?: string;
}

const QUEUE_STATUS_LABEL: Record<QueueStatus, string> = {
  waiting: "waiting…",
  uploading: "uploading…",
  queued: "queued ✓",
  failed: "failed",
};

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export default function DocumentsPanel({ ragId }: Props) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [queue, setQueue] = useState<UploadQueueItem[]>([]);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await apiClient.listDocuments(ragId);
      setDocuments(list.items);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "failed to load documents";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [ragId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while any document is still in-flight. The poll is cheap —
  // one HTTP call per 2.5s — and stops as soon as the queue drains.
  useEffect(() => {
    const inFlight = documents.some(
      (d) => d.status === "CREATED" || d.status === "PROCESSING",
    );
    if (!inFlight) return;
    const timer = setInterval(refresh, 2500);
    return () => clearInterval(timer);
  }, [documents, refresh]);

  // Multi-file selection → files are uploaded one by one (queued).
  // Each successful upload enqueues a Celery ingestion task, so the
  // heavy processing is serialised by the worker automatically.
  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length === 0) return;
    if (inputRef.current) inputRef.current.value = "";
    setBusy(true);
    setUploadError(null);

    setQueue(
      files.map((f) => ({
        name: f.name,
        size: f.size,
        sent: 0,
        total: f.size,
        status: "waiting" as QueueStatus,
      })),
    );

    let failures = 0;
    for (let i = 0; i < files.length; i++) {
      setQueue((prev) =>
        prev.map((item, idx) =>
          idx === i ? { ...item, status: "uploading" as QueueStatus } : item,
        ),
      );
      try {
        await apiClient.uploadDocument(ragId, files[i], (sent, total) =>
          setQueue((prev) =>
            prev.map((item, idx) =>
              idx === i ? { ...item, sent, total } : item,
            ),
          ),
        );
        setQueue((prev) =>
          prev.map((item, idx) =>
            idx === i ? { ...item, status: "queued" as QueueStatus } : item,
          ),
        );
        // Pull the freshly created document into the list; the polling
        // effect keeps its status live while the worker processes it.
        await refresh();
      } catch (err) {
        failures++;
        const msg = err instanceof ApiError ? err.message : "upload failed";
        setQueue((prev) =>
          prev.map((item, idx) =>
            idx === i
              ? { ...item, status: "failed" as QueueStatus, error: msg }
              : item,
          ),
        );
      }
    }

    setBusy(false);
    if (failures > 0) {
      setUploadError(`${failures} of ${files.length} file(s) failed to upload.`);
    }
  }

  async function onDelete(doc: Document) {
    if (!confirm(`Delete "${doc.filename}"? This also removes its chunks.`)) return;
    try {
      await apiClient.deleteDocument(doc.id);
      await refresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "delete failed";
      setError(msg);
    }
  }

  // Phase 13: hash-detected update via PUT. Triggers a hidden file
  // picker per document; the server decides unchanged vs replace vs
  // bluegreen and reports the outcome.
  async function onUpdate(doc: Document, file: File) {
    setBusy(true);
    setError(null);
    try {
      const result = (await apiClient.updateDocument(doc.id, file)) as {
        updated?: boolean;
        reason?: string;
        strategy?: string;
        new_document_id?: string;
      };
      if (!result.updated) {
        setNotice(`"${doc.filename}" is unchanged — nothing to update.`);
      } else if (result.strategy === "bluegreen") {
        setNotice(
          `"${doc.filename}" updated (bluegreen, new id ${result.new_document_id}). Old version is kept for rollback.`,
        );
      } else {
        setNotice(`"${doc.filename}" is being re-ingested…`);
      }
      await refresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "update failed";
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  const totalBytes = documents.reduce((acc, d) => acc + d.file_size, 0);

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Documents</h2>
          <p className="text-xs text-slate-500">
            {documents.length} file{documents.length === 1 ? "" : "s"}
            {totalBytes > 0 ? ` · ${formatBytes(totalBytes)}` : ""}
          </p>
        </div>
        <label
          className={`inline-flex cursor-pointer items-center gap-2 rounded bg-brand-600 px-3 py-2 text-sm font-medium text-white hover:bg-brand-700 ${busy ? "pointer-events-none opacity-60" : ""}`}
        >
          {busy ? "Uploading…" : "+ Upload files (PDF / TXT / CSV / JSON)"}
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.csv,.json,.jsonl,application/pdf,text/plain,text/csv,application/json"
            className="hidden"
            onChange={onUpload}
            disabled={busy}
          />
        </label>
      </div>

      {queue.length > 0 ? (
        <div className="space-y-2 rounded border border-slate-200 bg-white p-3 text-xs dark:border-slate-800 dark:bg-slate-900">
          {queue.map((item, i) => (
            <div key={`${item.name}-${i}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate font-medium">{item.name}</span>
                <span
                  className={`shrink-0 font-mono ${
                    item.status === "failed"
                      ? "text-rose-600 dark:text-rose-300"
                      : item.status === "queued"
                        ? "text-emerald-600 dark:text-emerald-300"
                        : "text-slate-500"
                  }`}
                >
                  {QUEUE_STATUS_LABEL[item.status]}
                  {item.status === "uploading"
                    ? ` ${formatBytes(item.sent)} / ${formatBytes(item.total)}`
                    : ""}
                </span>
              </div>
              {item.status === "uploading" ? (
                <div className="mt-1 h-1 overflow-hidden rounded bg-slate-200 dark:bg-slate-800">
                  <div
                    className="h-full bg-brand-600 transition-all"
                    style={{
                      width: `${Math.round((item.sent / Math.max(item.total, 1)) * 100)}%`,
                    }}
                  />
                </div>
              ) : null}
              {item.status === "failed" && item.error ? (
                <p className="mt-0.5 text-rose-600 dark:text-rose-300">{item.error}</p>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      {uploadError ? (
        <p className="rounded border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
          {uploadError}
        </p>
      ) : null}

      {loading && documents.length === 0 ? (
        <p className="text-sm text-slate-500">Loading documents…</p>
      ) : error ? (
        <p className="rounded border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
          {error}
        </p>
      ) : notice ? (
        <p className="rounded border border-emerald-300 bg-emerald-50 px-3 py-2 text-sm text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-200">
          {notice}
        </p>
      ) : documents.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900">
          No documents yet. Upload PDF / TXT / CSV / JSON files to get started.
        </div>
      ) : (
        <ul className="divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white dark:divide-slate-800 dark:border-slate-800 dark:bg-slate-900">
          {documents.map((d) => (
            <li
              key={d.id}
              className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 text-sm"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{d.filename}</p>
                <p className="text-xs text-slate-500">
                  {d.file_type.toUpperCase()} · {formatBytes(d.file_size)} ·
                  {d.page_count != null
                    ? ` ${d.page_count} page${d.page_count === 1 ? "" : "s"} · `
                    : " "}
                  created {new Date(d.created_at).toLocaleString()}
                </p>
                {d.status === "FAILED" && d.error_message ? (
                  <p className="mt-1 text-xs text-rose-600 dark:text-rose-300">
                    {d.error_message}
                  </p>
                ) : null}
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`rounded border px-2 py-0.5 text-xs font-medium uppercase tracking-wide ${STATUS_STYLES[d.status]}`}
                >
                  {STATUS_LABEL[d.status]}
                </span>
                <label
                  className={`inline-flex cursor-pointer items-center rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-50 disabled:opacity-50 dark:text-slate-200 dark:hover:bg-slate-900 ${busy ? "pointer-events-none opacity-60" : ""}`}
                  title={d.superseded_by ? "superseded by a newer version" : "update this document's content"}
                >
                  Update
                  <input
                    type="file"
                    accept=".pdf,.txt,.csv,.json,.jsonl"
                    className="hidden"
                    disabled={busy || d.superseded_by !== null}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      e.target.value = "";
                      if (f) void onUpdate(d, f);
                    }}
                  />
                </label>
                <button
                  type="button"
                  onClick={() => onDelete(d)}
                  className="rounded border border-rose-300 px-2 py-1 text-xs text-rose-700 hover:bg-rose-50 disabled:opacity-50 dark:text-rose-200 dark:hover:bg-rose-950/40"
                  disabled={d.status === "PROCESSING" || d.status === "CREATED"}
                  title={
                    d.status === "PROCESSING" || d.status === "CREATED"
                      ? "wait until processing finishes"
                      : "delete"
                  }
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
