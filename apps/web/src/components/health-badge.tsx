"use client";

import { useEffect, useState } from "react";
import { apiClient } from "@/lib/api-client";
import type { HealthResponse } from "@contextvault/shared";

type Status = "loading" | "ok" | "degraded" | "down";

export default function HealthBadge() {
  const [status, setStatus] = useState<Status>("loading");
  const [detail, setDetail] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h: HealthResponse = await apiClient.health();
        if (cancelled) return;
        const allOk = Object.values(h.checks).every((c) => c.ok);
        setStatus(allOk ? "ok" : "degraded");
        setDetail(`${h.app}@${h.version} (${h.environment})`);
      } catch (err) {
        if (cancelled) return;
        setStatus("down");
        setDetail(err instanceof Error ? err.message : "unknown error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const color =
    status === "ok"
      ? "bg-emerald-100 text-emerald-800 border-emerald-300"
      : status === "loading"
        ? "bg-slate-100 text-slate-600 border-slate-300"
        : status === "degraded"
          ? "bg-amber-100 text-amber-800 border-amber-300"
          : "bg-rose-100 text-rose-800 border-rose-300";

  return (
    <div className={`inline-flex items-center gap-2 rounded-md border px-3 py-1 text-sm ${color}`}>
      <span className="font-mono uppercase">{status}</span>
      <span className="text-xs opacity-75">{detail}</span>
    </div>
  );
}