"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { apiClient } from "@/lib/api-client";
import { useSession } from "@/lib/auth-store";

export default function HeaderNav() {
  const { user, hydrated, logout } = useSession();
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  if (!hydrated) {
    return (
      <nav className="flex gap-4 text-sm text-slate-600 dark:text-slate-300">
        <span className="opacity-40">…</span>
      </nav>
    );
  }

  async function onLogout() {
    setBusy(true);
    try {
      await apiClient.logout();
    } catch {
      // best-effort; the client-side discard is what really ends the session
    }
    logout();
    router.push("/login");
  }

  if (user) {
    return (
      <nav className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-300">
        <Link href="/dashboard" className="hover:text-brand-600">
          Dashboard
        </Link>
        <span className="text-xs text-slate-500">{user.email}</span>
        <button
          type="button"
          onClick={onLogout}
          disabled={busy}
          className="rounded border border-slate-300 px-2 py-1 text-xs hover:border-brand-500 disabled:opacity-50 dark:border-slate-700"
        >
          {busy ? "Logging out…" : "Logout"}
        </button>
      </nav>
    );
  }

  return (
    <nav className="flex gap-4 text-sm text-slate-600 dark:text-slate-300">
      <Link href="/login" className="hover:text-brand-600">
        Login
      </Link>
      <Link href="/register" className="hover:text-brand-600">
        Register
      </Link>
    </nav>
  );
}