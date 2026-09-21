"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useSession } from "@/lib/auth-store";

/**
 * Entry route — auth-aware redirect. Logged-in users land on the
 * dashboard; anonymous visitors go to the login page. The API health
 * badge lives on the dashboard, so the Phase-1 placeholder cards are
 * gone entirely.
 */
export default function HomePage() {
  const router = useRouter();
  const { hydrated, token } = useSession();

  useEffect(() => {
    if (!hydrated) return;
    router.replace(token ? "/dashboard" : "/login");
  }, [hydrated, token, router]);

  return (
    <p className="text-sm text-slate-500">
      {hydrated ? "Redirecting…" : "Loading…"}
    </p>
  );
}
