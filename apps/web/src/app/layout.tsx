import type { Metadata } from "next";
import "./globals.css";
import { APP_NAME } from "@contextvault/shared";
import HeaderNav from "@/components/header-nav";

export const metadata: Metadata = {
  title: `${APP_NAME} — Phase 2`,
  description: "ContextVault Phase 2 — user accounts and RAG management.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
            <a href="/" className="text-lg font-semibold text-brand-700">
              {APP_NAME}
            </a>
            <HeaderNav />
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}