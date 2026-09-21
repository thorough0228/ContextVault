"use client";

/**
 * Tiny client-side session store backed by ``localStorage``.
 *
 * Phase 2 ships JWT auth with no server-side revocation. The
 * ``Authorization: Bearer <token>`` header is the only place the
 * token lives; ``localStorage`` is convenient and good enough for
 * Phase 2. Phase 3 should migrate to ``httpOnly`` cookies + CSRF
 * defence once we add a real domain.
 */

import { useCallback, useEffect, useState } from "react";
import type { UserPublic } from "@contextvault/shared";

const TOKEN_KEY = "contextvault:token";
const USER_KEY = "contextvault:user";
/** Fired (same tab) whenever the session is written or cleared. */
const SESSION_EVENT = "contextvault:session-changed";

function readToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

function readUser(): UserPublic | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as UserPublic;
  } catch {
    return null;
  }
}

function emitSessionChanged() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(SESSION_EVENT));
}

export function persistSession(token: string, user: UserPublic) {
  window.localStorage.setItem(TOKEN_KEY, token);
  window.localStorage.setItem(USER_KEY, JSON.stringify(user));
  emitSessionChanged();
}

export function clearSession() {
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(USER_KEY);
  emitSessionChanged();
}

export function getStoredToken(): string | null {
  return readToken();
}

export function getStoredUser(): UserPublic | null {
  return readUser();
}

/**
 * React hook that re-renders when the session changes (login / logout).
 *
 * Every ``useSession()`` call owns independent state, so instances stay in
 * sync via the ``SESSION_EVENT`` (same tab) and the browser ``storage``
 * event (other tabs). Without this, the layout-level ``HeaderNav`` would
 * keep showing Login/Register after a client-side login navigation.
 */
export function useSession() {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<UserPublic | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const sync = () => {
      setToken(readToken());
      setUser(readUser());
    };
    sync();
    setHydrated(true);
    window.addEventListener(SESSION_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(SESSION_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const login = useCallback((nextToken: string, nextUser: UserPublic) => {
    persistSession(nextToken, nextUser);
  }, []);

  const logout = useCallback(() => {
    clearSession();
  }, []);

  return { token, user, hydrated, login, logout };
}
