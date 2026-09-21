/**
 * Lightweight browser API client.
 *
 * Phase 1 only called ``GET /api/v1/health``. Phase 2 adds:
 *   - bearer-token aware fetch helper
 *   - auth: register / login / logout / me
 *   - rags: list / create / get / patch / delete
 * Phase 3 adds:
 *   - documents: upload / list / get / delete
 *
 * The token is read from a single accessor so we can swap it for an
 * httpOnly cookie store in a later phase without rewriting call sites.
 */

import {
  AUTH_PATHS,
  CHAT_PATHS,
  DOCUMENT_PATHS,
  HEALTH_PATH,
  RAG_PATHS,
  SEARCH_PATHS,
  type ApiErrorBody,
  type ChatRequest,
  type ChatStreamEvent,
  type Chunk,
  type ConversationDetail,
  type ConversationPublic,
  type Document,
  type DocumentDetail,
  type DocumentListResponse,
  type HealthResponse,
  type Rag,
  type RagCreateRequest,
  type RagListResponse,
  type RagPatchRequest,
  type RagStatus,
  type SearchRequest,
  type SearchResponse,
  type TokenResponse,
  type UserPublic,
} from "@contextvault/shared";

import { clearSession } from "./auth-store";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  body: ApiErrorBody | null;
  constructor(status: number, message: string, body: ApiErrorBody | null) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export type TokenProvider = () => string | null;

const defaultTokenProvider: TokenProvider = () => {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem("contextvault:token");
};

let _tokenProvider: TokenProvider = defaultTokenProvider;

/** Test seam — swap the provider in tests / non-browser contexts. */
export function setTokenProvider(provider: TokenProvider) {
  _tokenProvider = provider;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  withAuth = true,
): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...(init.headers as Record<string, string> | undefined),
  };
  if (withAuth) {
    const token = _tokenProvider();
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch(url, {
    ...init,
    headers,
    cache: "no-store",
  });

  let parsed: unknown = null;
  const text = await res.text();
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }

  if (!res.ok) {
    // Expired/invalid bearer token on an authed call — clear the stale
    // session and bounce to login. (login/register use withAuth=false,
    // so their 401s stay inline as credential errors.)
    if (res.status === 401 && withAuth && _tokenProvider()) {
      clearSession();
      if (typeof window !== "undefined") {
        window.location.assign("/login?expired=1");
      }
    }
    const body = (parsed as ApiErrorBody) ?? null;
    const message = body?.error?.message ?? `HTTP ${res.status}`;
    throw new ApiError(res.status, message, body);
  }

  if (res.status === 204) return null as unknown as T;
  return parsed as T;
}

export const apiClient = {
  baseUrl: API_BASE_URL,

  health: () => request<HealthResponse>(HEALTH_PATH, {}, false),

  // ---- Auth -------------------------------------------------------------
  register: (email: string, password: string) =>
    request<TokenResponse>(
      AUTH_PATHS.register,
      { method: "POST", body: JSON.stringify({ email, password }) },
      false,
    ),
  login: (email: string, password: string) =>
    request<TokenResponse>(
      AUTH_PATHS.login,
      { method: "POST", body: JSON.stringify({ email, password }) },
      false,
    ),
  logout: () => request<{ status: string; message: string }>(AUTH_PATHS.logout, {
    method: "POST",
  }),
  me: () => request<UserPublic>(AUTH_PATHS.me),

  // ---- RAGs -------------------------------------------------------------
  listRags: () => request<RagListResponse>(RAG_PATHS.list),
  createRag: (payload: RagCreateRequest) =>
    request<Rag>(RAG_PATHS.create, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getRag: (ragId: string) => request<Rag>(RAG_PATHS.detail(ragId)),
  patchRag: (ragId: string, payload: RagPatchRequest) =>
    request<Rag>(RAG_PATHS.detail(ragId), {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteRag: (ragId: string) =>
    request<null>(RAG_PATHS.detail(ragId), { method: "DELETE" }),

  // ---- Documents (Phase 3) --------------------------------------------- --
  uploadDocument(ragId: string, file: File, onProgress?: (sent: number, total: number) => void) {
    return new Promise<Document>((resolve, reject) => {
      const url = `${API_BASE_URL}${DOCUMENT_PATHS.upload(ragId)}`;
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.responseType = "json";
      xhr.upload.onprogress = (evt) => {
        if (evt.lengthComputable && onProgress) {
          onProgress(evt.loaded, evt.total);
        }
      };
      xhr.onerror = () => reject(new ApiError(0, "network error", null));
      xhr.onload = () => {
        const parsed: unknown = xhr.response;
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(parsed as Document);
        } else {
          const body = (parsed as ApiErrorBody) ?? null;
          const msg = body?.error?.message ?? `HTTP ${xhr.status}`;
          reject(new ApiError(xhr.status, msg, body));
        }
      };
      const fd = new FormData();
      fd.append("file", file);
      const token = _tokenProvider();
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      xhr.send(fd);
    });
  },
  listDocuments: (ragId: string, page = 1, pageSize = 20) =>
    request<DocumentListResponse>(
      `${DOCUMENT_PATHS.list(ragId)}?page=${page}&page_size=${pageSize}`,
    ),
  getDocument: (documentId: string) =>
    request<DocumentDetail>(DOCUMENT_PATHS.detail(documentId)),
  deleteDocument: (documentId: string) =>
    request<null>(DOCUMENT_PATHS.delete(documentId), { method: "DELETE" }),

  // ---- Document update (Phase 13) ------------------------------------
  // Hash-detected update: 200 updated:false (unchanged), 202 with the
  // strategy result (replace / bluegreen), 409 on conflicts.
  updateDocument: (documentId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Record<string, unknown>>(
      `${DOCUMENT_PATHS.update(documentId)}`,
      { method: "PUT", body: form },
    );
  },
  rollbackDocument: (documentId: string) =>
    request<Record<string, unknown>>(
      DOCUMENT_PATHS.rollback(documentId),
      { method: "POST" },
    ),

  // ---- Search (Phase 4) ----------------------------------------------
  searchRags: (ragId: string, payload: SearchRequest) =>
    request<SearchResponse>(SEARCH_PATHS.search(ragId), {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ---- Chat (Phase 5) -------------------------------------------------
  // Returns an async iterable of events. Throws ApiError on non-2xx
  // response (e.g. 401 / 404). On 200, the response body is a
  // stream of NDJSON lines — we yield one event per line.
  async *streamChat(
    ragId: string,
    payload: ChatRequest,
  ): AsyncGenerator<ChatStreamEvent> {
    const url = `${API_BASE_URL}${CHAT_PATHS.stream(ragId)}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/x-ndjson",
    };
    const token = _tokenProvider();
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    if (!res.ok || !res.body) {
      let parsed: unknown = null;
      const text = await res.text();
      if (text) {
        try { parsed = JSON.parse(text); } catch { /* ignore */ }
      }
      const body = (parsed as ApiErrorBody) ?? null;
      throw new ApiError(
        res.status,
        body?.error?.message ?? `HTTP ${res.status}`,
        body,
      );
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          yield JSON.parse(line) as ChatStreamEvent;
        } catch {
          // Skip malformed lines — the server should never emit them
          // but a single corrupt line shouldn't kill the whole turn.
          continue;
        }
      }
    }
    if (buffer.trim()) {
      try { yield JSON.parse(buffer) as ChatStreamEvent; } catch { /* ignore */ }
    }
  },
  listConversations: (ragId: string) =>
    request<ConversationPublic[]>(CHAT_PATHS.conversations(ragId)),
  getConversation: (conversationId: string) =>
    request<ConversationDetail>(CHAT_PATHS.conversation(conversationId)),
};

export { API_BASE_URL };

export type {
  Rag,
  RagStatus,
  RagCreateRequest,
  RagPatchRequest,
  UserPublic,
  TokenResponse,
  Document,
  DocumentDetail,
  DocumentListResponse,
  Chunk,
  SearchRequest,
  SearchResponse,
  ChatRequest,
  ChatStreamEvent,
  ConversationPublic,
  ConversationDetail,
};