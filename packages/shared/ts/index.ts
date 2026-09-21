/**
 * Cross-app TypeScript constants — keep in sync with
 * packages/shared/python/contextvault_shared/constants.py
 */

export const APP_NAME = "contextvault";
export const API_VERSION = "0.4.0";
export const API_PREFIX = "/api/v1";
export const HEALTH_PATH = `${API_PREFIX}/health`;

export const SERVICE_API = "api";
export const SERVICE_WORKER = "worker";
export const SERVICE_WEB = "web";

// ---- Auth (Phase 2) ----
export const AUTH_PATHS = {
  register: `${API_PREFIX}/auth/register`,
  login: `${API_PREFIX}/auth/login`,
  logout: `${API_PREFIX}/auth/logout`,
  me: `${API_PREFIX}/auth/me`,
} as const;

// ---- RAG (Phase 2) ----
export const RAG_PATHS = {
  list: `${API_PREFIX}/rags`,
  create: `${API_PREFIX}/rags`,
  detail: (ragId: string) => `${API_PREFIX}/rags/${ragId}`,
} as const;

// ---- Documents (Phase 3) ----
export const DOCUMENT_PATHS = {
  upload: (ragId: string) => `${API_PREFIX}/rags/${ragId}/documents`,
  list: (ragId: string) => `${API_PREFIX}/rags/${ragId}/documents`,
  detail: (documentId: string) => `${API_PREFIX}/documents/${documentId}`,
  delete: (documentId: string) => `${API_PREFIX}/documents/${documentId}`,
  // Phase 13 document update (hash-detected) + bluegreen rollback.
  update: (documentId: string) => `${API_PREFIX}/documents/${documentId}/content`,
  rollback: (documentId: string) => `${API_PREFIX}/documents/${documentId}/rollback`,
} as const;

// ---- Search (Phase 4) ----
export const SEARCH_PATHS = {
  search: (ragId: string) => `${API_PREFIX}/rags/${ragId}/search`,
} as const;

export interface SearchHit {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_text: string;
  score: number;
  page_number: number;
  metadata: Record<string, unknown> | null;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
}

export interface SearchResponse {
  hits: SearchHit[];
  query: string;
  top_k: number;
  embedding_model: string;
  embedding_dimension: number;
}

export const DOCUMENT_FILE_TYPES = ["pdf", "txt", "csv", "json", "jsonl"] as const;
export const DOCUMENT_STATUSES = [
  "CREATED",
  "PROCESSING",
  "READY",
  "FAILED",
] as const;

export type RagStatus = "ACTIVE" | "PROCESSING" | "ERROR";
export type DocumentStatus =
  | "CREATED"
  | "PROCESSING"
  | "READY"
  | "FAILED";
export type DocumentFileType = "pdf" | "txt" | "csv" | "json" | "jsonl";

export interface UserPublic {
  id: string;
  email: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: UserPublic;
}

export interface Rag {
  id: string;
  user_id: string;
  name: string;
  description: string;
  status: RagStatus;
  created_at: string;
  updated_at: string;
}

export interface RagListResponse {
  items: Rag[];
  total: number;
}

export interface RagCreateRequest {
  name: string;
  description?: string;
}

export interface RagPatchRequest {
  name?: string;
  description?: string;
  status?: RagStatus;
}

export interface Document {
  id: string;
  rag_id: string;
  filename: string;
  file_type: DocumentFileType;
  file_size: number;
  storage_key: string;
  status: DocumentStatus;
  error_message: string | null;
  page_count: number | null;
  /** Bluegreen update chain: set when a newer version replaced this document. */
  superseded_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: Document[];
  total: number;
  page: number;
  page_size: number;
}

export interface Chunk {
  id: string;
  document_id: string;
  rag_id: string;
  page_number: number;
  chunk_index: number;
  text: string;
  metadata: Record<string, unknown> | null;
}

export interface DocumentDetail extends Document {
  chunks: Chunk[];
}

export interface HealthCheckResult {
  ok: boolean;
  error?: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  app: string;
  version: string;
  environment: string;
  checks: Record<string, HealthCheckResult>;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

// ---- Chat (Phase 5) ----

export const CHAT_PATHS = {
  stream: (ragId: string) => `${API_PREFIX}/rags/${ragId}/chat`,
  conversations: (ragId: string) =>
    `${API_PREFIX}/rags/${ragId}/conversations`,
  conversation: (conversationId: string) =>
    `${API_PREFIX}/conversations/${conversationId}`,
} as const;

export interface Citation {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_text: string;
  retrieval_score: number;
  metadata?: Record<string, unknown> | null;
}

export interface ChatCitationEvent {
  type: "citation";
  citations: Citation[];
}

export interface ChatTokenEvent {
  type: "token";
  delta: string;
}

export interface ChatDoneEvent {
  type: "done";
  message_id: string;
  conversation_id: string;
  citations: Citation[];
}

export interface ChatErrorEvent {
  type: "error";
  code: string;
  message: string;
}

export type ChatStreamEvent =
  | ChatCitationEvent
  | ChatTokenEvent
  | ChatDoneEvent
  | ChatErrorEvent;

export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
  top_k?: number;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Citation[] | null;
  created_at: string;
}

export interface ConversationPublic {
  id: string;
  user_id: string;
  rag_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationPublic {
  messages: ChatMessage[];
}