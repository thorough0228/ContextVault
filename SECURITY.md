# Security

Phase 6 hardening adds the boundaries below. **If you find a
hole, please report it** — see "Reporting" at the bottom.

## Authentication

* JWT bearer tokens (HS256, 60 min default).
* `JWT_SECRET` must be set via env in any non-development
  environment. The boot guard in `apps/api/app/config.py` rejects
  the placeholder `change-me-locally` default in production /
  staging.

## Authorisation

* Every user-scoped resource is owned by exactly one `user_id`
  through its parent `Rag`. Documents, Chunks, DocumentChunks,
  Conversations, Messages have no direct `user_id` column — ownership
  walks the parent chain.
* Every router that takes a `rag_id` / `document_id` /
  `conversation_id` calls `get_user_rag(...)` or equivalent BEFORE
  doing any work.
* Cross-tenant URLs return **404**, not 403. We never confirm
  existence to a non-owner.
* `RagResponse.user_id` and `ConversationPublic.user_id` are
  populated server-side from the ORM row. A response carries the
  caller's own `user_id` — never another user's.

## Body injection

* Pydantic v2 schemas are strict (`extra="ignore"` is the default).
  Any extra field — including `user_id`, `is_admin` — is dropped
  silently before the router runs.
* Request body schemas don't declare `user_id` at all.

## CORS

* `CORS_ALLOW_ORIGINS` is validated at boot: a wildcard (`*`) is
  rejected because the API sets `allow_credentials=True`. The
  browser / curl / httpx all respect this.

## File upload safety

* `Content-Length` header check in `SizeLimitMiddleware` rejects
  oversized uploads (default 20 MiB) **before** the body is read.
  The handler's own cap is a second line of defence.
* `apps/api/app/ingest/file_type.py` checks magic bytes (PDF
  header `%PDF`, UTF-8 text). `Content-Type` is **not** trusted —
  the magic-byte check is.
* Storage keys are server-generated UUIDs. The original filename
  is never used as a key, so path traversal is impossible.

## Errors

* Every error path goes through `apps/api/app/exceptions.py`. The
  envelope is `{"error": {"code": "...", "message": "...", "details": ...}}`.
* Unhandled exceptions return a generic 500. Tracebacks are logged
  but never in the response.

## Streaming chat

* The chat endpoint returns `application/x-ndjson`. The chat stream
  uses its own `{type, code, message}` envelope per line — NOT the
  global HTTP error envelope. This is intentional; clients parse
  the `type` discriminator.

## Logging

* Every log line carries a `request_id` (UUID). Same id echoes on
  the HTTP response header `X-Request-Id`.
* `LOG_FORMAT=json` switches to one-line JSON output for log
  shippers.

## Worker safety

* The worker lifespan sweeper marks documents stuck in `PROCESSING`
  longer than 5 minutes as `FAILED`. This catches rows orphaned by
  a worker crash between `mark_processing` and `mark_ready`.
* LLM transient errors retry up to 3× with exponential backoff via
  `tenacity`. After the budget is exhausted, the chat stream emits
  one terminal `error` event.

## Metrics

* `GET /api/v1/metrics` requires `User.is_admin == True`. Non-admin
  callers get 403.

## Reporting

If you find a security hole, please open a private issue or email
the maintainer. Do **not** file it as a public GitHub issue until
after a fix is shipped.