"""mineru.net Agent lightweight parse API client.

The mineru engine parses through the free Agent API (``/api/v1/agent/
parse/file``): no auth (IP rate-limited; 429 = too many requests),
single file ≤10 MB / ≤20 pages, markdown-only result on a CDN link.
Documents above 20 pages are pre-split by the caller (parser layer)
and each part parsed as its own task.

Wire flow (https://mineru.net/apiManage/docs):

1. ``POST /api/v1/agent/parse/file`` — body ``file_name`` (required,
   drives type detection) + ``is_ocr``/``enable_table``/``enable_formula``;
   returns ``data.task_id`` + ``data.file_url`` (signed OSS upload URL).
2. ``PUT <file_url>`` the raw PDF bytes (no Content-Type header);
   the service picks the file up automatically.
3. Poll ``GET /api/v1/agent/parse/{task_id}`` until ``state`` is
   ``done`` (→ ``data.markdown_url``) or ``failed`` (→ ``err_code`` /
   ``err_msg``).
4. ``GET markdown_url`` → the ``full.md`` text.

Every HTTP call is retried twice on transport/429/5xx errors; a
polling deadline miss or a terminal ``failed`` state raises
:class:`MineruApiError` for the caller to handle (the auto-chain
degrades to the next engine).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_SUBMIT_PATH = "/api/v1/agent/parse/file"
_TASK_PATH = "/api/v1/agent/parse/"
# per-HTTP-call retry budget (transport errors / 429 / 5xx)
_CALL_ATTEMPTS = 3


class MineruApiError(RuntimeError):
    """Terminal mineru API failure (rejected file, timeout, quota)."""


def _client(transport: Optional[httpx.BaseTransport] = None) -> httpx.Client:
    # trust_env=False: mineru API + its OSS/CDN hosts must always
    # connect DIRECT — a local HTTP(S)_PROXY (e.g. clash on :7897)
    # mangles some of these TLS streams ("UNEXPECTED_EOF_WHILE_READING")
    # and the retried failures cost minutes per document.
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(60.0, connect=10.0),
        trust_env=False,
    )


def _call_with_retry(client: httpx.Client, method: str, url: str, *,
                     content: Optional[bytes] = None,
                     json_body: Optional[dict] = None) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, _CALL_ATTEMPTS + 1):
        try:
            resp = client.request(
                method, url, content=content, json=json_body,
            )
            # 429 = IP rate limit; 5xx = upstream trouble — both retryable
            if resp.status_code in (429,) or resp.status_code >= 500:
                if attempt < _CALL_ATTEMPTS:
                    time.sleep(attempt)  # 1s, 2s backoff
                    continue
            return resp
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < _CALL_ATTEMPTS:
                time.sleep(attempt)
    raise MineruApiError(f"mineru api unreachable after retries: {last_exc}")


def _submit(client: httpx.Client, base_url: str,
            filename: str) -> tuple[str, str]:
    resp = _call_with_retry(
        client, "POST", base_url.rstrip("/") + _SUBMIT_PATH,
        json_body={
            "file_name": filename,
            "is_ocr": True,
            "enable_table": True,
            "enable_formula": True,
        },
    )
    if resp.status_code != 200:
        raise MineruApiError(
            f"mineru submit failed: HTTP {resp.status_code} {resp.text[:200]}"
        )
    body = resp.json()
    if body.get("code") != 0:
        raise MineruApiError(f"mineru submit rejected: {body.get('msg')}")
    data = body.get("data") or {}
    task_id = data.get("task_id")
    upload_url = data.get("file_url")
    if not task_id or not upload_url:
        raise MineruApiError("mineru submit returned no task_id/upload url")
    return task_id, upload_url


def _poll(client: httpx.Client, base_url: str, task_id: str, *,
          deadline: float, poll_interval: float) -> str:
    url = base_url.rstrip("/") + _TASK_PATH + task_id
    while True:
        resp = _call_with_retry(client, "GET", url)
        if resp.status_code != 200:
            raise MineruApiError(
                f"mineru poll failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        if body.get("code") != 0:
            raise MineruApiError(f"mineru poll rejected: {body.get('msg')}")
        data = body.get("data") or {}
        state = data.get("state")
        if state == "done":
            markdown_url = data.get("markdown_url")
            if not markdown_url:
                raise MineruApiError("mineru result done but no markdown_url")
            return markdown_url
        if state == "failed":
            err = data.get("err_msg") or f"err_code {data.get('err_code')}"
            raise MineruApiError(f"mineru parse failed: {err}")
        if time.monotonic() >= deadline:
            raise MineruApiError(
                f"mineru parse timed out after deadline (last state={state})"
            )
        time.sleep(poll_interval)


def _download_markdown(client: httpx.Client, markdown_url: str) -> str:
    resp = _call_with_retry(client, "GET", markdown_url)
    if resp.status_code != 200:
        raise MineruApiError(
            f"mineru markdown download failed: HTTP {resp.status_code}"
        )
    return resp.text


def fetch_markdown_via_api(
    pdf_bytes: bytes,
    *,
    base_url: str,
    timeout_s: float,
    poll_interval: float = 5.0,
    filename: str = "input.pdf",
    transport: Optional[httpx.BaseTransport] = None,
) -> str:
    """Parse one PDF (≤20 pages / ≤10 MB) through the Agent API;
    return markdown text.

    Raises :class:`MineruApiError` on rejected files, terminal
    ``failed`` states, or the polling deadline — the caller (the
    auto-chain) decides how to degrade.
    """

    deadline = time.monotonic() + max(timeout_s, poll_interval)
    with _client(transport) as client:
        task_id, upload_url = _submit(client, base_url, filename)
        logger.info(
            "mineru.api.submitted task_id=%s bytes=%d", task_id, len(pdf_bytes)
        )
        up = _call_with_retry(client, "PUT", upload_url, content=pdf_bytes)
        if up.status_code not in (200, 201):
            raise MineruApiError(
                f"mineru file upload failed: HTTP {up.status_code}"
            )
        markdown_url = _poll(
            client, base_url, task_id,
            deadline=deadline, poll_interval=poll_interval,
        )
        markdown = _download_markdown(client, markdown_url)
    logger.info(
        "mineru.api.ok task_id=%s markdown_chars=%d", task_id, len(markdown)
    )
    return markdown
