"""mineru.net official API client (v4 精准解析).

Replaces the local mineru CLI/server arrangement (Phase 15→19): PDF
bytes go to mineru.net, parsing happens server-side, and the result
zip's ``full.md`` comes back as markdown text for the shared cleaning
+ pagination pipeline (``clean_mineru_markdown`` →
``_markdown_to_pages``).

Wire flow (https://mineru.net/apiManage/docs):

1. ``POST /api/v4/file-urls/batch``  — request a presigned upload URL
   for one file; returns ``batch_id`` + ``file_urls[0]``.
2. ``PUT <file_url>`` the raw PDF bytes (no Content-Type header).
3. Poll ``GET /api/v4/extract-results/batch/{batch_id}`` until the
   file's ``state`` is ``done`` (→ ``full_zip_url``) or ``failed``
   (→ ``err_msg``).
4. ``GET full_zip_url`` → unzip in memory → the ``.md`` entry.

Every HTTP call is retried twice on transport/5xx errors (the
embedding-provider posture); a polling deadline miss or a terminal
``failed`` state raises :class:`MineruApiError` for the caller to
handle (the auto-chain degrades to the next engine).
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_SUBMIT_PATH = "/api/v4/file-urls/batch"
_RESULTS_PATH = "/api/v4/extract-results/batch/"
# per-HTTP-call retry budget (transport errors / 5xx)
_CALL_ATTEMPTS = 3


class MineruApiError(RuntimeError):
    """Terminal mineru API failure (bad token, rejected file, timeout)."""


def _client(transport: Optional[httpx.BaseTransport] = None) -> httpx.Client:
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(60.0, connect=10.0),
        headers={"Authorization": ""},  # real auth set per-request
    )


def _call_with_retry(client: httpx.Client, method: str, url: str, *,
                     headers: Optional[dict] = None,
                     content: Optional[bytes] = None,
                     json_body: Optional[dict] = None) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, _CALL_ATTEMPTS + 1):
        try:
            resp = client.request(
                method, url, headers=headers, content=content, json=json_body,
            )
            if resp.status_code >= 500 and attempt < _CALL_ATTEMPTS:
                raise httpx.TransportError(f"upstream {resp.status_code}")
            return resp
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < _CALL_ATTEMPTS:
                time.sleep(attempt)  # 1s, 2s backoff
    raise MineruApiError(f"mineru api unreachable after retries: {last_exc}")


def _submit(client: httpx.Client, base_url: str, token: str,
            filename: str) -> tuple[str, str]:
    resp = _call_with_retry(
        client, "POST", base_url.rstrip("/") + _SUBMIT_PATH,
        headers={"Authorization": f"Bearer {token}"},
        json_body={
            "enable_formula": True,
            "enable_table": True,
            "is_ocr": True,
            "files": [{"name": filename}],
        },
    )
    if resp.status_code in (401, 403):
        raise MineruApiError(
            f"mineru api auth failed ({resp.status_code}) — check MINERU_API_TOKEN"
        )
    if resp.status_code != 200:
        raise MineruApiError(
            f"mineru submit failed: HTTP {resp.status_code} {resp.text[:200]}"
        )
    body = resp.json()
    if body.get("code") != 0:
        raise MineruApiError(f"mineru submit rejected: {body.get('msg')}")
    data = body.get("data") or {}
    batch_id = data.get("batch_id")
    upload_urls = data.get("file_urls") or []
    if not batch_id or not upload_urls:
        raise MineruApiError("mineru submit returned no batch_id/upload url")
    return batch_id, upload_urls[0]


def _poll(client: httpx.Client, base_url: str, token: str,
          batch_id: str, *, deadline: float,
          poll_interval: float) -> str:
    url = base_url.rstrip("/") + _RESULTS_PATH + batch_id
    auth = {"Authorization": f"Bearer {token}"}
    while True:
        resp = _call_with_retry(client, "GET", url, headers=auth)
        if resp.status_code in (401, 403):
            raise MineruApiError(
                f"mineru api auth failed ({resp.status_code}) — check MINERU_API_TOKEN"
            )
        if resp.status_code != 200:
            raise MineruApiError(
                f"mineru poll failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        if body.get("code") != 0:
            raise MineruApiError(f"mineru poll rejected: {body.get('msg')}")
        result = ((body.get("data") or {}).get("extract_result") or [{}])[0]
        state = result.get("state")
        if state == "done":
            zip_url = result.get("full_zip_url")
            if not zip_url:
                raise MineruApiError("mineru result done but no full_zip_url")
            return zip_url
        if state == "failed":
            raise MineruApiError(
                f"mineru parse failed: {result.get('err_msg') or 'unknown error'}"
            )
        if time.monotonic() >= deadline:
            raise MineruApiError(
                f"mineru parse timed out after deadline (last state={state})"
            )
        time.sleep(poll_interval)


def _download_markdown(client: httpx.Client, zip_url: str) -> str:
    resp = _call_with_retry(client, "GET", zip_url)
    if resp.status_code != 200:
        raise MineruApiError(
            f"mineru result download failed: HTTP {resp.status_code}"
        )
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        md_names = [n for n in zf.namelist() if n.endswith(".md")]
        if not md_names:
            raise MineruApiError("mineru result zip contains no .md entry")
        # full.md is the canonical document; any other .md is supplementary
        md_names.sort(key=lambda n: (n != "full.md", n))
        return zf.read(md_names[0]).decode("utf-8", "ignore")


def fetch_markdown_via_api(
    pdf_bytes: bytes,
    *,
    token: str,
    base_url: str,
    timeout_s: float,
    poll_interval: float = 5.0,
    filename: str = "input.pdf",
    transport: Optional[httpx.BaseTransport] = None,
) -> str:
    """Parse one PDF through the mineru.net API; return markdown text.

    Raises :class:`MineruApiError` on auth problems, rejected files,
    terminal ``failed`` states, or the polling deadline — the caller
    (the auto-chain) decides how to degrade.
    """

    if not token:
        raise MineruApiError(
            "MINERU_API_TOKEN is not configured — get one from mineru.net"
            " (API manage page) and set it in .env"
        )
    deadline = time.monotonic() + max(timeout_s, poll_interval)
    with _client(transport) as client:
        batch_id, upload_url = _submit(
            client, base_url, token, filename
        )
        logger.info(
            "mineru.api.submitted batch_id=%s bytes=%d", batch_id, len(pdf_bytes)
        )
        up = _call_with_retry(client, "PUT", upload_url, content=pdf_bytes)
        if up.status_code not in (200, 201):
            raise MineruApiError(
                f"mineru file upload failed: HTTP {up.status_code}"
            )
        zip_url = _poll(
            client, base_url, token, batch_id,
            deadline=deadline, poll_interval=poll_interval,
        )
        markdown = _download_markdown(client, zip_url)
    logger.info(
        "mineru.api.ok batch_id=%s markdown_chars=%d", batch_id, len(markdown)
    )
    return markdown
