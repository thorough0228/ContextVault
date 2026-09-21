"""S3 / MinIO storage implementation."""

from __future__ import annotations

import io
import logging
from typing import BinaryIO

from app.config import Settings
from app.exceptions import BadRequestError, NotFoundError
from app.storage.base import Storage, StorageDownload

logger = logging.getLogger(__name__)


class S3Storage:
    """Thin wrapper over ``boto3.client('s3')``.

    The client is built lazily so test suites can override
    ``settings.s3_*`` before the first request lands.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    # ---- boto3 plumbing --------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self._settings.s3_endpoint_url,
                region_name=self._settings.s3_region,
                aws_access_key_id=self._settings.s3_access_key,
                aws_secret_access_key=self._settings.s3_secret_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        """Make sure the bucket exists. Cheap to call repeatedly because
        boto3 returns immediately on the head_bucket fast-path."""
        client = self._client
        bucket = self._settings.s3_bucket
        try:
            client.head_bucket(Bucket=bucket)
            return
        except client.exceptions.ClientError as exc:  # type: ignore[attr-defined]
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                # Surface other errors — e.g. auth failures.
                raise
        try:
            client.create_bucket(Bucket=bucket)
            logger.info("storage.s3.bucket.created bucket=%s", bucket)
        except client.exceptions.ClientError as exc:  # type: ignore[attr-defined]
            # Race with another worker that's creating it.
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise

    # ---- Storage protocol -----------------------------------------------

    def upload(
        self,
        *,
        key: str,
        body: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> None:
        if hasattr(body, "read"):
            body.seek(0)
        else:
            if not body:
                raise BadRequestError("upload body is empty")
        client = self._get_client()
        client.put_object(
            Bucket=self._settings.s3_bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )

    def download(self, *, key: str) -> StorageDownload:
        client = self._get_client()
        try:
            obj = client.get_object(Bucket=self._settings.s3_bucket, Key=key)
        except client.exceptions.ClientError as exc:  # type: ignore[attr-defined]
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}:
                raise NotFoundError(f"object {key} not found") from exc
            raise
        body = obj["Body"].read()
        return StorageDownload(
            key=key,
            body=body,
            content_type=obj.get("ContentType"),
        )

    def delete(self, *, key: str) -> None:
        client = self._get_client()
        client.delete_object(Bucket=self._settings.s3_bucket, Key=key)

    def exists(self, *, key: str) -> bool:
        client = self._get_client()
        try:
            client.head_object(Bucket=self._settings.s3_bucket, Key=key)
            return True
        except client.exceptions.ClientError as exc:  # type: ignore[attr-defined]
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}:
                return False
            raise

    def open_stream(self, *, key: str) -> BinaryIO:
        """Return a binary stream over the object's bytes — used by the
        parser to avoid holding the full blob in memory when we move
        off the upload size cap."""
        obj = self.download(key=key)
        return io.BytesIO(obj.body)