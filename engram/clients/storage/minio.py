"""MinIO ObjectStore backend — S3-compatible via aioboto3.

Configuration comes from environment variables (see engram.config):
  MINIO_ENDPOINT  — e.g. "http://localhost:9000"  (default for dev)
  MINIO_ACCESS_KEY / MINIO_SECRET_KEY
  MINIO_BUCKET    — default "engram"

startup() creates the bucket if it does not exist (idempotent under concurrent restart).
"""

from __future__ import annotations

import logging
from typing import Any

from engram.clients.storage.base import ObjectStore

log = logging.getLogger(__name__)

# Exceptions that mean the bucket already exists — both are success.
_BUCKET_EXISTS_CODES = {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}


class MinioObjectStore(ObjectStore):
    """S3-compatible object store backed by MinIO (via aioboto3)."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "engram",
    ) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._client: Any = None
        self._session: Any = None

    def _make_client(self) -> Any:
        import aioboto3

        session = aioboto3.Session(
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
        )
        self._session = session
        return session.client(
            "s3",
            endpoint_url=self._endpoint,
            region_name="us-east-1",
        )

    async def startup(self) -> None:
        """Open client and create bucket if absent."""
        ctx = self._make_client()
        async with ctx as client:
            try:
                await client.create_bucket(Bucket=self._bucket)
                log.info("MinIO: created bucket %r", self._bucket)
            except client.exceptions.ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code not in _BUCKET_EXISTS_CODES:
                    raise
                log.debug("MinIO: bucket %r already exists", self._bucket)

    async def shutdown(self) -> None:
        """No persistent connection to close — sessions are context-managed per call."""

    async def put(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        async with self._make_client() as client:
            await client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )

    async def get(self, key: str) -> bytes:
        async with self._make_client() as client:
            try:
                response = await client.get_object(Bucket=self._bucket, Key=key)
                body = response["Body"]
                return bytes(await body.read())
            except client.exceptions.NoSuchKey:
                raise KeyError(f"Object not found: {key!r}") from None

    async def exists(self, key: str) -> bool:
        async with self._make_client() as client:
            try:
                await client.head_object(Bucket=self._bucket, Key=key)
                return True
            except client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    return False
                raise

    async def delete(self, key: str) -> None:
        """Delete is idempotent — S3 delete_object on a missing key is a no-op."""
        async with self._make_client() as client:
            await client.delete_object(Bucket=self._bucket, Key=key)

    async def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        async with self._make_client() as client:
            url: str = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
