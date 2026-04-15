"""Apply ObjectStoreContract to MinioObjectStore.

Requires compose.test.yml MinIO running on port 19000:
    docker compose -f compose.test.yml up -d minio
    pytest -m integration tests/integration/test_storage_minio.py

Contract equivalence with InMemory is guaranteed by sharing ObjectStoreContract.
"""

from __future__ import annotations

import pytest

from engram.clients.storage.minio import MinioObjectStore
from tests.unit.test_storage_contract import ObjectStoreContract

pytestmark = pytest.mark.integration

_ENDPOINT = "http://localhost:19000"
_ACCESS_KEY = "minioadmin"
_SECRET_KEY = "minioadmin"
_BUCKET = "engram-test"


@pytest.fixture
async def minio_store() -> MinioObjectStore:
    store = MinioObjectStore(
        endpoint=_ENDPOINT,
        access_key=_ACCESS_KEY,
        secret_key=_SECRET_KEY,
        bucket=_BUCKET,
    )
    await store.startup()
    return store


class TestMinioObjectStore(ObjectStoreContract):
    @pytest.fixture
    async def store(self, minio_store: MinioObjectStore) -> MinioObjectStore:
        return minio_store

    # ── MinIO-specific ─────────────────────────────────────────────────────

    async def test_presigned_url_is_http(self, store: MinioObjectStore) -> None:
        await store.put("integration/test.bin", b"integration bytes")
        url = await store.presigned_url("integration/test.bin")
        assert url.startswith("http")

    async def test_startup_idempotent_real(self, store: MinioObjectStore) -> None:
        """Calling startup() a second time must not raise (bucket already exists)."""
        await store.startup()
