"""Apply ObjectStoreContract to InMemoryObjectStore.

Also tests InMemory-specific behaviour (sentinel presigned URL).
"""

from __future__ import annotations

import pytest

from engram.clients.storage.memory import InMemoryObjectStore
from tests.unit.test_storage_contract import ObjectStoreContract


class TestInMemoryObjectStore(ObjectStoreContract):
    @pytest.fixture
    async def store(self) -> InMemoryObjectStore:
        s = InMemoryObjectStore()
        await s.startup()
        return s

    # ── InMemory-specific ─────────────────────────────────────────────────

    async def test_presigned_url_sentinel_format(self, store: InMemoryObjectStore) -> None:
        await store.put("my/file.pdf", b"bytes")
        url = await store.presigned_url("my/file.pdf")
        assert url == "/documents/_object/my/file.pdf"

    async def test_isolated_instances(self) -> None:
        """Two InMemoryObjectStore instances do not share state."""
        a = InMemoryObjectStore()
        b = InMemoryObjectStore()
        await a.put("key", b"from-a")
        assert not await b.exists("key")
