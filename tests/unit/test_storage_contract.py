"""ObjectStoreContract — base class applied to every ObjectStore implementation.

Any backend that passes this contract is proven interchangeable with InMemory.
Apply by subclassing and overriding ``make_store``.
"""

from __future__ import annotations

import pytest

from engram.clients.storage.base import ObjectStore


class ObjectStoreContract:
    """Mixin — subclass + override ``make_store`` to apply to a backend."""

    @pytest.fixture
    async def store(self) -> ObjectStore:  # type: ignore[override]
        raise NotImplementedError("Override make_store fixture")

    # ── startup / shutdown ────────────────────────────────────────────────

    async def test_startup_idempotent(self, store: ObjectStore) -> None:
        """startup() called twice must not raise."""
        await store.startup()
        await store.startup()

    async def test_shutdown_idempotent(self, store: ObjectStore) -> None:
        """shutdown() called twice must not raise."""
        await store.shutdown()
        await store.shutdown()

    # ── put / get ─────────────────────────────────────────────────────────

    async def test_put_and_get_roundtrip(self, store: ObjectStore) -> None:
        await store.put("k1", b"hello world")
        assert await store.get("k1") == b"hello world"

    async def test_put_overwrites(self, store: ObjectStore) -> None:
        await store.put("k2", b"first")
        await store.put("k2", b"second")
        assert await store.get("k2") == b"second"

    async def test_get_missing_raises_key_error(self, store: ObjectStore) -> None:
        with pytest.raises(KeyError):
            await store.get("no-such-key")

    # ── exists ────────────────────────────────────────────────────────────

    async def test_exists_true_after_put(self, store: ObjectStore) -> None:
        await store.put("k3", b"data")
        assert await store.exists("k3") is True

    async def test_exists_false_for_missing(self, store: ObjectStore) -> None:
        assert await store.exists("no-such-key-exists") is False

    # ── delete ────────────────────────────────────────────────────────────

    async def test_delete_removes_key(self, store: ObjectStore) -> None:
        await store.put("k4", b"to delete")
        await store.delete("k4")
        assert await store.exists("k4") is False

    async def test_delete_missing_is_noop(self, store: ObjectStore) -> None:
        """delete() on a non-existent key must not raise."""
        await store.delete("no-such-key-delete")

    # ── presigned_url ─────────────────────────────────────────────────────

    async def test_presigned_url_returns_str(self, store: ObjectStore) -> None:
        await store.put("k5", b"file bytes")
        url = await store.presigned_url("k5")
        assert isinstance(url, str)
        assert url  # non-empty
