"""LocalFileObjectStore — the ObjectStore contract, plus filesystem-specific behaviour."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from engram.clients.storage.local import LocalFileObjectStore
from tests.unit.test_storage_contract import ObjectStoreContract


class TestLocalFileObjectStore(ObjectStoreContract):
    @pytest.fixture
    async def store(self, tmp_path: Path) -> AsyncIterator[LocalFileObjectStore]:
        s = LocalFileObjectStore(tmp_path / "objects")
        await s.startup()
        yield s
        await s.shutdown()


async def test_objects_land_under_the_root(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path / "objects")
    await store.startup()
    await store.put("collection-1/doc-1", b"bytes")
    assert (tmp_path / "objects" / "collection-1" / "doc-1").read_bytes() == b"bytes"


async def test_survives_reopening(tmp_path: Path) -> None:
    first = LocalFileObjectStore(tmp_path / "objects")
    await first.startup()
    await first.put("c/1", b"kept")
    assert await LocalFileObjectStore(tmp_path / "objects").get("c/1") == b"kept"


async def test_keys_cannot_escape_the_root(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path / "objects")
    await store.startup()
    victim = tmp_path / "secret.txt"
    victim.write_text("do not touch", encoding="utf-8")

    for key in ["../secret.txt", "c/../../secret.txt", "/etc/passwd", ""]:
        with pytest.raises(ValueError):
            await store.put(key, b"x")
        with pytest.raises(ValueError):
            await store.get(key)
        with pytest.raises(ValueError):
            await store.delete(key)

    assert victim.read_text(encoding="utf-8") == "do not touch"

    # A doubled separator is not an escape — it normalises to the same object.
    await store.put("c/1", b"same object")
    assert await store.get("c//1") == b"same object"


async def test_a_failed_write_leaves_no_partial_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalFileObjectStore(tmp_path / "objects")
    await store.startup()
    await store.put("c/1", b"original")

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", _boom)
    with pytest.raises(OSError):
        await store.put("c/1", b"replacement")

    assert await store.get("c/1") == b"original"
    assert list((tmp_path / "objects" / "c").glob("*.part")) == []


async def test_delete_prunes_empty_directories(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path / "objects")
    await store.startup()
    await store.put("c/1", b"a")
    await store.put("c/2", b"b")

    await store.delete("c/1")
    assert (tmp_path / "objects" / "c").is_dir()
    await store.delete("c/2")
    assert not (tmp_path / "objects" / "c").exists()
    assert (tmp_path / "objects").is_dir()
