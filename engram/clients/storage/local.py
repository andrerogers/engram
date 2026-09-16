"""LocalFileObjectStore — object storage on the filesystem, no Docker required.

Objects live at ``<root>/<key>``, where the key is ``<collection_id>/<object_id>``. Writes are
atomic (temp file in the destination directory, then ``os.replace``), so a crashed write never
leaves a half-written object where a reader can see it.

Keys reach the filesystem, so every key is validated: relative, no ``..`` segment, and the
resolved path must stay inside the root.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from engram.clients.storage.base import ObjectStore

log = logging.getLogger(__name__)


class LocalFileObjectStore(ObjectStore):
    """Filesystem-backed object store rooted at *root*."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).expanduser()

    def _path(self, key: str) -> Path:
        if not key or key.startswith("/") or Path(key).is_absolute():
            raise ValueError(f"Object key must be relative: {key!r}")
        if any(part in ("..", "") for part in Path(key).parts):
            raise ValueError(f"Object key must not traverse directories: {key!r}")
        root = self._root.resolve()
        path = (root / key).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Object key escapes the object store: {key!r}")
        return path

    async def startup(self) -> None:
        """Create the object root. Idempotent."""
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)
        log.info("engram: object store at %s", self._root)

    async def shutdown(self) -> None:
        """No-op — nothing is held open."""

    def _put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    async def put(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        """Store *data* under *key*, atomically. ``content_type`` is not persisted."""
        await asyncio.to_thread(self._put, key, data)

    async def get(self, key: str) -> bytes:
        def _read() -> bytes:
            try:
                return self._path(key).read_bytes()
            except (FileNotFoundError, IsADirectoryError):
                raise KeyError(f"Object not found: {key!r}") from None

        return await asyncio.to_thread(_read)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(lambda: self._path(key).is_file())

    async def delete(self, key: str) -> None:
        """Delete *key*, and any directory it leaves empty. No-op if absent."""

        def _delete() -> None:
            path = self._path(key)
            path.unlink(missing_ok=True)
            parent = path.parent
            root = self._root.resolve()
            while parent != root and parent.is_relative_to(root) and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent

        await asyncio.to_thread(_delete)

    async def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Sentinel path served by the read-through route — the filesystem cannot sign URLs."""
        return f"/documents/_object/{key}"
