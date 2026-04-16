"""In-memory ObjectStore implementation — for tests and local dev without MinIO."""

from __future__ import annotations

from engram.clients.storage.base import ObjectStore


class InMemoryObjectStore(ObjectStore):
    """Thread-safe (within asyncio) in-memory store.

    Presigned URLs are sentinel paths of the form ``/documents/_object/{key}``
    which the read-through route in the API serves (config-gated).
    """

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._content_types: dict[str, str] = {}

    async def startup(self) -> None:
        """No-op — nothing to initialise."""

    async def shutdown(self) -> None:
        """No-op — nothing to release."""

    async def put(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        self._data[key] = data
        self._content_types[key] = content_type

    async def get(self, key: str) -> bytes:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"Object not found: {key!r}") from None

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._content_types.pop(key, None)

    async def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return f"/documents/_object/{key}"
