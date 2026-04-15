"""ObjectStore abstract base class.

All storage backends (InMemory, MinIO) implement this interface.
The contract is enforced by ObjectStoreContract in tests/unit/test_storage_contract.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ObjectStore(ABC):
    """Async object storage interface."""

    @abstractmethod
    async def startup(self) -> None:
        """Initialise the backend (create buckets, open connections, etc.).

        Must be idempotent — safe to call on concurrent restart.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources gracefully."""

    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Store *data* under *key*. Overwrites if key already exists."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Return the bytes stored under *key*.

        Raises:
            KeyError: If *key* does not exist.
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if *key* exists in the store."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete *key* from the store.

        No-op if *key* does not exist (idempotent).
        """

    @abstractmethod
    async def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Return a URL that gives direct access to *key* for *expires_in* seconds.

        For backends without real URL signing (e.g. InMemory), a sentinel
        path ``/documents/_object/{key}`` is returned.
        """
