"""Object storage backends for Engram."""

from engram.clients.storage.base import ObjectStore
from engram.clients.storage.memory import InMemoryObjectStore

__all__ = ["InMemoryObjectStore", "ObjectStore"]
