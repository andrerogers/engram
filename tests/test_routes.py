"""Tests for Engram API routes.

The store and embeddings are mocked so tests run without Postgres or API keys.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from engram.app import app
from engram.processors.base import ChunkCandidate, ChunkerKind, Modality

client = TestClient(app)

_STORE = "engram.app._store"
_EMBED = "engram.app.embeddings.embed"
_PROCESSOR = "engram.app._processor"


def _mock_store() -> AsyncMock:
    store = AsyncMock()
    store.init_db = AsyncMock()
    return store


def _mock_processor(chunks: list[str]) -> MagicMock:
    """Return a mock processor whose async process() returns ChunkCandidates."""
    candidates = [
        ChunkCandidate(
            content=c,
            chunk_index=i,
            modality=Modality.TEXT,
            chunker=ChunkerKind.TIKTOKEN_FALLBACK,
        )
        for i, c in enumerate(chunks)
    ]
    proc = MagicMock()
    proc.process = AsyncMock(return_value=candidates)
    return proc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_index_documents() -> None:
    store = _mock_store()
    store.get_or_create_collection = AsyncMock(return_value="coll-1")
    store.index_document = AsyncMock(return_value=("doc-1", 2))
    mock_embed = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
    mock_proc = _mock_processor(["chunk 1", "chunk 2"])

    with (
        patch(_STORE, store),
        patch(_EMBED, mock_embed),
        patch(_PROCESSOR, mock_proc),
    ):
        r = client.post(
            "/index",
            json={
                "workspace_id": "ws-1",
                "collection_name": "test",
                "documents": [{"path": "test.py", "content": "def hello(): pass"}],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["indexed_count"] == 1
    assert body["collection_id"] == "coll-1"
    # Verify chunker label passed through to store
    call_kwargs = store.index_document.call_args.kwargs
    assert all(c.chunker == ChunkerKind.TIKTOKEN_FALLBACK for c in call_kwargs["candidates"])


def test_index_with_existing_collection_id() -> None:
    store = _mock_store()
    store.index_document = AsyncMock(return_value=("doc-1", 1))
    mock_embed = AsyncMock(return_value=[[0.1] * 1536])
    mock_proc = _mock_processor(["chunk 1"])

    with (
        patch(_STORE, store),
        patch(_EMBED, mock_embed),
        patch(_PROCESSOR, mock_proc),
    ):
        r = client.post(
            "/index",
            json={
                "collection_id": "existing-coll",
                "documents": [{"content": "some text"}],
            },
        )
    assert r.status_code == 200
    assert r.json()["collection_id"] == "existing-coll"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieve() -> None:
    store = _mock_store()
    store.retrieve = AsyncMock(
        return_value=[
            {
                "chunk_id": "c-1",
                "document_path": "test.py",
                "content": "def hello(): pass",
                "modality": "text",
                "chunker": "tiktoken-fallback",
                "score": 0.92,
            }
        ]
    )
    mock_embed = AsyncMock(return_value=[[0.1] * 1536])

    with patch(_STORE, store), patch(_EMBED, mock_embed):
        r = client.get(
            "/retrieve", params={"q": "hello function", "collection_id": "coll-1", "k": 3}
        )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["content"] == "def hello(): pass"
    assert results[0]["chunker"] == "tiktoken-fallback"
    assert results[0]["modality"] == "text"


def test_retrieve_requires_collection_id() -> None:
    r = client.get("/retrieve", params={"q": "test"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def test_list_collections() -> None:
    store = _mock_store()
    store.list_collections = AsyncMock(
        return_value=[
            {
                "id": "coll-1",
                "workspace_id": "ws-1",
                "name": "test",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )
    with patch(_STORE, store):
        r = client.get("/collections", params={"workspace_id": "ws-1"})
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_delete_collection() -> None:
    store = _mock_store()
    store.delete_collection = AsyncMock(return_value=True)
    with patch(_STORE, store):
        r = client.delete("/collections/coll-1")
    assert r.status_code == 204


def test_delete_collection_not_found() -> None:
    store = _mock_store()
    store.delete_collection = AsyncMock(return_value=False)
    with patch(_STORE, store):
        r = client.delete("/collections/nonexistent")
    assert r.status_code == 404
