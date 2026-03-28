"""Tests for Engram API routes.

The store and embeddings are mocked so tests run without Postgres or API keys.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from engram.app import app

client = TestClient(app)

_STORE = "engram.app._store"
_EMBED = "engram.app.embeddings.embed"


def _mock_store() -> AsyncMock:
    store = AsyncMock()
    store.init_db = AsyncMock()
    return store


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

    with (
        patch(_STORE, store),
        patch(_EMBED, mock_embed),
        patch("engram.app.chunk_text", return_value=["chunk 1", "chunk 2"]),
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


def test_index_with_existing_collection_id() -> None:
    store = _mock_store()
    store.index_document = AsyncMock(return_value=("doc-1", 1))

    mock_embed = AsyncMock(return_value=[[0.1] * 1536])

    with (
        patch(_STORE, store),
        patch(_EMBED, mock_embed),
        patch("engram.app.chunk_text", return_value=["chunk 1"]),
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
