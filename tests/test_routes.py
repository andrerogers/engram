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


def test_retrieve_passes_modalities_to_store() -> None:
    store = _mock_store()
    store.retrieve = AsyncMock(return_value=[])
    mock_embed = AsyncMock(return_value=[[0.1] * 1536])

    with patch(_STORE, store), patch(_EMBED, mock_embed):
        r = client.get(
            "/retrieve",
            params={"q": "test", "collection_id": "coll-1", "modalities": ["image", "text"]},
        )

    assert r.status_code == 200
    call_kwargs = store.retrieve.call_args.kwargs
    assert set(call_kwargs["modalities"]) == {"image", "text"}


def test_retrieve_default_modalities_is_none() -> None:
    store = _mock_store()
    store.retrieve = AsyncMock(return_value=[])
    mock_embed = AsyncMock(return_value=[[0.1] * 1536])

    with patch(_STORE, store), patch(_EMBED, mock_embed):
        client.get("/retrieve", params={"q": "test", "collection_id": "coll-1"})

    call_kwargs = store.retrieve.call_args.kwargs
    assert call_kwargs["modalities"] is None


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


# ── Facts ─────────────────────────────────────────────────────────────────


def _fact(fact_id: str = "f1", **over: object) -> dict[str, object]:
    return {
        "id": fact_id,
        "workspace_id": "ws-1",
        "content": "uses uv",
        "tags": [],
        "source": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "pinned": False,
        **over,
    }


def test_list_facts() -> None:
    store = _mock_store()
    store.list_facts = AsyncMock(return_value=[_fact(pinned=True)])
    with patch(_STORE, store):
        r = client.get("/facts", params={"workspace_id": "ws-1", "pinned_only": True, "limit": 10})
    assert r.status_code == 200
    assert r.json() == [_fact(pinned=True, score=0.0)]
    store.list_facts.assert_awaited_once_with("ws-1", pinned_only=True, limit=10, offset=0)


def test_get_fact_and_missing_fact() -> None:
    store = _mock_store()
    store.get_fact = AsyncMock(side_effect=[_fact(), None])
    with patch(_STORE, store):
        assert client.get("/facts/f1").json()["content"] == "uses uv"
        assert client.get("/facts/gone").status_code == 404


def test_pinning_a_fact_does_not_re_embed_it() -> None:
    """Re-embedding on every pin would spend an API call on a change that is not about content."""
    store = _mock_store()
    store.set_fact_pinned = AsyncMock(return_value=True)
    store.get_fact = AsyncMock(return_value=_fact(pinned=True))
    with patch(_STORE, store), patch(_EMBED, new=AsyncMock()) as embed:
        r = client.patch("/facts/f1", json={"pinned": True})
    assert r.status_code == 200
    assert r.json()["pinned"] is True
    store.set_fact_pinned.assert_awaited_once_with("f1", pinned=True)
    embed.assert_not_awaited()
    store.upsert_fact.assert_not_awaited()


def test_editing_a_fact_re_embeds_it_and_keeps_its_workspace() -> None:
    """An edited fact whose vector still points at the old wording is recalled by the old wording."""
    store = _mock_store()
    store.get_fact = AsyncMock(side_effect=[_fact(), _fact(content="uses uv 0.9")])
    with patch(_STORE, store), patch(_EMBED, new=AsyncMock(return_value=[[0.1, 0.2]])):
        r = client.patch("/facts/f1", json={"content": "uses uv 0.9"})
    assert r.status_code == 200
    assert r.json()["content"] == "uses uv 0.9"
    store.upsert_fact.assert_awaited_once_with(
        fact_id="f1",
        workspace_id="ws-1",
        content="uses uv 0.9",
        tags=[],
        source=None,
        embedding=[0.1, 0.2],
    )


def test_patch_with_nothing_to_change_is_rejected() -> None:
    store = _mock_store()
    store.get_fact = AsyncMock(return_value=None)
    with patch(_STORE, store):
        assert client.patch("/facts/f1", json={}).status_code == 400
        assert client.patch("/facts/gone", json={"pinned": True}).status_code == 404
