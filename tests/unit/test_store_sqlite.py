"""Engram Store against a real engram.db with sqlite-vec (no external services)."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engram.processors.base import ChunkCandidate
from engram.store import Store

DIMS = 8


def _vec(*head: float) -> list[float]:
    v = [*head, *([0.0] * (DIMS - len(head)))]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _chunk(i: int, text: str, modality: str = "text") -> ChunkCandidate:
    return ChunkCandidate(
        content=text, chunk_index=i, modality=modality, chunker="tiktoken-fallback"
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[Store]:
    s = Store(tmp_path / "engram.db", dimensions=DIMS)
    await s.init_db()
    yield s
    await s.close()


def _backdate(store: Store, created_at: dict[str, str]) -> None:
    """Rows created microseconds apart do not order reliably by timestamp."""
    store._execute(
        lambda c: [
            c.execute("UPDATE facts SET created_at = ? WHERE id = ?", (when, fact_id))
            for fact_id, when in created_at.items()
        ]
    )


def _count(store: Store, table: str) -> int:
    return store._execute(lambda c: c.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


# ── Collections ───────────────────────────────────────────────────────────


async def test_get_or_create_collection_is_idempotent(store: Store) -> None:
    first = await store.get_or_create_collection("ws", "docs")
    again = await store.get_or_create_collection("ws", "docs", collection_id="ignored")
    other_ws = await store.get_or_create_collection("ws2", "docs")
    assert first == again != other_ws
    assert [c["id"] for c in await store.list_collections("ws")] == [first]
    assert len(await store.list_collections()) == 2


# ── Documents, chunks, retrieval ──────────────────────────────────────────


async def test_retrieve_ranks_by_cosine_and_scopes_to_collection(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    other = await store.get_or_create_collection("ws", "b")
    await store.index_document(
        cid,
        "/doc.md",
        {"k": "v"},
        [_chunk(0, "exact"), _chunk(1, "near"), _chunk(2, "far")],
        [_vec(1, 0), _vec(1, 0.2), _vec(0, 1)],
    )
    await store.index_document(other, "/other.md", None, [_chunk(0, "elsewhere")], [_vec(1, 0)])

    results = await store.retrieve(_vec(1, 0), cid, k=2)

    assert [r["content"] for r in results] == ["exact", "near"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["document_path"] == "/doc.md"


async def test_retrieve_filters_modality_and_defaults_to_text(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    await store.index_document(
        cid, None, None, [_chunk(0, "words"), _chunk(1, "picture", "image")], [_vec(1), _vec(1)]
    )
    assert [r["content"] for r in await store.retrieve(_vec(1), cid)] == ["words"]
    images = await store.retrieve(_vec(1), cid, modalities=["image"])
    assert [r["content"] for r in images] == ["picture"]
    assert await store.retrieve(_vec(1), cid, modalities=[]) == []


async def test_document_metadata_and_object_fields_round_trip(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    doc_id, n = await store.insert_document_with_chunks(
        cid,
        "/f.pdf",
        {"src": "upload"},
        [_chunk(0, "x")],
        [_vec(1)],
        object_key="k/1",
        source_mime="application/pdf",
        file_size=10,
        file_hash="abc",
    )
    assert n == 1
    doc = await store.get_document(doc_id)
    assert doc is not None
    assert doc["metadata"] == {"src": "upload"} and doc["object_key"] == "k/1"
    assert (await store.list_documents(cid))[0]["id"] == doc_id
    assert await store.find_document_by_hash(cid, "abc") == doc_id
    assert await store.find_document_by_hash(cid, "zzz") is None


async def test_same_hash_twice_in_a_collection_is_rejected(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    await store.insert_document_with_chunks(cid, None, None, [], [], file_hash="abc")
    with pytest.raises(sqlite3.IntegrityError):
        await store.insert_document_with_chunks(cid, None, None, [], [], file_hash="abc")


async def test_failed_insert_leaves_no_partial_document(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    with pytest.raises(ValueError):
        await store.insert_document_with_chunks(
            cid, None, None, [_chunk(0, "a"), _chunk(1, "b")], [_vec(1)]
        )
    assert await store.list_documents(cid) == []
    assert _count(store, "chunk_vectors") == 0


async def test_deleting_a_document_removes_its_chunks_and_vectors(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    doc_id, _ = await store.insert_document_with_chunks(
        cid, None, None, [_chunk(0, "a")], [_vec(1)], object_key="obj"
    )
    assert await store.delete_document(doc_id) == "obj"
    assert await store.delete_document(doc_id) is None
    assert _count(store, "chunks") == 0
    assert _count(store, "chunk_vectors") == 0
    assert await store.retrieve(_vec(1), cid) == []


async def test_deleting_a_collection_cascades_to_vectors(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    await store.index_document(
        cid, None, None, [_chunk(0, "a"), _chunk(1, "b")], [_vec(1), _vec(0, 1)]
    )
    await store.create_ingest_job(cid)
    assert await store.delete_collection(cid) is True
    assert await store.delete_collection(cid) is False
    for table in ("documents", "chunks", "chunk_vectors", "ingest_jobs"):
        assert _count(store, table) == 0, table


# ── Ingest jobs ───────────────────────────────────────────────────────────


async def test_ingest_job_lifecycle(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    job_id = await store.create_ingest_job(cid, "f.pdf", "k", "hash")
    job = await store.get_ingest_job(job_id)
    assert job is not None and job["status"] == "pending" and job["file_hash"] == "hash"

    await store.update_ingest_job(job_id, "failed", error_message="boom")
    await store.update_ingest_job(job_id, "processing")
    job = await store.get_ingest_job(job_id)
    assert job is not None and job["status"] == "processing" and job["error_message"] is None

    doc_id, _ = await store.index_document(cid, None, None, [], [])
    await store.update_ingest_job(job_id, "completed", document_id=doc_id)
    [listed] = await store.list_ingest_jobs(collection_id=cid, status="completed")
    assert listed["document_id"] == doc_id
    assert await store.list_ingest_jobs(status="pending") == []


async def test_recover_orphan_jobs_requeues_only_stale_heartbeats(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    stale = await store.create_ingest_job(cid)
    fresh = await store.create_ingest_job(cid)
    for job in (stale, fresh):
        await store.update_ingest_job(job, "processing")
        await store.bump_heartbeat(job)
    old = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    store._execute(
        lambda c: c.execute("UPDATE ingest_jobs SET last_heartbeat = ? WHERE id = ?", (old, stale))
    )

    assert await store.recover_orphan_jobs(stale_seconds=60) == 1
    recovered = await store.get_ingest_job(stale)
    assert recovered is not None and recovered["status"] == "pending"
    still = await store.get_ingest_job(fresh)
    assert still is not None and still["status"] == "processing"


async def test_delete_old_ingest_jobs_keeps_recent_and_active(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    old_done = await store.create_ingest_job(cid)
    new_done = await store.create_ingest_job(cid)
    old_pending = await store.create_ingest_job(cid)
    for job in (old_done, new_done):
        await store.update_ingest_job(job, "completed")
    long_ago = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    store._execute(
        lambda c: c.execute(
            "UPDATE ingest_jobs SET updated_at = ? WHERE id IN (?, ?)",
            (long_ago, old_done, old_pending),
        )
    )
    assert await store.delete_old_ingest_jobs(retention_days=7) == 1
    assert await store.get_ingest_job(old_done) is None
    assert await store.get_ingest_job(new_done) is not None
    assert await store.get_ingest_job(old_pending) is not None


# ── Facts and signals ─────────────────────────────────────────────────────


async def test_facts_recall_within_workspace_and_update_in_place(store: Store) -> None:
    await store.upsert_fact("f1", "ws", "uses uv", ["tooling"], "session", _vec(1))
    await store.upsert_fact("f2", "ws", "prefers ruff", [], None, _vec(0, 1))
    await store.upsert_fact("f3", "other", "unrelated", [], None, _vec(1))

    [top, second] = await store.recall_facts("ws", _vec(1), k=5)
    assert top["id"] == "f1" and top["tags"] == ["tooling"] and second["id"] == "f2"

    await store.upsert_fact("f1", "ws", "uses uv 0.9", ["tooling"], "session", _vec(0, 0, 1))
    [top, *_] = await store.recall_facts("ws", _vec(0, 0, 1), k=5)
    assert top["id"] == "f1" and top["content"] == "uses uv 0.9"
    assert top["score"] == pytest.approx(1.0)
    assert _count(store, "fact_vectors") == 3


async def test_fact_without_embedding_is_stored_but_not_recallable(store: Store) -> None:
    await store.upsert_fact("f1", "ws", "embedded", [], None, _vec(1))
    await store.upsert_fact("f1", "ws", "embedding dropped", [], None, None)
    assert await store.recall_facts("ws", _vec(1)) == []
    assert _count(store, "facts") == 1
    assert await store.delete_fact("f1") is True
    assert await store.delete_fact("f1") is False


async def test_listing_facts_is_browsable_without_a_query(store: Store) -> None:
    """Recall needs a question. An inspector opens on everything, newest first, pinned on top."""
    await store.upsert_fact("f1", "ws", "oldest", [], None, _vec(1))
    await store.upsert_fact("f2", "ws", "middle", [], None, None)
    await store.upsert_fact("f3", "ws", "newest", [], None, _vec(0, 1))
    await store.upsert_fact("f4", "other", "unrelated", [], None, _vec(1))
    _backdate(store, {"f1": "2026-01-01T00:00:00+00:00", "f2": "2026-02-01T00:00:00+00:00"})

    listed = await store.list_facts("ws")
    assert [f["id"] for f in listed] == ["f3", "f2", "f1"]

    # A fact with no embedding is invisible to recall but must still be listable — otherwise the
    # one memory you cannot search for is also the one you cannot find and delete.
    assert "f2" in [f["id"] for f in listed]

    assert await store.set_fact_pinned("f1", pinned=True) is True
    assert [f["id"] for f in await store.list_facts("ws")] == ["f1", "f3", "f2"]
    assert await store.list_facts("ws", pinned_only=True) == [await store.get_fact("f1")]

    page = await store.list_facts("ws", limit=1, offset=1)
    assert [f["id"] for f in page] == ["f3"]


async def test_a_fact_reports_its_pin_everywhere_it_can_be_read(store: Store) -> None:
    await store.upsert_fact("f1", "ws", "uses uv", ["tooling"], "session", _vec(1))
    await store.set_fact_pinned("f1", pinned=True)

    [recalled] = await store.recall_facts("ws", _vec(1), k=5)
    assert recalled["pinned"] is True
    fact = await store.get_fact("f1")
    assert fact is not None and fact["pinned"] is True and fact["content"] == "uses uv"

    # Rewriting the content leaves the pin alone: pinning is the user's decision about a fact,
    # not part of the fact.
    await store.upsert_fact("f1", "ws", "uses uv 0.9", ["tooling"], "session", _vec(1))
    fact = await store.get_fact("f1")
    assert fact is not None and fact["pinned"] is True

    assert await store.set_fact_pinned("missing", pinned=True) is False
    assert await store.get_fact("missing") is None


async def test_signals_filter_by_type_and_ignore_duplicate_ids(store: Store) -> None:
    await store.record_signal("s1", "ws", "sess", "accepted", "good plan", _vec(1))
    await store.record_signal("s1", "ws", "sess", "rejected", "duplicate id", _vec(0, 1))
    await store.record_signal("s2", "ws", None, "rejected", "bad plan", _vec(1, 0.1))

    all_signals = await store.recall_signals("ws", _vec(1), k=5)
    assert [s["id"] for s in all_signals] == ["s1", "s2"]
    assert all_signals[0]["signal_type"] == "accepted"
    rejected = await store.recall_signals("ws", _vec(1), k=5, signal_type="rejected")
    assert [s["id"] for s in rejected] == ["s2"]


# ── Durability ────────────────────────────────────────────────────────────


async def test_everything_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "engram.db"
    first = Store(path, dimensions=DIMS)
    cid = await first.get_or_create_collection("ws", "a")
    await first.index_document(cid, "/d.md", None, [_chunk(0, "kept")], [_vec(1)])
    await first.close()

    reopened = Store(path, dimensions=DIMS)
    assert [r["content"] for r in await reopened.retrieve(_vec(1), cid)] == ["kept"]
    await reopened.close()
