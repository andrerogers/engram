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
    job = await store.get_ingest_job(job_id)
    assert job is not None and job["status"] == "completed" and job["document_id"] == doc_id


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


# ── Hybrid retrieval ──────────────────────────────────────────────────────


async def test_a_query_finds_an_identifier_the_vectors_rank_last(store: Store) -> None:
    """The case hybrid retrieval exists for: the embedding points elsewhere, the name is exact."""
    cid = await store.get_or_create_collection("ws", "code")
    await store.index_document(
        cid,
        "/a.py",
        None,
        [_chunk(0, "def route_web_socket(conn): ..."), _chunk(1, "websockets overview")],
        [_vec(0, 1), _vec(1, 0)],
    )

    dense_only = await store.retrieve(_vec(1, 0), cid, k=1)
    hybrid = await store.retrieve(_vec(1, 0), cid, k=2, query="route_web_socket")

    assert [r["content"] for r in dense_only] == ["websockets overview"]
    assert hybrid[0]["content"] == "def route_web_socket(conn): ..."
    # score stays cosine similarity even for a chunk found lexically.
    assert hybrid[0]["score"] == pytest.approx(0.0, abs=1e-6)


async def test_an_identifier_is_one_token_not_its_parts(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "code")
    await store.index_document(
        cid, "/a.py", None, [_chunk(0, "workspace path handling")], [_vec(1, 0)]
    )
    # With "_" kept inside tokens, workspace_path is not "workspace" followed by "path".
    hits = store._execute(
        lambda c: c.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH '\"workspace_path\"'"
        ).fetchone()[0]
    )
    assert hits == 0


async def test_lexical_search_stays_inside_the_collection(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    other = await store.get_or_create_collection("ws", "b")
    await store.index_document(cid, "/a.md", None, [_chunk(0, "nothing here")], [_vec(1, 0)])
    await store.index_document(other, "/b.md", None, [_chunk(0, "needle_token")], [_vec(1, 0)])

    results = await store.retrieve(_vec(0, 1), cid, k=5, query="needle_token")

    assert [r["document_path"] for r in results] == ["/a.md"]


async def test_deleting_a_document_removes_it_from_the_lexical_index(store: Store) -> None:
    cid = await store.get_or_create_collection("ws", "a")
    doc, _ = await store.index_document(
        cid, "/a.md", None, [_chunk(0, "ephemeral_marker")], [_vec(1, 0)]
    )
    await store.delete_document(doc)
    assert _count(store, "chunks") == 0
    assert (
        store._execute(
            lambda c: c.execute(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'ephemeral_marker'"
            ).fetchone()[0]
        )
        == 0
    )


async def test_a_fact_with_no_vector_is_reachable_by_its_words(store: Store) -> None:
    """Its embedding call failed, so vector recall can never return it. The words still can."""
    await store.upsert_fact("f1", "ws", "deploys go through ArgoCD", [], None, None)

    assert await store.recall_facts("ws", _vec(1), k=5) == []
    [hit] = await store.recall_facts("ws", _vec(1), k=5, query="ArgoCD")
    assert hit["id"] == "f1" and hit["score"] == 0.0


async def test_editing_a_fact_reindexes_its_words(store: Store) -> None:
    await store.upsert_fact("f1", "ws", "uses poetry", [], None, None)
    await store.upsert_fact("f1", "ws", "uses uv", [], None, None)

    assert await store.recall_facts("ws", _vec(1), query="poetry") == []
    assert [f["id"] for f in await store.recall_facts("ws", _vec(1), query="uv")] == ["f1"]


async def test_lexical_fact_recall_stays_inside_the_workspace(store: Store) -> None:
    await store.upsert_fact("f1", "other", "secret_codename", [], None, None)
    assert await store.recall_facts("ws", _vec(1), query="secret_codename") == []


async def test_upgrading_an_existing_database_indexes_what_is_already_there(
    tmp_path: Path,
) -> None:
    """The migration's backfill: rows written before the lexical index existed are searchable."""
    from engram.store import _migrations

    path = tmp_path / "engram.db"
    old = Store(path, dimensions=DIMS)
    await old.init_db()
    cid = await old.get_or_create_collection("ws", "a")
    await old.index_document(cid, "/a.md", None, [_chunk(0, "legacy_marker")], [_vec(1, 0)])
    await old.upsert_fact("f1", "ws", "legacy_fact_marker", [], None, None)
    # Wind the file back to before the lexical migration, as a database from the last release.
    migrations = _migrations(DIMS)
    old._execute(
        lambda c: [
            c.execute("DROP TRIGGER chunks_fts_insert"),
            c.execute("DROP TRIGGER chunks_fts_delete"),
            c.execute("DROP TRIGGER chunks_fts_update"),
            c.execute("DROP TRIGGER facts_fts_insert"),
            c.execute("DROP TRIGGER facts_fts_delete"),
            c.execute("DROP TRIGGER facts_fts_update"),
            c.execute("DROP TABLE chunks_fts"),
            c.execute("DROP TABLE facts_fts"),
            # Everything the migrations after the lexical one added, so the file really is the
            # older schema. Rewinding the version alone re-runs them, and `ALTER TABLE ... ADD
            # COLUMN` is not something you can run twice.
            c.execute("DROP INDEX facts_project"),
            c.execute("ALTER TABLE facts DROP COLUMN project_id"),
        ]
    )
    # The version before the lexical migration, found by what it creates rather than by
    # counting from the end — which pointed at whichever migration was added last.
    lexical = next(i for i, m in enumerate(migrations) if "chunks_fts USING fts5" in m)
    old._execute(lambda c: c.execute(f"PRAGMA user_version = {lexical}"))
    await old.close()

    upgraded = Store(path, dimensions=DIMS)
    await upgraded.init_db()
    hits = await upgraded.retrieve(_vec(0, 1), cid, k=5, query="legacy_marker")
    facts = await upgraded.recall_facts("ws", _vec(1), query="legacy_fact_marker")
    await upgraded.close()

    assert [h["content"] for h in hits] == ["legacy_marker"]
    assert [f["id"] for f in facts] == ["f1"]


# --- two pools: a project's facts, and the workspace's own -----------------------------------


async def _pools(tmp_path: Path) -> Store:
    store = Store(tmp_path / "engram.db", dimensions=DIMS)
    await store.init_db()
    await store.upsert_fact("shared", "ws", "the team deploys on Fridays", [], None, _vec(1, 0))
    await store.upsert_fact(
        "mine", "ws", "this service owns billing", [], None, _vec(1, 0), project_id="p1"
    )
    await store.upsert_fact(
        "theirs", "ws", "this service owns search", [], None, _vec(1, 0), project_id="p2"
    )
    return store


async def test_a_project_recalls_its_own_facts_and_the_shared_ones(tmp_path: Path) -> None:
    store = await _pools(tmp_path)
    found = await store.recall_facts("ws", _vec(1, 0), k=10, project_id="p1")
    await store.close()
    assert sorted(f["id"] for f in found) == ["mine", "shared"]


async def test_another_projects_facts_are_not_visible(tmp_path: Path) -> None:
    """The pool is what makes two projects in one workspace separate."""
    store = await _pools(tmp_path)
    found = await store.list_facts("ws", project_id="p1")
    await store.close()
    assert "theirs" not in [f["id"] for f in found]


async def test_without_a_project_the_whole_workspace_answers(tmp_path: Path) -> None:
    """What the memory inspector and a voice session with no project ask for."""
    store = await _pools(tmp_path)
    found = await store.list_facts("ws")
    await store.close()
    assert sorted(f["id"] for f in found) == ["mine", "shared", "theirs"]


async def test_a_fact_remembers_which_pool_it_is_in(tmp_path: Path) -> None:
    store = await _pools(tmp_path)
    rows = {f["id"]: f["project_id"] for f in await store.list_facts("ws")}
    await store.close()
    assert rows == {"shared": None, "mine": "p1", "theirs": "p2"}


async def test_the_lexical_half_of_recall_respects_the_pool(tmp_path: Path) -> None:
    """Hybrid recall fuses BM25 with vectors; the filter has to cover both halves."""
    store = await _pools(tmp_path)
    found = await store.recall_facts("ws", _vec(0, 1), k=10, query="owns", project_id="p1")
    await store.close()
    ids = [f["id"] for f in found]
    assert "mine" in ids and "theirs" not in ids


# --- the vector tables' block size -----------------------------------------------------------


def _vector_table_sql(store: Store) -> dict[str, str]:
    return dict(
        store._execute(
            lambda c: c.execute(
                "SELECT name, sql FROM sqlite_master WHERE name IN "
                "('chunk_vectors', 'fact_vectors', 'signal_vectors')"
            ).fetchall()
        )
    )


async def test_a_fresh_database_reserves_small_vector_blocks(store: Store) -> None:
    tables = _vector_table_sql(store)
    assert set(tables) == {"chunk_vectors", "fact_vectors", "signal_vectors"}
    assert all("chunk_size=64" in sql for sql in tables.values())


async def test_upgrading_rebuilds_the_vector_tables_and_keeps_every_vector(
    tmp_path: Path,
) -> None:
    from engram.store import CHUNKS, FACTS, SIGNALS, _migrations

    path = tmp_path / "engram.db"
    old = Store(path, dimensions=DIMS)
    await old.init_db()
    cid = await old.get_or_create_collection("ws", "a")
    await old.index_document(cid, "/a.md", None, [_chunk(0, "kept")], [_vec(1, 0)])
    await old.upsert_fact("f1", "ws", "kept fact", [], None, _vec(0, 1))
    # Wind the file back to the last release: vector tables at vec0's default block size.
    ddl = {t.table: t for t in (CHUNKS, FACTS, SIGNALS)}

    def rewind(c: sqlite3.Connection) -> None:
        for table in ddl.values():
            cols = ", ".join([table._id, table._partition, *table._metadata, "embedding"])
            c.execute(f"CREATE TEMP TABLE keep_{table.table} AS SELECT {cols} FROM {table.table}")
            c.execute(f"DROP TABLE {table.table}")
            c.execute(table.ddl(DIMS))
            c.execute(f"INSERT INTO {table.table} ({cols}) SELECT {cols} FROM keep_{table.table}")
        c.execute(f"PRAGMA user_version = {len(_migrations(DIMS)) - 1}")

    old._execute(rewind)
    assert not any("chunk_size" in sql for sql in _vector_table_sql(old).values())
    await old.close()

    upgraded = Store(path, dimensions=DIMS)
    await upgraded.init_db()
    tables = _vector_table_sql(upgraded)
    hits = await upgraded.retrieve(_vec(1, 0), cid, k=5)
    facts = await upgraded.recall_facts("ws", _vec(0, 1))
    await upgraded.close()

    assert all("chunk_size=64" in sql for sql in tables.values())
    assert [h["content"] for h in hits] == ["kept"]
    assert [f["id"] for f in facts] == ["f1"]


async def test_many_small_collections_stay_small_on_disk(tmp_path: Path) -> None:
    """At full width. vec0's default block made each one-chunk collection cost 6 MB: 222 of them
    were a 1.6 GB engram.db."""
    dims = 1536
    path = tmp_path / "engram.db"
    store = Store(path, dimensions=dims)
    await store.init_db()
    for i in range(20):
        cid = await store.get_or_create_collection("ws", f"c{i}")
        vector = [1.0] + [0.0] * (dims - 1)
        await store.index_document(cid, f"/{i}.md", None, [_chunk(0, f"doc {i}")], [vector])
    await store.close()

    # 20 collections at the default block size measured ~120 MB.
    assert path.stat().st_size < 16 * 1024 * 1024
