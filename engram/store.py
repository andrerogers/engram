"""SQLite store for Engram — collections, documents, chunks, ingest jobs, facts, signals.

One file (``ENGRAM_DB_PATH``, default ``~/.brainstack/engram.db``), opened lazily with the
sqlite-vec extension loaded. Embeddings live in vec0 tables behind ``engram.vector_store``;
triggers remove a row's vector when the row is deleted, including by cascade. One connection is
shared behind a lock and every public method hops to a worker thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import sqlite_vec

from engram import telemetry
from engram.clients.storage.base import ObjectStore
from engram.hybrid import CANDIDATE_FACTOR, TOKENIZE, fts_query, fuse
from engram.processors.base import ChunkCandidate
from engram.vector_store import CHUNKS, FACTS, SIGNALS

log = logging.getLogger(__name__)

_T = TypeVar("_T")


def _migrations(dimensions: int) -> tuple[str, ...]:
    """Ordered schema versions, tracked with PRAGMA user_version. Never edit a shipped entry."""
    return (
        f"""
        CREATE TABLE collections (
            id           TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            name         TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            UNIQUE (workspace_id, name)
        );
        CREATE TABLE documents (
            id            TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            path          TEXT,
            metadata      TEXT NOT NULL DEFAULT '{{}}',
            object_key    TEXT,
            source_mime   TEXT,
            file_size     INTEGER,
            file_hash     TEXT,
            created_at    TEXT NOT NULL
        );
        CREATE INDEX documents_collection ON documents (collection_id);
        CREATE UNIQUE INDEX documents_collection_hash
            ON documents (collection_id, file_hash) WHERE file_hash IS NOT NULL;
        CREATE TABLE chunks (
            id              TEXT PRIMARY KEY,
            document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content         TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            modality        TEXT NOT NULL DEFAULT 'text'
                CHECK (modality IN ('text', 'image', 'audio', 'video')),
            chunker         TEXT NOT NULL DEFAULT 'tiktoken-fallback',
            chunker_version TEXT,
            media_ref       TEXT,
            media_metadata  TEXT NOT NULL DEFAULT '{{}}',
            created_at      TEXT NOT NULL
        );
        CREATE INDEX chunks_document ON chunks (document_id);
        CREATE TABLE ingest_jobs (
            id             TEXT PRIMARY KEY,
            collection_id  TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            document_id    TEXT REFERENCES documents(id) ON DELETE SET NULL,
            status         TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
            filename       TEXT,
            object_key     TEXT,
            file_hash      TEXT,
            error_message  TEXT,
            last_heartbeat TEXT,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL
        );
        CREATE INDEX ingest_jobs_collection ON ingest_jobs (collection_id);
        CREATE INDEX ingest_jobs_status ON ingest_jobs (status);
        CREATE TABLE facts (
            id           TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            content      TEXT NOT NULL,
            tags         TEXT NOT NULL DEFAULT '[]',
            source       TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX facts_workspace ON facts (workspace_id);
        CREATE TABLE signals (
            id           TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            session_id   TEXT,
            signal_type  TEXT NOT NULL,
            content      TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );
        CREATE INDEX signals_workspace ON signals (workspace_id);
        {CHUNKS.ddl(dimensions)}
        {FACTS.ddl(dimensions)}
        {SIGNALS.ddl(dimensions)}
        CREATE TRIGGER chunks_drop_vector AFTER DELETE ON chunks
            BEGIN DELETE FROM chunk_vectors WHERE chunk_id = old.id; END;
        CREATE TRIGGER facts_drop_vector AFTER DELETE ON facts
            BEGIN DELETE FROM fact_vectors WHERE fact_id = old.id; END;
        CREATE TRIGGER signals_drop_vector AFTER DELETE ON signals
            BEGIN DELETE FROM signal_vectors WHERE signal_id = old.id; END;
        """,
        """
        ALTER TABLE facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
        """,
        # Lexical indexes for hybrid retrieval (engram.hybrid). External-content tables keyed on
        # the implicit rowid: VACUUM may renumber rowids of a table without an INTEGER PRIMARY
        # KEY, so anything that vacuums must follow with INSERT INTO <x>_fts(<x>_fts)
        # VALUES ('rebuild').
        f"""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, content='chunks', content_rowid='rowid', tokenize="{TOKENIZE}"
        );
        INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild');
        CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts (rowid, content) VALUES (new.rowid, new.content);
        END;
        CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts (chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
        END;
        CREATE TRIGGER chunks_fts_update AFTER UPDATE OF content ON chunks BEGIN
            INSERT INTO chunks_fts (chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            INSERT INTO chunks_fts (rowid, content) VALUES (new.rowid, new.content);
        END;
        CREATE VIRTUAL TABLE facts_fts USING fts5(
            content, content='facts', content_rowid='rowid', tokenize="{TOKENIZE}"
        );
        INSERT INTO facts_fts (facts_fts) VALUES ('rebuild');
        CREATE TRIGGER facts_fts_insert AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts (rowid, content) VALUES (new.rowid, new.content);
        END;
        CREATE TRIGGER facts_fts_delete AFTER DELETE ON facts BEGIN
            INSERT INTO facts_fts (facts_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
        END;
        CREATE TRIGGER facts_fts_update AFTER UPDATE OF content ON facts BEGIN
            INSERT INTO facts_fts (facts_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            INSERT INTO facts_fts (rowid, content) VALUES (new.rowid, new.content);
        END;
        """,
        # A fact belongs to one project, or to none — which makes it the workspace's, shared by
        # every project in it (TDD §5.1). The vector partition stays the workspace, so one search
        # reaches both pools and the SQL below narrows to the asking project's own plus the
        # shared ones.
        """
        ALTER TABLE facts ADD COLUMN project_id TEXT;
        CREATE INDEX facts_project ON facts (workspace_id, project_id);
        """,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


_FACT_COLUMNS = (
    "SELECT id, workspace_id, content, tags, source, created_at, updated_at, pinned, project_id "
    "FROM facts"
)

# A fact with no project is the workspace's own pool: every project in it sees it. Asking with
# no project means "the workspace and all of it", which is what an inspector or IRIS wants.
_POOL_SQL = "(project_id IS NULL OR project_id = ?)"


def _pool(project_id: str | None) -> tuple[str, tuple[Any, ...]]:
    """The WHERE fragment and parameters that select a project's facts plus the shared pool."""
    return (" AND " + _POOL_SQL, (project_id,)) if project_id else ("", ())


def _fact(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "workspace_id": row[1],
        "content": row[2],
        "tags": json.loads(row[3]),
        "source": row[4],
        "created_at": row[5],
        "updated_at": row[6],
        "pinned": bool(row[7]),
        "project_id": row[8],
    }


def _job(row: tuple[Any, ...], with_hash: bool) -> dict[str, Any]:
    job = {
        "id": row[0],
        "collection_id": row[1],
        "document_id": row[2],
        "status": row[3],
        "filename": row[4],
        "object_key": row[5],
    }
    rest = list(row[6:])
    if with_hash:
        job["file_hash"] = rest.pop(0)
    job["error_message"], job["last_heartbeat"], job["created_at"], job["updated_at"] = rest
    return job


class Store:
    """SQLite store for the Engram service."""

    def __init__(self, path: Path, dimensions: int = 1536) -> None:
        self._path = path
        self._dimensions = dimensions
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self._path, check_same_thread=False, timeout=5.0, isolation_level=None
            )
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            migrations = _migrations(self._dimensions)
            for number, script in enumerate(migrations[version:], start=version + 1):
                conn.executescript(f"BEGIN; {script}; PRAGMA user_version = {number}; COMMIT;")
            self._conn = conn
        return self._conn

    def _execute(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return result

    async def _run(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        return await asyncio.to_thread(self._execute, fn)

    async def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def init_db(self) -> None:
        """Open engram.db and apply migrations."""
        await self._run(lambda c: None)
        log.info("engram: store ready at %s", self._path)

    # ── Collections ───────────────────────────────────────────────────────

    async def get_or_create_collection(
        self, workspace_id: str, name: str, collection_id: str | None = None
    ) -> str:
        cid = collection_id or str(uuid.uuid4())

        def _do(c: sqlite3.Connection) -> str:
            row = c.execute(
                "SELECT id FROM collections WHERE workspace_id = ? AND name = ?",
                (workspace_id, name),
            ).fetchone()
            if row:
                return str(row[0])
            c.execute(
                "INSERT INTO collections (id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
                (cid, workspace_id, name, _now()),
            )
            return cid

        return await self._run(_do)

    async def list_collections(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        def _do(c: sqlite3.Connection) -> list[dict[str, Any]]:
            sql = "SELECT id, workspace_id, name, created_at FROM collections"
            params: tuple[str, ...] = ()
            if workspace_id:
                sql += " WHERE workspace_id = ?"
                params = (workspace_id,)
            rows = c.execute(sql + " ORDER BY created_at DESC", params).fetchall()
            return [
                {"id": r[0], "workspace_id": r[1], "name": r[2], "created_at": r[3]} for r in rows
            ]

        return await self._run(_do)

    async def delete_collection(self, collection_id: str) -> bool:
        return await self._run(
            lambda c: (
                c.execute("DELETE FROM collections WHERE id = ?", (collection_id,)).rowcount > 0
            )
        )

    # ── Documents ─────────────────────────────────────────────────────────

    async def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        """Return all documents in a collection (no chunk data)."""

        def _do(c: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = c.execute(
                "SELECT id, collection_id, path, metadata, created_at FROM documents "
                "WHERE collection_id = ? ORDER BY created_at DESC",
                (collection_id,),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "collection_id": r[1],
                    "path": r[2],
                    "metadata": json.loads(r[3]),
                    "created_at": r[4],
                }
                for r in rows
            ]

        return await self._run(_do)

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Return a single document by ID, or None if not found."""

        def _do(c: sqlite3.Connection) -> dict[str, Any] | None:
            row = c.execute(
                "SELECT id, collection_id, path, metadata, object_key, created_at "
                "FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "collection_id": row[1],
                "path": row[2],
                "metadata": json.loads(row[3]),
                "object_key": row[4],
                "created_at": row[5],
            }

        return await self._run(_do)

    async def find_document_by_hash(self, collection_id: str, file_hash: str) -> str | None:
        """Return the id of a document with the same hash in this collection, or None."""

        def _do(c: sqlite3.Connection) -> str | None:
            row = c.execute(
                "SELECT id FROM documents WHERE collection_id = ? AND file_hash = ? LIMIT 1",
                (collection_id, file_hash),
            ).fetchone()
            return str(row[0]) if row else None

        return await self._run(_do)

    async def delete_document(self, document_id: str) -> str | None:
        """Delete a document and its chunks. Returns object_key if present (for caller cleanup).

        Returns None if the document does not exist.
        """

        def _do(c: sqlite3.Connection) -> str | None:
            row = c.execute(
                "DELETE FROM documents WHERE id = ? RETURNING object_key", (document_id,)
            ).fetchone()
            return row[0] if row else None

        return await self._run(_do)

    # ── Documents + Chunks ────────────────────────────────────────────────

    @staticmethod
    def _insert_chunks(
        c: sqlite3.Connection,
        doc_id: str,
        collection_id: str,
        candidates: list[ChunkCandidate],
        embeddings: list[list[float]],
    ) -> None:
        now = _now()
        for candidate, embedding in zip(candidates, embeddings, strict=True):
            chunk_id = str(uuid.uuid4())
            c.execute(
                "INSERT INTO chunks (id, document_id, content, chunk_index, modality, chunker, "
                "chunker_version, media_ref, media_metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    doc_id,
                    candidate.content,
                    candidate.chunk_index,
                    candidate.modality,
                    candidate.chunker,
                    candidate.chunker_version,
                    candidate.media_ref,
                    json.dumps(candidate.media_metadata or {}),
                    now,
                ),
            )
            CHUNKS.upsert(c, chunk_id, collection_id, embedding, {"modality": candidate.modality})

    async def index_document(
        self,
        collection_id: str,
        path: str | None,
        metadata: dict[str, str] | None,
        candidates: list[ChunkCandidate],
        embeddings: list[list[float]],
    ) -> tuple[str, int]:
        """Store a document and its chunks with embeddings. Returns (document_id, chunk_count)."""
        return await self.insert_document_with_chunks(
            collection_id, path, metadata, candidates, embeddings
        )

    async def insert_document_with_chunks(  # noqa: PLR0913
        self,
        collection_id: str,
        path: str | None,
        metadata: dict[str, str] | None,
        candidates: list[ChunkCandidate],
        embeddings: list[list[float]],
        object_key: str | None = None,
        source_mime: str | None = None,
        file_size: int | None = None,
        file_hash: str | None = None,
    ) -> tuple[str, int]:
        """Store a document (with optional object-store fields) and its chunks, atomically.

        Returns:
            (document_id, chunk_count)
        """
        doc_id = str(uuid.uuid4())

        def _do(c: sqlite3.Connection) -> tuple[str, int]:
            c.execute(
                "INSERT INTO documents (id, collection_id, path, metadata, object_key, "
                "source_mime, file_size, file_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    collection_id,
                    path,
                    json.dumps(metadata or {}),
                    object_key,
                    source_mime,
                    file_size,
                    file_hash,
                    _now(),
                ),
            )
            Store._insert_chunks(c, doc_id, collection_id, candidates, embeddings)
            return doc_id, len(candidates)

        stored = await self._run(_do)
        telemetry.indexed(stored[1])
        return stored

    # ── Ingest jobs ───────────────────────────────────────────────────────

    async def create_ingest_job(
        self,
        collection_id: str,
        filename: str | None = None,
        object_key: str | None = None,
        file_hash: str | None = None,
    ) -> str:
        """Create a new ingest job in 'pending' state. Returns job_id."""
        job_id = str(uuid.uuid4())
        now = _now()
        await self._run(
            lambda c: c.execute(
                "INSERT INTO ingest_jobs (id, collection_id, filename, object_key, file_hash, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (job_id, collection_id, filename, object_key, file_hash, now, now),
            )
        )
        return job_id

    async def get_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a job row or None."""

        def _do(c: sqlite3.Connection) -> dict[str, Any] | None:
            row = c.execute(
                "SELECT id, collection_id, document_id, status, filename, object_key, file_hash, "
                "error_message, last_heartbeat, created_at, updated_at "
                "FROM ingest_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return _job(row, with_hash=True) if row else None

        return await self._run(_do)

    async def update_ingest_job(
        self,
        job_id: str,
        status: str,
        document_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update job status, and set document_id / error_message when provided.

        Passing None for document_id or error_message leaves the existing value unchanged,
        except that a transition to 'processing' clears a stale error_message.
        """
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, _now()]
        if document_id is not None:
            fields.append("document_id = ?")
            params.append(document_id)
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)
        elif status == "processing":
            fields.append("error_message = NULL")
        params.append(job_id)
        await self._run(
            lambda c: c.execute(
                f"UPDATE ingest_jobs SET {', '.join(fields)} WHERE id = ?",
                params,
            )
        )

    async def bump_heartbeat(self, job_id: str) -> None:
        """Update last_heartbeat to now — called by the worker every ~10s."""
        now = _now()
        await self._run(
            lambda c: c.execute(
                "UPDATE ingest_jobs SET last_heartbeat = ? WHERE id = ?", (now, job_id)
            )
        )

    async def list_ingest_jobs(
        self,
        collection_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List jobs newest first, optionally filtered by collection and/or status."""
        filters: list[str] = []
        params: list[Any] = []
        if collection_id:
            filters.append("collection_id = ?")
            params.append(collection_id)
        if status:
            filters.append("status = ?")
            params.append(status)
        where = "WHERE " + " AND ".join(filters) if filters else ""

        def _do(c: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = c.execute(
                "SELECT id, collection_id, document_id, status, filename, object_key, "
                "error_message, last_heartbeat, created_at, updated_at "
                f"FROM ingest_jobs {where} ORDER BY created_at DESC",
                params,
            ).fetchall()
            return [_job(r, with_hash=False) for r in rows]

        return await self._run(_do)

    async def delete_old_ingest_jobs(self, retention_days: int = 7) -> int:
        """Delete completed/failed jobs older than retention_days. Returns count deleted."""
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        return await self._run(
            lambda c: (
                c.execute(
                    "DELETE FROM ingest_jobs WHERE status IN ('completed', 'failed') AND updated_at < ?",
                    (cutoff,),
                ).rowcount
            )
        )

    async def recover_orphan_jobs(self, stale_seconds: int = 60) -> int:
        """Re-queue jobs stuck in 'processing' whose heartbeat is older than stale_seconds.

        Engram and its workers share one process and clock, so the cutoff is computed here.
        Returns count recovered.
        """
        now = datetime.now(UTC)
        cutoff = (now - timedelta(seconds=stale_seconds)).isoformat()
        return await self._run(
            lambda c: (
                c.execute(
                    "UPDATE ingest_jobs SET status = 'pending', "
                    "error_message = 'recovered: worker heartbeat stale', updated_at = ? "
                    "WHERE status = 'processing' AND last_heartbeat < ?",
                    (now.isoformat(), cutoff),
                ).rowcount
            )
        )

    async def _sweep_orphan_objects(self, object_store: ObjectStore) -> int:
        """Delete object-store keys that have no matching job or document.

        Placeholder until the object store can list its keys.
        """
        return 0

    # ── Facts ─────────────────────────────────────────────────────────────

    async def upsert_fact(  # noqa: PLR0913
        self,
        fact_id: str,
        workspace_id: str,
        content: str,
        tags: list[str],
        source: str | None,
        embedding: list[float] | None,
        project_id: str | None = None,
    ) -> None:
        """Insert or update a distilled fact. A fact without an embedding is not recallable."""
        now = _now()

        def _do(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT INTO facts "
                "(id, workspace_id, project_id, content, tags, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET "
                "content = excluded.content, tags = excluded.tags, source = excluded.source, "
                "project_id = excluded.project_id, updated_at = excluded.updated_at",
                (fact_id, workspace_id, project_id, content, json.dumps(tags), source, now, now),
            )
            if embedding is None:
                FACTS.delete(c, fact_id)
            else:
                FACTS.upsert(c, fact_id, workspace_id, embedding)

        await self._run(_do)

    async def recall_facts(
        self,
        workspace_id: str,
        embedding: list[float],
        k: int = 5,
        query: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k facts a project can see: its own, plus the workspace's shared pool.

        With *query*, vector and BM25 results are fused (engram.hybrid) — which also reaches
        facts whose embedding call failed, stored with no vector and invisible to vector search.
        ``score`` is always cosine similarity, 0.0 for a fact that has no vector.
        """
        match = fts_query(query) if query else None
        pool_sql, pool_params = _pool(project_id)

        def _do(c: sqlite3.Connection) -> list[dict[str, Any]]:
            depth = k * CANDIDATE_FACTOR if match else k
            # The vectors are partitioned by workspace, so a project's own facts and the shared
            # ones come back together — and so do other projects', which the pool filter drops.
            with telemetry.search("vector"):
                dense = [i for i, _ in FACTS.similarity_search(c, workspace_id, embedding, depth)]
            if project_id:
                visible = {
                    row[0]
                    for row in c.execute(
                        "SELECT id FROM facts WHERE workspace_id = ?" + pool_sql,
                        (workspace_id, *pool_params),
                    )
                }
                dense = [i for i in dense if i in visible]
            ranked = dense
            if match:
                with telemetry.search("bm25"):
                    lexical = [
                        row[0]
                        for row in c.execute(
                            "SELECT f.id FROM facts_fts x JOIN facts f ON f.rowid = x.rowid "
                            "WHERE facts_fts MATCH ? AND f.workspace_id = ? "
                            + pool_sql.replace("project_id", "f.project_id")
                            + " ORDER BY bm25(facts_fts) LIMIT ?",
                            (match, workspace_id, *pool_params, depth),
                        )
                    ]
                ranked = fuse(dense, lexical)
            results = []
            for fact_id in ranked[:k]:
                r = c.execute(_FACT_COLUMNS + " WHERE id = ?", (fact_id,)).fetchone()
                score = FACTS.similarity(c, fact_id, embedding)
                results.append({**_fact(r), "score": 0.0 if score is None else score})
            return results

        return await self._run(_do)

    async def get_fact(self, fact_id: str) -> dict[str, Any] | None:
        """Return one fact by id, or None."""

        def _do(c: sqlite3.Connection) -> dict[str, Any] | None:
            row = c.execute(_FACT_COLUMNS + " WHERE id = ?", (fact_id,)).fetchone()
            return _fact(row) if row else None

        return await self._run(_do)

    async def list_facts(
        self,
        workspace_id: str,
        *,
        pinned_only: bool = False,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Browse the facts a project can see, pinned first then newest.

        Recall answers a question; this answers "what do you remember about me", which has no
        query, and must reach facts whose embedding failed and which recall can never return.
        """
        pool_sql, pool_params = _pool(project_id)
        where = " WHERE workspace_id = ?" + pool_sql + (" AND pinned = 1" if pinned_only else "")

        def _do(c: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = c.execute(
                _FACT_COLUMNS
                + where
                + " ORDER BY pinned DESC, created_at DESC, id LIMIT ? OFFSET ?",
                (workspace_id, *pool_params, limit, offset),
            ).fetchall()
            return [_fact(r) for r in rows]

        return await self._run(_do)

    async def set_fact_pinned(self, fact_id: str, *, pinned: bool) -> bool:
        """Pin or unpin a fact. Returns True if the fact existed."""
        return await self._run(
            lambda c: (
                c.execute(
                    "UPDATE facts SET pinned = ? WHERE id = ?", (int(pinned), fact_id)
                ).rowcount
                > 0
            )
        )

    async def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID. Returns True if deleted."""
        return await self._run(
            lambda c: c.execute("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount > 0
        )

    # ── Signals ───────────────────────────────────────────────────────────

    async def record_signal(  # noqa: PLR0913
        self,
        signal_id: str,
        workspace_id: str,
        session_id: str | None,
        signal_type: str,
        content: str,
        embedding: list[float] | None,
    ) -> None:
        """Record an outcome quality signal. Recording the same id twice is a no-op."""

        def _do(c: sqlite3.Connection) -> None:
            inserted = c.execute(
                "INSERT INTO signals (id, workspace_id, session_id, signal_type, content, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                (signal_id, workspace_id, session_id, signal_type, content, _now()),
            ).rowcount
            if inserted and embedding is not None:
                SIGNALS.upsert(c, signal_id, workspace_id, embedding, {"signal_type": signal_type})

        await self._run(_do)

    async def recall_signals(
        self,
        workspace_id: str,
        embedding: list[float],
        k: int = 5,
        signal_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k signals nearest to *embedding*."""
        filters = {"signal_type": [signal_type]} if signal_type else None

        def _do(c: sqlite3.Connection) -> list[dict[str, Any]]:
            with telemetry.search("vector"):
                hits = SIGNALS.similarity_search(c, workspace_id, embedding, k, filters)
            results = []
            for signal_id, score in hits:
                r = c.execute(
                    "SELECT id, session_id, signal_type, content, created_at FROM signals "
                    "WHERE id = ?",
                    (signal_id,),
                ).fetchone()
                results.append(
                    {
                        "id": r[0],
                        "session_id": r[1],
                        "signal_type": r[2],
                        "content": r[3],
                        "created_at": r[4],
                        "score": score,
                    }
                )
            return results

        return await self._run(_do)

    async def retrieve(
        self,
        embedding: list[float],
        collection_id: str,
        k: int = 5,
        modalities: list[str] | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search chunks in a collection.

        With *query*, vector and BM25 results are fused (engram.hybrid); without it, this is
        vector search alone. ``score`` is always cosine similarity to *embedding*, including for
        a chunk found only lexically — callers compare it against thresholds.

        Args:
            modalities: Filter to these modalities (default: ["text"]).
        """
        effective_modalities = modalities if modalities is not None else ["text"]
        match = fts_query(query) if query else None

        def _do(c: sqlite3.Connection) -> list[dict[str, Any]]:
            depth = k * CANDIDATE_FACTOR if match else k
            with telemetry.search("vector"):
                dense = [
                    chunk_id
                    for chunk_id, _ in CHUNKS.similarity_search(
                        c, collection_id, embedding, depth, {"modality": effective_modalities}
                    )
                ]
            ranked = dense
            if match and effective_modalities:
                marks = ", ".join("?" * len(effective_modalities))
                with telemetry.search("bm25"):
                    lexical = [
                        row[0]
                        for row in c.execute(
                            "SELECT c.id FROM chunks_fts x JOIN chunks c ON c.rowid = x.rowid "
                            "JOIN documents d ON d.id = c.document_id "
                            "WHERE chunks_fts MATCH ? AND d.collection_id = ? "
                            f"AND c.modality IN ({marks}) "
                            "ORDER BY bm25(chunks_fts) LIMIT ?",
                            (match, collection_id, *effective_modalities, depth),
                        )
                    ]
                ranked = fuse(dense, lexical)
            results = []
            for chunk_id in ranked[:k]:
                score = CHUNKS.similarity(c, chunk_id, embedding) or 0.0
                r = c.execute(
                    "SELECT c.id, d.path, c.content, c.modality, c.chunker FROM chunks c "
                    "JOIN documents d ON d.id = c.document_id WHERE c.id = ?",
                    (chunk_id,),
                ).fetchone()
                results.append(
                    {
                        "chunk_id": r[0],
                        "document_path": r[1],
                        "content": r[2],
                        "modality": r[3],
                        "chunker": r[4],
                        "score": score,
                    }
                )
            return results

        return await self._run(_do)
