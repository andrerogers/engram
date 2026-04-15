"""Async PostgreSQL store for Engram — collections, documents, chunks with vectors.

Uses psycopg3 async with a connection pool (psycopg-pool). Schema: 'engram'.
The pool handles reconnection automatically.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from engram.processors.base import ChunkCandidate

log = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"

_T = TypeVar("_T")


class Store:
    """Async Postgres store for the Engram service."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                from psycopg_pool import AsyncConnectionPool

                log.info("engram: opening PostgreSQL connection pool")
                pool = AsyncConnectionPool(
                    self._dsn,
                    min_size=2,
                    max_size=10,
                    open=False,
                    kwargs={"autocommit": True},
                )
                await pool.open()
                self._pool = pool
        return self._pool

    async def _run(self, fn: Callable[[Any], Awaitable[_T]]) -> _T:
        """Call fn(conn) with a connection from the pool."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            return await fn(conn)

    async def close(self) -> None:
        """Shut down the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def init_db(self) -> None:
        """Run yoyo migrations in a thread executor."""
        dsn = self._dsn
        migrations_dir = str(_MIGRATIONS_DIR)

        def _migrate() -> None:
            from yoyo import get_backend, read_migrations

            yoyo_dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
            backend = get_backend(yoyo_dsn, migration_table="_engram_yoyo_migrations")
            # Isolate log + version tables per-service so they don't collide
            # when Hive/Mneme/Engram share the same Postgres database.
            backend.log_table = "_engram_yoyo_log"
            backend.version_table = "_engram_yoyo_version"
            migrations = read_migrations(migrations_dir)
            with backend.lock():
                backend.apply_migrations(backend.to_apply(migrations))

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _migrate)
        log.info("engram: migrations applied")

    # ── Collections ───────────────────────────────────────────────────────

    async def get_or_create_collection(
        self, workspace_id: str, name: str, collection_id: str | None = None
    ) -> str:
        cid = collection_id or str(uuid.uuid4())

        async def _do(conn: Any) -> str:
            row = await (
                await conn.execute(
                    "SELECT id FROM engram.collections WHERE workspace_id = %s AND name = %s",
                    (workspace_id, name),
                )
            ).fetchone()
            if row:
                return str(row[0])
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO engram.collections (id, workspace_id, name) VALUES (%s, %s, %s)",
                    (cid, workspace_id, name),
                )
            return cid

        return await self._run(_do)

    async def list_collections(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        async def _do(conn: Any) -> list[dict[str, Any]]:
            if workspace_id:
                rows = await (
                    await conn.execute(
                        "SELECT id, workspace_id, name, created_at FROM engram.collections "
                        "WHERE workspace_id = %s ORDER BY created_at DESC",
                        (workspace_id,),
                    )
                ).fetchall()
            else:
                rows = await (
                    await conn.execute(
                        "SELECT id, workspace_id, name, created_at FROM engram.collections "
                        "ORDER BY created_at DESC",
                    )
                ).fetchall()
            return [
                {
                    "id": r[0],
                    "workspace_id": r[1],
                    "name": r[2],
                    "created_at": r[3].isoformat(),
                }
                for r in rows
            ]

        return await self._run(_do)

    async def delete_collection(self, collection_id: str) -> bool:
        async def _do(conn: Any) -> bool:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM engram.collections WHERE id = %s", (collection_id,)
                )
            return bool(result.rowcount > 0)

        return await self._run(_do)

    # ── Documents ─────────────────────────────────────────────────────────

    async def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        """Return all documents in a collection (no chunk data)."""

        async def _do(conn: Any) -> list[dict[str, Any]]:
            rows = await (
                await conn.execute(
                    "SELECT id, collection_id, path, metadata, created_at "
                    "FROM engram.documents WHERE collection_id = %s ORDER BY created_at DESC",
                    (collection_id,),
                )
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "collection_id": r[1],
                    "path": r[2],
                    "metadata": r[3],
                    "created_at": r[4].isoformat(),
                }
                for r in rows
            ]

        return await self._run(_do)

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Return a single document by ID, or None if not found."""

        async def _do(conn: Any) -> dict[str, Any] | None:
            row = await (
                await conn.execute(
                    "SELECT id, collection_id, path, metadata, created_at "
                    "FROM engram.documents WHERE id = %s",
                    (document_id,),
                )
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "collection_id": row[1],
                "path": row[2],
                "metadata": row[3],
                "created_at": row[4].isoformat(),
            }

        return await self._run(_do)

    async def delete_document(self, document_id: str) -> str | None:
        """Delete a document and its chunks. Returns object_key if present (for caller cleanup).

        Returns None if the document does not exist.
        """

        async def _do(conn: Any) -> str | None:
            row = await (
                await conn.execute(
                    "SELECT object_key FROM engram.documents WHERE id = %s",
                    (document_id,),
                )
            ).fetchone()
            if row is None:
                return None
            object_key: str | None = row[0]
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM engram.documents WHERE id = %s", (document_id,)
                )
            return object_key

        return await self._run(_do)

    # ── Documents + Chunks ────────────────────────────────────────────────

    async def index_document(
        self,
        collection_id: str,
        path: str | None,
        metadata: dict[str, str] | None,
        candidates: list[ChunkCandidate],
        embeddings: list[list[float]],
    ) -> tuple[str, int]:
        """Store a document and its chunks with embeddings.

        Args:
            candidates: ChunkCandidate list from a processor (carries modality/chunker metadata).
            embeddings: Per-candidate embedding vectors (must match len(candidates)).

        Returns:
            (document_id, chunk_count)
        """
        doc_id = str(uuid.uuid4())
        meta_json = _json.dumps(metadata or {})

        async def _do(conn: Any) -> tuple[str, int]:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO engram.documents (id, collection_id, path, metadata) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    (doc_id, collection_id, path, meta_json),
                )
                for candidate, embedding in zip(candidates, embeddings, strict=True):
                    chunk_id = str(uuid.uuid4())
                    await conn.execute(
                        "INSERT INTO engram.chunks "
                        "(id, document_id, content, chunk_index, embedding, "
                        " modality, chunker, chunker_version, media_ref, media_metadata) "
                        "VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s::jsonb)",
                        (
                            chunk_id,
                            doc_id,
                            candidate.content,
                            candidate.chunk_index,
                            str(embedding),
                            candidate.modality,
                            candidate.chunker,
                            candidate.chunker_version,
                            candidate.media_ref,
                            _json.dumps(candidate.media_metadata or {}),
                        ),
                    )
            return doc_id, len(candidates)

        return await self._run(_do)

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
        """Store a document (with optional object-store fields) and its chunks.

        Used by the async ingest job runner (E9) where files have already been
        uploaded to MinIO before chunking begins.

        Returns:
            (document_id, chunk_count)
        """
        doc_id = str(uuid.uuid4())
        meta_json = _json.dumps(metadata or {})

        async def _do(conn: Any) -> tuple[str, int]:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO engram.documents "
                    "(id, collection_id, path, metadata, object_key, source_mime, file_size, file_hash) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s)",
                    (doc_id, collection_id, path, meta_json, object_key, source_mime, file_size, file_hash),
                )
                for candidate, embedding in zip(candidates, embeddings, strict=True):
                    chunk_id = str(uuid.uuid4())
                    await conn.execute(
                        "INSERT INTO engram.chunks "
                        "(id, document_id, content, chunk_index, embedding, "
                        " modality, chunker, chunker_version, media_ref, media_metadata) "
                        "VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s::jsonb)",
                        (
                            chunk_id,
                            doc_id,
                            candidate.content,
                            candidate.chunk_index,
                            str(embedding),
                            candidate.modality,
                            candidate.chunker,
                            candidate.chunker_version,
                            candidate.media_ref,
                            _json.dumps(candidate.media_metadata or {}),
                        ),
                    )
            return doc_id, len(candidates)

        return await self._run(_do)

    # ── Ingest jobs ───────────────────────────────────────────────────────

    async def create_ingest_job(
        self,
        collection_id: str,
        filename: str | None = None,
        object_key: str | None = None,
    ) -> str:
        """Create a new ingest job in 'pending' state. Returns job_id."""
        job_id = str(uuid.uuid4())

        async def _do(conn: Any) -> str:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO engram.ingest_jobs "
                    "(id, collection_id, filename, object_key, status) "
                    "VALUES (%s, %s, %s, %s, 'pending')",
                    (job_id, collection_id, filename, object_key),
                )
            return job_id

        return await self._run(_do)

    async def get_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a job row or None."""

        async def _do(conn: Any) -> dict[str, Any] | None:
            row = await (
                await conn.execute(
                    "SELECT id, collection_id, document_id, status, filename, "
                    "object_key, error_message, last_heartbeat, created_at, updated_at "
                    "FROM engram.ingest_jobs WHERE id = %s",
                    (job_id,),
                )
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "collection_id": row[1],
                "document_id": row[2],
                "status": row[3],
                "filename": row[4],
                "object_key": row[5],
                "error_message": row[6],
                "last_heartbeat": row[7].isoformat() if row[7] else None,
                "created_at": row[8].isoformat(),
                "updated_at": row[9].isoformat(),
            }

        return await self._run(_do)

    async def update_ingest_job(
        self,
        job_id: str,
        status: str,
        document_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update job status (and optionally document_id / error_message)."""

        async def _do(conn: Any) -> None:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE engram.ingest_jobs SET status = %s, document_id = COALESCE(%s, document_id), "
                    "error_message = COALESCE(%s, error_message), updated_at = now() "
                    "WHERE id = %s",
                    (status, document_id, error_message, job_id),
                )

        await self._run(_do)

    async def bump_heartbeat(self, job_id: str) -> None:
        """Update last_heartbeat to now() — called by the worker every ~10s."""

        async def _do(conn: Any) -> None:
            await conn.execute(
                "UPDATE engram.ingest_jobs SET last_heartbeat = now() WHERE id = %s",
                (job_id,),
            )

        await self._run(_do)

    async def list_ingest_jobs(
        self,
        collection_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List jobs, optionally filtered by collection and/or status."""

        async def _do(conn: Any) -> list[dict[str, Any]]:
            filters: list[str] = []
            params: list[Any] = []
            if collection_id:
                filters.append("collection_id = %s")
                params.append(collection_id)
            if status:
                filters.append("status = %s")
                params.append(status)
            where = "WHERE " + " AND ".join(filters) if filters else ""
            rows = await (
                await conn.execute(
                    f"SELECT id, collection_id, document_id, status, filename, "
                    f"object_key, error_message, last_heartbeat, created_at, updated_at "
                    f"FROM engram.ingest_jobs {where} ORDER BY created_at DESC",
                    params,
                )
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "collection_id": r[1],
                    "document_id": r[2],
                    "status": r[3],
                    "filename": r[4],
                    "object_key": r[5],
                    "error_message": r[6],
                    "last_heartbeat": r[7].isoformat() if r[7] else None,
                    "created_at": r[8].isoformat(),
                    "updated_at": r[9].isoformat(),
                }
                for r in rows
            ]

        return await self._run(_do)

    async def delete_old_ingest_jobs(self, retention_days: int = 7) -> int:
        """Delete completed/failed jobs older than retention_days. Returns count deleted."""

        async def _do(conn: Any) -> int:
            result = await conn.execute(
                "DELETE FROM engram.ingest_jobs "
                "WHERE status IN ('completed', 'failed') "
                "AND updated_at < now() - interval '1 day' * %s",
                (retention_days,),
            )
            return int(result.rowcount)

        return await self._run(_do)

    async def recover_orphan_jobs(self, stale_seconds: int = 60) -> int:
        """Re-queue jobs stuck in 'processing' with a stale heartbeat.

        A job is stale if its last_heartbeat is older than stale_seconds ago
        (using DB now() to avoid app-clock drift). Returns count recovered.
        """

        async def _do(conn: Any) -> int:
            result = await conn.execute(
                "UPDATE engram.ingest_jobs SET status = 'pending', "
                "error_message = 'recovered: worker heartbeat stale', updated_at = now() "
                "WHERE status = 'processing' "
                "AND last_heartbeat < now() - interval '1 second' * %s",
                (stale_seconds,),
            )
            return int(result.rowcount)

        return await self._run(_do)

    async def _sweep_orphan_objects(
        self, object_store: object, log: Any = None
    ) -> int:
        """Delete object-store keys that have no matching job or document.

        Called on startup to clean up partial uploads from crashed workers.
        Returns the count of swept objects.

        Note: This requires a concrete object_store implementation — the store
        itself has no dependency on the ObjectStore ABC to keep the layers clean.
        The caller (lifespan) passes the configured store.
        """
        # Placeholder implementation — full sweep requires listing object keys
        # from MinIO and cross-referencing against DB. This will be fleshed out
        # in E9 (job runner + lifespan wiring) when both stores are available.
        return 0

    async def retrieve(
        self,
        embedding: list[float],
        collection_id: str,
        k: int = 5,
        modalities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search chunks in a collection.

        Args:
            modalities: Filter to these modalities (default: ["text"]).
        """
        effective_modalities = modalities if modalities is not None else ["text"]

        async def _do(conn: Any) -> list[dict[str, Any]]:
            vec_str = str(embedding)
            # Build modality placeholder list: (%s, %s, ...)
            placeholders = ", ".join(["%s"] * len(effective_modalities))
            rows = await (
                await conn.execute(
                    f"SELECT c.id, d.path, c.content, c.modality, c.chunker, "
                    f"1 - (c.embedding <=> %s::vector) AS score "
                    f"FROM engram.chunks c "
                    f"JOIN engram.documents d ON d.id = c.document_id "
                    f"WHERE d.collection_id = %s AND c.embedding IS NOT NULL "
                    f"AND c.modality IN ({placeholders}) "
                    f"ORDER BY c.embedding <=> %s::vector LIMIT %s",
                    (vec_str, collection_id, *effective_modalities, vec_str, k),
                )
            ).fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "document_path": r[1],
                    "content": r[2],
                    "modality": r[3],
                    "chunker": r[4],
                    "score": float(r[5]),
                }
                for r in rows
            ]

        return await self._run(_do)
