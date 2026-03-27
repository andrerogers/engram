"""Async PostgreSQL store for Engram — collections, documents, chunks with vectors.

Uses psycopg3 async with a lazy singleton connection. Schema: 'engram'.

Reconnect: each public method calls _run(fn) which catches OperationalError /
InterfaceError, resets the connection singleton, and retries once — surviving
Postgres restarts without requiring a service restart.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"

_T = TypeVar("_T")


class Store:
    """Async Postgres store for the Engram service."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: object | None = None
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> Any:
        import psycopg

        async with self._lock:
            if self._conn is None or getattr(self._conn, "closed", True):
                log.info("engram: opening PostgreSQL connection")
                self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        return self._conn

    async def _run(self, fn: Callable[[Any], Awaitable[_T]]) -> _T:
        """Call fn(conn) with one reconnect retry on transient connection errors."""
        import psycopg

        try:
            conn = await self._get_conn()
            return await fn(conn)
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            log.warning("engram: connection lost (%s) — reconnecting", exc)
            async with self._lock:
                self._conn = None
            conn = await self._get_conn()
            return await fn(conn)

    async def init_db(self) -> None:
        """Run yoyo migrations in a thread executor."""
        dsn = self._dsn
        migrations_dir = str(_MIGRATIONS_DIR)

        def _migrate() -> None:
            from yoyo import get_backend, read_migrations

            yoyo_dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
            backend = get_backend(yoyo_dsn)
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

    # ── Documents + Chunks ────────────────────────────────────────────────

    async def index_document(
        self,
        collection_id: str,
        path: str | None,
        metadata: dict[str, str] | None,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> tuple[str, int]:
        """Store a document and its chunks with embeddings.

        Returns (document_id, chunk_count).
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
                for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                    chunk_id = str(uuid.uuid4())
                    await conn.execute(
                        "INSERT INTO engram.chunks "
                        "(id, document_id, content, chunk_index, embedding) "
                        "VALUES (%s, %s, %s, %s, %s::vector)",
                        (chunk_id, doc_id, chunk_text, i, str(embedding)),
                    )
            return doc_id, len(chunks)

        return await self._run(_do)

    async def retrieve(
        self,
        embedding: list[float],
        collection_id: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        async def _do(conn: Any) -> list[dict[str, Any]]:
            vec_str = str(embedding)
            rows = await (
                await conn.execute(
                    "SELECT c.id, d.path, c.content, "
                    "1 - (c.embedding <=> %s::vector) AS score "
                    "FROM engram.chunks c "
                    "JOIN engram.documents d ON d.id = c.document_id "
                    "WHERE d.collection_id = %s AND c.embedding IS NOT NULL "
                    "ORDER BY c.embedding <=> %s::vector LIMIT %s",
                    (vec_str, collection_id, vec_str, k),
                )
            ).fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "document_path": r[1],
                    "content": r[2],
                    "score": float(r[3]),
                }
                for r in rows
            ]

        return await self._run(_do)
