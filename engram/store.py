"""Async PostgreSQL store for Engram — collections, documents, chunks with vectors.

Uses psycopg3 async with a lazy singleton connection. Schema: 'engram'.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


class Store:
    """Async Postgres store for the Engram service."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: object | None = None
        self._lock = asyncio.Lock()

    async def _get_conn(self):  # type: ignore[no-untyped-def]
        import psycopg

        async with self._lock:
            if self._conn is None or getattr(self._conn, "closed", True):
                log.info("engram: opening PostgreSQL connection")
                self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        return self._conn

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
        conn = await self._get_conn()
        # Check for existing
        row = await (
            await conn.execute(
                "SELECT id FROM engram.collections WHERE workspace_id = %s AND name = %s",
                (workspace_id, name),
            )
        ).fetchone()
        if row:
            return row[0]
        cid = collection_id or str(uuid.uuid4())
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO engram.collections (id, workspace_id, name) VALUES (%s, %s, %s)",
                (cid, workspace_id, name),
            )
        return cid

    async def list_collections(self, workspace_id: str | None = None) -> list[dict]:
        conn = await self._get_conn()
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

    async def delete_collection(self, collection_id: str) -> bool:
        conn = await self._get_conn()
        async with conn.transaction():
            result = await conn.execute(
                "DELETE FROM engram.collections WHERE id = %s", (collection_id,)
            )
        return result.rowcount > 0  # type: ignore[union-attr]

    # ── Documents + Chunks ────────────────────────────────────────────────

    async def index_document(
        self,
        collection_id: str,
        path: str | None,
        metadata: dict | None,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> tuple[str, int]:
        """Store a document and its chunks with embeddings.

        Returns (document_id, chunk_count).
        """
        conn = await self._get_conn()
        doc_id = str(uuid.uuid4())
        import json as _json

        meta_json = _json.dumps(metadata or {})

        async with conn.transaction():
            await conn.execute(
                "INSERT INTO engram.documents (id, collection_id, path, metadata) "
                "VALUES (%s, %s, %s, %s::jsonb)",
                (doc_id, collection_id, path, meta_json),
            )
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                chunk_id = str(uuid.uuid4())
                await conn.execute(
                    "INSERT INTO engram.chunks (id, document_id, content, chunk_index, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    (chunk_id, doc_id, chunk_text, i, str(embedding)),
                )
        return doc_id, len(chunks)

    async def retrieve(
        self,
        embedding: list[float],
        collection_id: str,
        k: int = 5,
    ) -> list[dict]:
        conn = await self._get_conn()
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
