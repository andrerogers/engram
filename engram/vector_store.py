"""VectorStore protocol and its sqlite-vec implementation (TDD §5.2).

Nothing else in Engram touches sqlite-vec. Each index is a ``vec0`` virtual table keyed by the
owning row's id, partitioned by the scope every query filters on (collection or workspace), with
optional metadata columns for exact-match filters. Search is exact k-nearest-neighbour by cosine
distance; sqlite-vec has no approximate index, which is fine at single-user corpus sizes.

Methods take the caller's connection so vector writes commit or roll back with the row they
belong to.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Protocol


class VectorStore(Protocol):
    def upsert(
        self,
        conn: sqlite3.Connection,
        item_id: str,
        partition: str,
        embedding: Sequence[float],
        metadata: Mapping[str, str] | None = None,
    ) -> None: ...

    def delete(self, conn: sqlite3.Connection, item_id: str) -> None: ...

    def similarity_search(
        self,
        conn: sqlite3.Connection,
        partition: str,
        embedding: Sequence[float],
        k: int,
        filters: Mapping[str, Sequence[str]] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(item_id, cosine similarity)`` pairs, most similar first."""
        ...

    def similarity(
        self, conn: sqlite3.Connection, item_id: str, embedding: Sequence[float]
    ) -> float | None:
        """Cosine similarity of one stored item to *embedding*, or None if it has no vector."""
        ...


class SqliteVecStore:
    def __init__(
        self,
        table: str,
        id_column: str,
        partition_column: str,
        metadata_columns: Sequence[str] = (),
    ) -> None:
        self.table = table
        self._id = id_column
        self._partition = partition_column
        self._metadata = tuple(metadata_columns)

    def ddl(self, dimensions: int) -> str:
        columns = [
            f"{self._id} TEXT PRIMARY KEY",
            f"{self._partition} TEXT PARTITION KEY",
            *(f"{m} TEXT" for m in self._metadata),
            f"embedding FLOAT[{dimensions}] distance_metric=cosine",
        ]
        return f"CREATE VIRTUAL TABLE {self.table} USING vec0({', '.join(columns)});"

    def upsert(
        self,
        conn: sqlite3.Connection,
        item_id: str,
        partition: str,
        embedding: Sequence[float],
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        meta = metadata or {}
        if set(meta) != set(self._metadata):
            raise ValueError(f"{self.table} expects metadata {self._metadata}, got {tuple(meta)}")
        self.delete(conn, item_id)  # vec0 has no ON CONFLICT
        columns = [self._id, self._partition, *self._metadata, "embedding"]
        values = [
            item_id,
            partition,
            *(meta[m] for m in self._metadata),
            json.dumps(list(embedding)),
        ]
        conn.execute(
            f"INSERT INTO {self.table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            values,
        )

    def delete(self, conn: sqlite3.Connection, item_id: str) -> None:
        conn.execute(f"DELETE FROM {self.table} WHERE {self._id} = ?", (item_id,))

    def similarity_search(
        self,
        conn: sqlite3.Connection,
        partition: str,
        embedding: Sequence[float],
        k: int,
        filters: Mapping[str, Sequence[str]] | None = None,
    ) -> list[tuple[str, float]]:
        clauses = [f"{self._partition} = ?"]
        params: list[object] = [json.dumps(list(embedding)), k, partition]
        for column, allowed in (filters or {}).items():
            if column not in self._metadata:
                raise ValueError(f"{self.table} cannot filter on {column}")
            if not allowed:
                return []
            clauses.append(f"{column} IN ({', '.join('?' * len(allowed))})")
            params.extend(allowed)
        rows = conn.execute(
            f"SELECT {self._id}, distance FROM {self.table} "
            f"WHERE embedding MATCH ? AND k = ? AND {' AND '.join(clauses)} "
            "ORDER BY distance",
            params,
        ).fetchall()
        return [(row[0], 1.0 - float(row[1])) for row in rows]

    def similarity(
        self, conn: sqlite3.Connection, item_id: str, embedding: Sequence[float]
    ) -> float | None:
        row = conn.execute(
            f"SELECT vec_distance_cosine(embedding, ?) FROM {self.table} WHERE {self._id} = ?",
            (json.dumps(list(embedding)), item_id),
        ).fetchone()
        return None if row is None else 1.0 - float(row[0])


CHUNKS = SqliteVecStore("chunk_vectors", "chunk_id", "collection_id", ("modality",))
FACTS = SqliteVecStore("fact_vectors", "fact_id", "workspace_id")
SIGNALS = SqliteVecStore("signal_vectors", "signal_id", "workspace_id", ("signal_type",))
