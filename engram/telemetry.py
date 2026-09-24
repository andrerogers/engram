"""Engram's spans and metrics, in one place, and never able to break the request they measure.

A recall is one ``engram.recall`` span with a child per part: ``engram.embed`` for the embedding
call, ``engram.search.vector`` and ``engram.search.bm25`` for the two halves of the search. One
span over the whole of it would repeat what HTTP duration already says; the children are what
name the slow part. The searches run in a worker thread, and ``asyncio.to_thread`` carries the
context there, so they nest under the recall that started them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from optics import span
from optics._metrics import EngramMetrics

log = logging.getLogger(__name__)


def _safely(record: Callable[..., None], **kwargs: Any) -> None:
    try:
        record(**kwargs)
    except Exception:
        # Nothing about *record* in the message: a handler that can itself raise is not a guard.
        log.debug("engram telemetry: a metric could not be recorded", exc_info=True)


@contextmanager
def recall(kind: str, k: int) -> Iterator[None]:
    """One recall — ``chunks``, ``facts`` or ``signals`` — timed from query to results."""
    started = time.monotonic()
    try:
        with span("engram.recall", attributes={"engram.recall.kind": kind, "engram.recall.k": k}):
            yield
    finally:
        _safely(EngramMetrics.record_recall, kind=kind, duration_s=time.monotonic() - started)


@contextmanager
def search(half: str) -> Iterator[None]:
    """One half of a hybrid search: ``vector`` or ``bm25``."""
    with span(f"engram.search.{half}"):
        yield


@contextmanager
def embedding(texts: int) -> Iterator[None]:
    """One embed() call, every batch and retry inside it."""
    started = time.monotonic()
    outcome = "error"
    try:
        with span("engram.embed", attributes={"engram.embed.texts": texts}):
            yield
        outcome = "ok"
    finally:
        _safely(
            EngramMetrics.record_embedding,
            duration_s=time.monotonic() - started,
            outcome=outcome,
        )


def indexed(chunks: int) -> None:
    """Chunks just stored with their embeddings."""
    _safely(EngramMetrics.record_indexed, chunks=chunks)


def ingest_finished(outcome: str) -> None:
    """An ingest job ended: ``completed`` or ``failed``."""
    _safely(EngramMetrics.record_ingest, outcome=outcome)
