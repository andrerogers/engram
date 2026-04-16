"""Tests for engram.jobs.runner and engram.jobs.ingest."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.jobs.ingest import run_ingest_job
from engram.jobs.runner import Runner
from engram.processors.base import ChunkCandidate, ChunkerKind, Modality

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JOB_ID = "job-abc-123"
_COLLECTION_ID = "coll-1"
_OBJECT_KEY = "uploads/sample.pdf"
_DOC_ID = "doc-xyz"
_FAKE_BYTES = b"fake pdf bytes"

_FAKE_CHUNKS = [
    ChunkCandidate(
        content="chunk one",
        chunk_index=0,
        modality=Modality.TEXT,
        chunker=ChunkerKind.DOCLING_HYBRID,
    ),
    ChunkCandidate(
        content="chunk two",
        chunk_index=1,
        modality=Modality.TEXT,
        chunker=ChunkerKind.DOCLING_HYBRID,
    ),
]


def _make_store(job: dict | None = None) -> AsyncMock:
    store = AsyncMock()
    store.get_ingest_job = AsyncMock(
        return_value=job
        or {
            "id": _JOB_ID,
            "collection_id": _COLLECTION_ID,
            "filename": "sample.pdf",
            "object_key": _OBJECT_KEY,
            "status": "pending",
        }
    )
    store.update_ingest_job = AsyncMock()
    store.insert_document_with_chunks = AsyncMock(return_value=(_DOC_ID, 2))
    store.bump_heartbeat = AsyncMock()
    store.recover_orphan_jobs = AsyncMock(return_value=0)
    store.delete_old_ingest_jobs = AsyncMock(return_value=0)
    return store


def _make_object_store(data: bytes = _FAKE_BYTES) -> AsyncMock:
    obj = AsyncMock()
    obj.get = AsyncMock(return_value=data)
    obj.delete = AsyncMock()
    return obj


def _make_file_processor(chunks: list[ChunkCandidate] = _FAKE_CHUNKS) -> AsyncMock:
    proc = AsyncMock()
    proc.process = AsyncMock(return_value=chunks)
    return proc


def _make_docling_client() -> MagicMock:
    client = MagicMock()
    client.clear_results = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# run_ingest_job — happy path
# ---------------------------------------------------------------------------


async def test_ingest_job_happy_path() -> None:
    store = _make_store()
    obj = _make_object_store()
    proc = _make_file_processor()

    with patch(
        "engram.jobs.ingest.embeddings.embed", AsyncMock(return_value=[[0.1] * 3, [0.2] * 3])
    ):
        await run_ingest_job(_JOB_ID, store, obj, proc)

    store.update_ingest_job.assert_awaited_once_with(
        _JOB_ID, status="completed", document_id=_DOC_ID
    )
    obj.get.assert_awaited_once_with(_OBJECT_KEY)
    proc.process.assert_awaited_once_with(_FAKE_BYTES, "sample.pdf")


async def test_ingest_job_stores_document_with_object_key() -> None:
    store = _make_store()
    obj = _make_object_store()
    proc = _make_file_processor()

    with patch(
        "engram.jobs.ingest.embeddings.embed", AsyncMock(return_value=[[0.1] * 3, [0.2] * 3])
    ):
        await run_ingest_job(_JOB_ID, store, obj, proc)

    call_kwargs = store.insert_document_with_chunks.call_args.kwargs
    assert call_kwargs["collection_id"] == _COLLECTION_ID
    assert call_kwargs["path"] == "sample.pdf"
    assert call_kwargs["object_key"] == _OBJECT_KEY


async def test_ingest_job_empty_chunks_stores_empty_document() -> None:
    """Zero chunks from the processor must still complete — doc stored, no embed call."""
    store = _make_store()
    obj = _make_object_store()
    proc = _make_file_processor(chunks=[])

    with patch("engram.jobs.ingest.embeddings.embed", AsyncMock()) as mock_embed:
        await run_ingest_job(_JOB_ID, store, obj, proc)
        mock_embed.assert_not_awaited()

    store.update_ingest_job.assert_awaited_once_with(
        _JOB_ID, status="completed", document_id=_DOC_ID
    )


async def test_ingest_job_missing_job_is_noop() -> None:
    store = _make_store()
    store.get_ingest_job = AsyncMock(return_value=None)
    obj = _make_object_store()
    proc = _make_file_processor()

    await run_ingest_job(_JOB_ID, store, obj, proc)

    store.update_ingest_job.assert_not_awaited()
    obj.get.assert_not_awaited()


# ---------------------------------------------------------------------------
# run_ingest_job — failure path
# ---------------------------------------------------------------------------


async def test_ingest_job_processor_failure_marks_failed() -> None:
    store = _make_store()
    obj = _make_object_store()
    proc = _make_file_processor()
    proc.process = AsyncMock(side_effect=RuntimeError("Docling crashed"))

    with patch("engram.jobs.ingest.embeddings.embed", AsyncMock()):
        await run_ingest_job(_JOB_ID, store, obj, proc)

    call_kwargs = store.update_ingest_job.call_args.kwargs
    assert call_kwargs["status"] == "failed"
    assert "Docling crashed" in call_kwargs["error_message"]


async def test_ingest_job_processor_failure_deletes_object() -> None:
    store = _make_store()
    obj = _make_object_store()
    proc = _make_file_processor()
    proc.process = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("engram.jobs.ingest.embeddings.embed", AsyncMock()):
        await run_ingest_job(_JOB_ID, store, obj, proc)

    obj.delete.assert_awaited_once_with(_OBJECT_KEY)


async def test_ingest_job_object_store_failure_marks_failed() -> None:
    store = _make_store()
    obj = _make_object_store()
    obj.get = AsyncMock(side_effect=KeyError("object gone"))
    proc = _make_file_processor()

    with patch("engram.jobs.ingest.embeddings.embed", AsyncMock()):
        await run_ingest_job(_JOB_ID, store, obj, proc)

    call_kwargs = store.update_ingest_job.call_args.kwargs
    assert call_kwargs["status"] == "failed"


async def test_ingest_job_store_update_failure_swallowed() -> None:
    """Failure to mark job failed must not propagate — ingest is already broken."""
    store = _make_store()
    store.update_ingest_job = AsyncMock(side_effect=RuntimeError("db gone"))
    obj = _make_object_store()
    proc = _make_file_processor()
    proc.process = AsyncMock(side_effect=RuntimeError("processor broke"))

    with patch("engram.jobs.ingest.embeddings.embed", AsyncMock()):
        await run_ingest_job(_JOB_ID, store, obj, proc)  # must not raise


# ---------------------------------------------------------------------------
# Runner — startup / shutdown
# ---------------------------------------------------------------------------


async def test_runner_startup_initialises_semaphore() -> None:
    runner = Runner(max_concurrent=4, docling=_make_docling_client())
    store = _make_store()
    await runner.startup(store)
    assert runner._sem is not None
    await runner.shutdown_wait(timeout=0.1)


async def test_runner_startup_calls_recover_orphan_jobs() -> None:
    runner = Runner(max_concurrent=4, docling=_make_docling_client())
    store = _make_store()
    await runner.startup(store)
    store.recover_orphan_jobs.assert_awaited_once()
    await runner.shutdown_wait(timeout=0.1)


async def test_runner_schedule_before_startup_raises() -> None:
    runner = Runner(max_concurrent=4, docling=_make_docling_client())
    with pytest.raises(RuntimeError, match="startup"):
        runner.schedule(_JOB_ID, _make_store(), _make_object_store(), _make_file_processor())


# ---------------------------------------------------------------------------
# Runner — happy path scheduling
# ---------------------------------------------------------------------------


async def test_runner_schedule_runs_job() -> None:
    runner = Runner(max_concurrent=4, docling=_make_docling_client())
    store = _make_store()
    await runner.startup(store)

    with patch(
        "engram.jobs.ingest.embeddings.embed",
        AsyncMock(return_value=[[0.1] * 3, [0.2] * 3]),
    ):
        runner.schedule(_JOB_ID, store, _make_object_store(), _make_file_processor())
        await runner.shutdown_wait(timeout=5)

    store.update_ingest_job.assert_awaited_with(_JOB_ID, status="completed", document_id=_DOC_ID)


# ---------------------------------------------------------------------------
# Runner — concurrency cap
# ---------------------------------------------------------------------------


async def test_runner_concurrency_cap() -> None:
    """At most MAX concurrent jobs run simultaneously."""
    max_concurrent = 2
    runner = Runner(max_concurrent=max_concurrent, docling=_make_docling_client())
    store = _make_store()
    await runner.startup(store)

    active_count = 0
    max_observed = 0
    gate = asyncio.Event()  # holds all jobs until we release

    async def slow_ingest(*_args: object, **_kwargs: object) -> None:
        nonlocal active_count, max_observed
        active_count += 1
        max_observed = max(max_observed, active_count)
        await gate.wait()
        active_count -= 1

    with patch("engram.jobs.runner.run_ingest_job", slow_ingest):
        for i in range(max_concurrent + 1):
            store_i = _make_store(
                {
                    "id": f"job-{i}",
                    "collection_id": _COLLECTION_ID,
                    "filename": "f.pdf",
                    "object_key": _OBJECT_KEY,
                    "status": "pending",
                }
            )
            runner.schedule(f"job-{i}", store_i, _make_object_store(), _make_file_processor())

        # Give tasks time to acquire semaphore slots
        await asyncio.sleep(0.05)
        assert max_observed <= max_concurrent
        gate.set()
        await runner.shutdown_wait(timeout=5)


# ---------------------------------------------------------------------------
# Runner — shutdown drain
# ---------------------------------------------------------------------------


async def test_runner_shutdown_wait_drains_tasks() -> None:
    runner = Runner(max_concurrent=4, docling=_make_docling_client())
    store = _make_store()
    await runner.startup(store)

    job_started = asyncio.Event()
    job_can_finish = asyncio.Event()

    async def slow_ingest(*_args: object, **_kwargs: object) -> None:
        job_started.set()
        await job_can_finish.wait()

    with patch("engram.jobs.runner.run_ingest_job", slow_ingest):
        runner.schedule(_JOB_ID, store, _make_object_store(), _make_file_processor())
        await job_started.wait()
        assert len(runner._tasks) == 1
        job_can_finish.set()
        await runner.shutdown_wait(timeout=5)

    assert len(runner._tasks) == 0


async def test_runner_shutdown_wait_with_no_tasks() -> None:
    """shutdown_wait must be safe to call with no scheduled jobs."""
    runner = Runner(max_concurrent=4, docling=_make_docling_client())
    store = _make_store()
    await runner.startup(store)
    await runner.shutdown_wait(timeout=0.1)  # must not raise
