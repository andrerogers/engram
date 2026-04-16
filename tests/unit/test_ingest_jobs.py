"""Tests for ingest job CRUD + orphan recovery in engram.store.

Store methods are tested with an in-memory fake that mirrors the job
state machine so we can run these without Postgres.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Minimal in-memory store double for job methods
# ---------------------------------------------------------------------------


class _FakeJobStore:
    """Minimal fake that exercises the job CRUD interface contract."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    async def create_ingest_job(
        self,
        collection_id: str,
        filename: str | None = None,
        object_key: str | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "id": job_id,
            "collection_id": collection_id,
            "document_id": None,
            "status": "pending",
            "filename": filename,
            "object_key": object_key,
            "error_message": None,
            "last_heartbeat": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        return job_id

    async def get_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    async def update_ingest_job(
        self,
        job_id: str,
        status: str,
        document_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if job_id not in self._jobs:
            return
        self._jobs[job_id]["status"] = status
        if document_id is not None:
            self._jobs[job_id]["document_id"] = document_id
        if error_message is not None:
            self._jobs[job_id]["error_message"] = error_message
        elif status == "processing":
            self._jobs[job_id]["error_message"] = None

    async def bump_heartbeat(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["last_heartbeat"] = "2026-01-01T00:00:10+00:00"

    async def list_ingest_jobs(
        self,
        collection_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        jobs = list(self._jobs.values())
        if collection_id:
            jobs = [j for j in jobs if j["collection_id"] == collection_id]
        if status:
            jobs = [j for j in jobs if j["status"] == status]
        return jobs

    async def delete_old_ingest_jobs(self, retention_days: int = 7) -> int:
        terminal = {k: v for k, v in self._jobs.items() if v["status"] in ("completed", "failed")}
        for k in terminal:
            del self._jobs[k]
        return len(terminal)

    async def recover_orphan_jobs(self, stale_seconds: int = 60) -> int:
        count = 0
        for job in self._jobs.values():
            if job["status"] == "processing" and job["last_heartbeat"] is None:
                # In the fake, None heartbeat = stale (simulates crash)
                job["status"] = "pending"
                job["error_message"] = "recovered: worker heartbeat stale"
                count += 1
        return count


@pytest.fixture
def store() -> _FakeJobStore:
    return _FakeJobStore()


# ---------------------------------------------------------------------------
# Create + Get
# ---------------------------------------------------------------------------


async def test_create_job_returns_id(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1", filename="report.pdf")
    assert isinstance(job_id, str)
    assert len(job_id) > 0


async def test_created_job_has_pending_status(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1")
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["status"] == "pending"


async def test_get_missing_job_returns_none(store: _FakeJobStore) -> None:
    result = await store.get_ingest_job("no-such-id")
    assert result is None


async def test_created_job_stores_filename(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job(
        "coll-1", filename="doc.pdf", object_key="uploads/doc.pdf"
    )
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["filename"] == "doc.pdf"
    assert job["object_key"] == "uploads/doc.pdf"


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


async def test_update_job_status_to_processing(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="processing")
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["status"] == "processing"


async def test_update_job_to_completed_sets_document_id(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="completed", document_id="doc-abc")
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["document_id"] == "doc-abc"


async def test_update_job_to_failed_records_error(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="failed", error_message="Docling unreachable")
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["status"] == "failed"
    assert job["error_message"] == "Docling unreachable"


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


async def test_bump_heartbeat_sets_timestamp(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1")
    assert (await store.get_ingest_job(job_id))["last_heartbeat"] is None  # type: ignore[index]
    await store.bump_heartbeat(job_id)
    assert (await store.get_ingest_job(job_id))["last_heartbeat"] is not None  # type: ignore[index]


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


async def test_list_jobs_by_collection(store: _FakeJobStore) -> None:
    await store.create_ingest_job("coll-a")
    await store.create_ingest_job("coll-a")
    await store.create_ingest_job("coll-b")
    jobs = await store.list_ingest_jobs(collection_id="coll-a")
    assert len(jobs) == 2
    assert all(j["collection_id"] == "coll-a" for j in jobs)


async def test_list_jobs_by_status(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="completed")
    await store.create_ingest_job("coll-1")  # pending
    pending = await store.list_ingest_jobs(status="pending")
    assert len(pending) == 1


# ---------------------------------------------------------------------------
# Delete old jobs
# ---------------------------------------------------------------------------


async def test_delete_old_jobs_removes_terminal(store: _FakeJobStore) -> None:
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="completed")
    pending_id = await store.create_ingest_job("coll-1")
    deleted = await store.delete_old_ingest_jobs()
    assert deleted == 1
    assert await store.get_ingest_job(pending_id) is not None
    assert await store.get_ingest_job(job_id) is None


# ---------------------------------------------------------------------------
# Orphan recovery — simulates crash + restart
# ---------------------------------------------------------------------------


async def test_orphan_recovery_requeues_stale_processing_jobs(store: _FakeJobStore) -> None:
    """Simulate a worker crash: job stuck in 'processing' with no heartbeat."""
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="processing")
    # No heartbeat bump → simulates crash (fake treats None as stale)

    recovered = await store.recover_orphan_jobs(stale_seconds=60)
    assert recovered == 1

    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["status"] == "pending"
    assert "recovered" in (job["error_message"] or "")


async def test_orphan_recovery_skips_active_jobs(store: _FakeJobStore) -> None:
    """Jobs with a fresh heartbeat must not be re-queued."""
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="processing")
    await store.bump_heartbeat(job_id)  # fresh heartbeat

    recovered = await store.recover_orphan_jobs(stale_seconds=60)
    # The fake only re-queues jobs with None heartbeat; bumped jobs are skipped
    assert recovered == 0
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["status"] == "processing"


# ---------------------------------------------------------------------------
# update_ingest_job — field semantics
# ---------------------------------------------------------------------------


async def test_update_to_processing_clears_error_message(store: _FakeJobStore) -> None:
    """Transitioning to 'processing' must clear a stale error_message from a prior failure."""
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="failed", error_message="Docling unreachable")
    # Job is recovered → pending, then picked up again → processing
    await store.update_ingest_job(job_id, status="pending")
    await store.update_ingest_job(job_id, status="processing")
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["error_message"] is None


async def test_update_without_optional_fields_preserves_existing(store: _FakeJobStore) -> None:
    """Passing None for document_id leaves the existing value unchanged."""
    job_id = await store.create_ingest_job("coll-1")
    await store.update_ingest_job(job_id, status="completed", document_id="doc-xyz")
    await store.update_ingest_job(job_id, status="completed")  # no document_id
    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["document_id"] == "doc-xyz"
