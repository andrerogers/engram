"""Runner — app-level background job manager.

Owns:
- Concurrency cap (asyncio.Semaphore, default MAX_CONCURRENT_INGEST_JOBS=4)
- Per-job heartbeat task (bumps last_heartbeat every 10s)
- Orphan recovery on startup
- Hourly cleanup (docling.clear_results + delete_old_ingest_jobs)
- shutdown_wait(timeout=30s) to drain in-flight jobs on FastAPI lifespan exit

Usage in app.py:
    _runner = Runner(MAX_CONCURRENT_INGEST_JOBS, _docling)

    async with lifespan:
        await _runner.startup(store)
        yield
        await _runner.shutdown_wait()
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from engram.config import INGEST_HEARTBEAT_STALE_SECONDS, INGEST_JOB_RETENTION_DAYS
from engram.jobs.ingest import run_ingest_job

if TYPE_CHECKING:
    from engram.clients.docling import DoclingClient
    from engram.clients.storage.base import ObjectStore
    from engram.processors.base import FileProcessor
    from engram.store import Store

log = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 10  # seconds
_CLEANUP_INTERVAL = 3600  # seconds (hourly)
_SHUTDOWN_TIMEOUT = 30  # seconds


class Runner:
    """App-level ingest job manager."""

    def __init__(self, max_concurrent: int, docling: DoclingClient) -> None:
        self._max_concurrent = max_concurrent
        self._docling = docling
        self._sem: asyncio.Semaphore | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._cleanup_task: asyncio.Task[None] | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def startup(self, store: Store) -> None:
        """Initialise semaphore, recover orphan jobs, start cleanup loop."""
        self._sem = asyncio.Semaphore(self._max_concurrent)
        recovered = await store.recover_orphan_jobs(stale_seconds=INGEST_HEARTBEAT_STALE_SECONDS)
        if recovered:
            log.info("runner: recovered %d orphan job(s) → pending", recovered)
        self._cleanup_task = asyncio.create_task(self._hourly_cleanup(store), name="engram-cleanup")

    async def shutdown_wait(self, timeout: float = _SHUTDOWN_TIMEOUT) -> None:
        """Cancel the cleanup loop and drain all in-flight jobs (best-effort)."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        if self._tasks:
            log.info("runner: waiting up to %ss for %d job(s)", timeout, len(self._tasks))
            _done, pending = await asyncio.wait(self._tasks, timeout=timeout)
            if pending:
                log.warning("runner: %d job(s) still running after shutdown timeout", len(pending))

    # ── Scheduling ─────────────────────────────────────────────────────────

    def schedule(
        self,
        job_id: str,
        store: Store,
        object_store: ObjectStore,
        file_processor: FileProcessor,
    ) -> None:
        """Enqueue *job_id* to run as soon as a semaphore slot is free."""
        if self._sem is None:
            raise RuntimeError("Runner.startup() must be called before schedule()")
        task: asyncio.Task[None] = asyncio.create_task(
            self._run(job_id, store, object_store, file_processor),
            name=f"ingest-{job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ── Internal ───────────────────────────────────────────────────────────

    async def _run(
        self,
        job_id: str,
        store: Store,
        object_store: ObjectStore,
        file_processor: FileProcessor,
    ) -> None:
        """Acquire semaphore, run heartbeat + ingest concurrently, release."""
        assert self._sem is not None
        async with self._sem:
            heartbeat: asyncio.Task[None] = asyncio.create_task(
                self._heartbeat(job_id, store), name=f"heartbeat-{job_id}"
            )
            try:
                await run_ingest_job(job_id, store, object_store, file_processor)
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat

    async def _heartbeat(self, job_id: str, store: Store) -> None:
        """Bump last_heartbeat every 10s until cancelled."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            with suppress(Exception):
                await store.bump_heartbeat(job_id)

    async def _hourly_cleanup(self, store: Store) -> None:
        """Every hour: clear Docling result cache + purge old terminal jobs."""
        while True:
            await asyncio.sleep(_CLEANUP_INTERVAL)
            with suppress(Exception):
                await self._docling.clear_results()
            with suppress(Exception):
                deleted = await store.delete_old_ingest_jobs(
                    retention_days=INGEST_JOB_RETENTION_DAYS
                )
                if deleted:
                    log.info("runner: purged %d old ingest job(s)", deleted)
