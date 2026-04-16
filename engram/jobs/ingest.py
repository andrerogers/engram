"""run_ingest_job — pure ingest execution logic.

Fetches file bytes from object storage, chunks via Docling, embeds, stores,
then marks the job completed.  On any failure the job is marked failed and
the object is best-effort deleted (to avoid orphaned storage).

Called by the Runner — callers must already hold the semaphore.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from engram import embeddings

if TYPE_CHECKING:
    from engram.clients.storage.base import ObjectStore
    from engram.processors.base import FileProcessor
    from engram.store import Store

log = logging.getLogger(__name__)


async def run_ingest_job(
    job_id: str,
    store: Store,
    object_store: ObjectStore,
    file_processor: FileProcessor,
) -> None:
    """Execute one ingest job end-to-end.

    Pipeline:
        get_ingest_job → get object bytes → chunk → embed →
        insert_document_with_chunks → mark completed

    On any exception: marks the job failed, best-effort deletes the object,
    then swallows the exception (the Runner does not need to handle it).
    """
    job = await store.get_ingest_job(job_id)
    if job is None:
        log.error("run_ingest_job: job %s not found — skipping", job_id)
        return

    object_key: str | None = job.get("object_key")
    filename: str = job.get("filename") or object_key or job_id
    collection_id: str = job["collection_id"]

    try:
        data = await object_store.get(object_key or "")
        candidates = await file_processor.process(data, filename)

        if candidates:
            vecs = await embeddings.embed([c.content for c in candidates])
        else:
            vecs = []

        doc_id, chunk_count = await store.insert_document_with_chunks(
            collection_id=collection_id,
            path=filename,
            metadata={},
            candidates=candidates,
            embeddings=vecs,
            object_key=object_key,
        )
        await store.update_ingest_job(job_id, status="completed", document_id=doc_id)
        log.info("ingest job %s completed: doc=%s chunks=%d", job_id, doc_id, chunk_count)

    except Exception as exc:
        log.error("ingest job %s failed: %s", job_id, exc)
        with suppress(Exception):
            await store.update_ingest_job(job_id, status="failed", error_message=str(exc))
        if object_key:
            with suppress(Exception):
                await object_store.delete(object_key)
