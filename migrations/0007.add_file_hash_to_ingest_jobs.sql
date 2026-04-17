-- depends: 0006.add_document_object_storage

ALTER TABLE engram.ingest_jobs ADD COLUMN IF NOT EXISTS file_hash TEXT;
