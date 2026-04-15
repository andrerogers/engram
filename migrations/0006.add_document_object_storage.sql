-- depends: 0005.create_ingest_jobs

-- Object storage fields on documents.
-- file_hash enables per-collection SHA-256 deduplication.
-- Partial unique constraint: same file in the same collection is a no-op.

ALTER TABLE engram.documents
    ADD COLUMN IF NOT EXISTS object_key  TEXT,
    ADD COLUMN IF NOT EXISTS source_mime TEXT,
    ADD COLUMN IF NOT EXISTS file_size   BIGINT,
    ADD COLUMN IF NOT EXISTS file_hash   TEXT;

-- Per-collection content-addressed dedup (only applies when file_hash is set).
CREATE UNIQUE INDEX IF NOT EXISTS idx_engram_documents_collection_hash
    ON engram.documents(collection_id, file_hash)
    WHERE file_hash IS NOT NULL;
