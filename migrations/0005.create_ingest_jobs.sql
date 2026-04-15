-- depends: 0004.add_modality_and_chunker

CREATE TABLE IF NOT EXISTS engram.ingest_jobs (
    id              TEXT PRIMARY KEY,
    collection_id   TEXT NOT NULL REFERENCES engram.collections(id) ON DELETE CASCADE,
    document_id     TEXT REFERENCES engram.documents(id) ON DELETE SET NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    filename        TEXT,
    object_key      TEXT,
    error_message   TEXT,
    last_heartbeat  TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_engram_ingest_jobs_collection
    ON engram.ingest_jobs(collection_id);

CREATE INDEX IF NOT EXISTS idx_engram_ingest_jobs_status
    ON engram.ingest_jobs(status);
