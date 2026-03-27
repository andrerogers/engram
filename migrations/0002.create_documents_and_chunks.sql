-- depends: 0001.create_collections

CREATE TABLE IF NOT EXISTS engram.documents (
    id              TEXT PRIMARY KEY,
    collection_id   TEXT REFERENCES engram.collections(id) ON DELETE CASCADE,
    path            TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS engram.chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT REFERENCES engram.documents(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    embedding       vector(1536),
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_engram_chunks_doc
    ON engram.chunks(document_id);
