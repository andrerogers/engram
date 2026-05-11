-- depends: 0007.add_file_hash_to_ingest_jobs

CREATE TABLE IF NOT EXISTS engram.facts (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    content       TEXT NOT NULL,
    tags          TEXT[] DEFAULT '{}',
    source        TEXT,
    embedding     vector(1536),
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_engram_facts_workspace
    ON engram.facts(workspace_id);

CREATE INDEX IF NOT EXISTS idx_engram_facts_embedding
    ON engram.facts USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS engram.signals (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    session_id    TEXT,
    signal_type   TEXT NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector(1536),
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_engram_signals_workspace
    ON engram.signals(workspace_id);

CREATE INDEX IF NOT EXISTS idx_engram_signals_type
    ON engram.signals(signal_type);

CREATE INDEX IF NOT EXISTS idx_engram_signals_embedding
    ON engram.signals USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
