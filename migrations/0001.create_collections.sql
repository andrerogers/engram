-- depends:

CREATE SCHEMA IF NOT EXISTS engram;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS engram.collections (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    name          TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_engram_coll_ws_name
    ON engram.collections(workspace_id, name);
