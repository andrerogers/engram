-- depends: 0003.add_indexes

-- Multimodal-ready schema: add modality, chunker, and media fields to chunks.
-- The partial HNSW index on text chunks keeps retrieval fast without indexing
-- non-text modalities (image/audio/video embeddings land in future phases).

ALTER TABLE engram.chunks
    ADD COLUMN IF NOT EXISTS modality        TEXT NOT NULL DEFAULT 'text'
        CHECK (modality IN ('text', 'image', 'audio', 'video')),
    ADD COLUMN IF NOT EXISTS chunker         TEXT NOT NULL DEFAULT 'tiktoken-fallback',
    ADD COLUMN IF NOT EXISTS chunker_version TEXT,
    ADD COLUMN IF NOT EXISTS media_ref       TEXT,
    ADD COLUMN IF NOT EXISTS media_metadata  JSONB DEFAULT '{}';

-- Partial HNSW index: only text chunks are vector-searched in Phase 1.
-- pgvector >= 0.7.0 required for partial index on HNSW.
DROP INDEX IF EXISTS idx_engram_chunks_embedding;

CREATE INDEX IF NOT EXISTS idx_engram_chunks_embedding_text
    ON engram.chunks USING hnsw (embedding vector_cosine_ops)
    WHERE modality = 'text';
