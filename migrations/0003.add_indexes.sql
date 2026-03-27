-- depends: 0002.create_documents_and_chunks

CREATE INDEX IF NOT EXISTS idx_engram_documents_collection
    ON engram.documents(collection_id);

CREATE INDEX IF NOT EXISTS idx_engram_chunks_embedding
    ON engram.chunks USING hnsw (embedding vector_cosine_ops);
