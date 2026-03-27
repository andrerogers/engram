"""Configuration from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
ENGRAM_PORT: int = int(os.environ.get("ENGRAM_PORT", "8613"))
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_EMBEDDINGS_URL: str = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536

# Chunking defaults
CHUNK_SIZE_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64
EMBEDDING_BATCH_SIZE: int = 100
