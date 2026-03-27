"""Embedding client — calls OpenRouter for text-embedding-3-small."""

from __future__ import annotations

import logging

import httpx

from engram.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_EMBEDDINGS_URL,
)

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter (OpenAI-compatible endpoint).

    Automatically splits into sub-batches of EMBEDDING_BATCH_SIZE.
    Returns a list of float vectors, one per input text.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot generate embeddings")

    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        resp = await client.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": batch,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend(item["embedding"] for item in sorted_items)

    return all_embeddings
