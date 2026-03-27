"""Embedding client — calls OpenRouter for text-embedding-3-small.

Retries each batch on 429 / 5xx with exponential backoff: 0.5s → 1.0s → 2.0s (3 attempts).
"""

from __future__ import annotations

import asyncio
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
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_DELAYS = [0.5, 1.0, 2.0]


def _get_client() -> httpx.AsyncClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def _post_with_retry(client: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
    """POST with retry on transient HTTP errors (429 / 5xx)."""
    resp: httpx.Response | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        resp = await client.post(url, **kwargs)  # type: ignore[arg-type]
        if resp.status_code not in _RETRY_STATUSES:
            return resp
        if attempt < len(_RETRY_DELAYS) - 1:
            log.warning(
                "embed: HTTP %s on attempt %d/%d — retrying in %.1fs",
                resp.status_code,
                attempt + 1,
                len(_RETRY_DELAYS),
                delay,
            )
            await asyncio.sleep(delay)
    assert resp is not None
    return resp


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter (OpenAI-compatible endpoint).

    Automatically splits into sub-batches of EMBEDDING_BATCH_SIZE.
    Retries each batch up to 3 times on 429 / 5xx before raising.
    Returns a list of float vectors, one per input text.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot generate embeddings")

    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        resp = await _post_with_retry(
            client,
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
