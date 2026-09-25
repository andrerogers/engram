"""The substitute embedding: what the e2e suite embeds with, so a run calls no model.

It has to be deterministic (a run is reproducible), make no network call at all (the point of
it), and still rank by meaning well enough that recall returns the fact that shares the query's
words — a random or constant vector would make every recall test pass or fail by luck.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import engram.app as app_module
from engram import embeddings
from engram.app import app
from engram.store import Store


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture()
def substitute() -> Iterator[None]:
    def no_network() -> None:
        raise AssertionError("the substitute made a network call")

    with (
        patch.object(embeddings, "EMBEDDING_PROVIDER", "substitute"),
        patch.object(embeddings, "OPENROUTER_API_KEY", ""),
        patch.object(embeddings, "_get_client", side_effect=no_network),
    ):
        yield


async def test_it_is_deterministic_unit_length_and_calls_nothing(substitute: None) -> None:
    first, again, empty = await embeddings.embed(["Deploys go out on Fridays", "", ""])
    [repeat] = await embeddings.embed(["Deploys go out on Fridays"])

    assert first == repeat
    assert len(first) == 1536
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)
    # Text with no words still has a direction; a zero vector would make cosine undefined.
    assert again == empty and math.isclose(math.sqrt(sum(x * x for x in empty)), 1.0)


async def test_shared_words_score_higher_than_unrelated_text(substitute: None) -> None:
    query, related, unrelated = await embeddings.embed(
        ["when do deploys go out", "deploys go out on fridays", "the cat sat on the mat"]
    )
    assert _cosine(query, related) > _cosine(query, unrelated)


async def test_an_unknown_provider_is_refused_by_name() -> None:
    with (
        patch.object(embeddings, "EMBEDDING_PROVIDER", "openai"),
        pytest.raises(RuntimeError, match="openai"),
    ):
        await embeddings.embed(["x"])


def test_recall_returns_the_fact_that_shares_the_query_words(
    substitute: None, tmp_path: Path
) -> None:
    """Through the app and a real store, as the e2e suite uses it."""
    store = Store(tmp_path / "engram.db")
    asyncio.run(store.init_db())
    with patch.object(app_module, "_store", store):
        client = TestClient(app)
        for content in (
            "the cat sat on the mat",
            "deploys go out on fridays",
            "tea is best without milk",
        ):
            client.post("/facts", json={"workspace_id": "w", "content": content})
        r = client.get(
            "/facts/recall", params={"workspace_id": "w", "q": "when do deploys go out", "k": 3}
        )

    assert r.json()[0]["content"] == "deploys go out on fridays"


def test_the_key_set_in_settings_is_used_and_wins(monkeypatch) -> None:
    """Hive writes the key a user sets in Settings to <home>/credentials.json. Engram read only
    OPENROUTER_API_KEY, at import — so memory broke for a user who set the key in the panel."""
    from engram import config

    monkeypatch.setattr(embeddings, "OPENROUTER_API_KEY", "sk-or-from-the-env-9999")
    path = config.BRAINSTACK_HOME / "credentials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"openrouter": {"api_key": "sk-or-from-settings-1234"}}')
    try:
        assert embeddings._api_key() == "sk-or-from-settings-1234"
        path.unlink()
        assert embeddings._api_key() == "sk-or-from-the-env-9999"
    finally:
        path.unlink(missing_ok=True)
