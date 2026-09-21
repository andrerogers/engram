"""Lexical search over FTS5 and its fusion with vector search.

Dense retrieval misses the queries a developer tool gets most — identifiers,
error strings, flag names — because an embedding of ``route_web_socket`` is
about WebSockets, not that function. BM25 over an FTS5 index finds them.
The two are fused by Reciprocal Rank Fusion rather than by blending scores,
because BM25 and cosine similarity are not on a comparable scale; only their
ranks are.

The tokenizer keeps ``_`` inside tokens, so ``workspace_path`` is one token
and an identifier query is an exact match. It was chosen on
``evals/retrieval`` in the brainstack repo: against plain ``unicode61`` and
``trigram`` it scored best on identifiers and no worse elsewhere.
"""

from __future__ import annotations

import re

TOKENIZE = "unicode61 tokenchars '_'"
RRF_K = 60
CANDIDATE_FACTOR = 3


def fts_query(text: str) -> str | None:
    """An FTS5 MATCH expression for free text, or None if it has no searchable terms.

    Each whitespace-separated term is quoted, so FTS5 syntax in the input is
    inert, and the table's tokenizer splits it into a phrase. Terms are ORed:
    a question is a bag of words weighted by BM25, not a conjunction.
    """
    terms = [t for t in text.split() if re.search(r"\w", t)]
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def fuse(*rankings: list[str]) -> list[str]:
    """Reciprocal Rank Fusion: an item's score is the sum of 1 / (60 + rank) over rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda item: -scores[item])
