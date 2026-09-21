"""engram.hybrid — the FTS5 query builder and rank fusion."""

from __future__ import annotations

from engram.hybrid import fts_query, fuse


def test_every_term_is_quoted_so_fts_syntax_in_input_is_inert() -> None:
    assert fts_query('NEAR(a b) OR "x') == '"NEAR(a" OR "b)" OR "OR" OR """x"'


def test_a_query_with_nothing_searchable_is_none() -> None:
    assert fts_query("  -- ?? ") is None
    assert fts_query("") is None


def test_fusion_rewards_agreement_over_a_single_first_place() -> None:
    # b is second in both; a and c are each first in one and absent from the other.
    assert fuse(["a", "b"], ["c", "b"])[0] == "b"


def test_fusion_counts_an_item_once_per_ranking() -> None:
    """A file repeated across chunks must not collect a score for every repeat."""
    assert fuse(["a", "a", "a", "b"], ["b"]) == ["b", "a"]
