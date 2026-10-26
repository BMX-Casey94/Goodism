"""Conflict resolution tests: precedence then authority decide the winner."""
from __future__ import annotations

# The resolution rule: lower precedence number wins; ties broken by
# authority rank canonical > supporting > contextual > superseded.

AUTHORITY_RANK = {"canonical": 0, "supporting": 1, "contextual": 2, "superseded": 3}


def _winner(a_precedence, a_authority, b_precedence, b_authority):
    a_key = (a_precedence, AUTHORITY_RANK[a_authority])
    b_key = (b_precedence, AUTHORITY_RANK[b_authority])
    return "a" if a_key < b_key else "b"


def test_lower_precedence_wins():
    assert _winner(10, "supporting", 50, "supporting") == "a"
    assert _winner(50, "supporting", 10, "supporting") == "b"


def test_authority_breaks_precedence_tie():
    assert _winner(50, "canonical", 50, "contextual") == "a"
    assert _winner(50, "contextual", 50, "supporting") == "b"


def test_precedence_beats_authority():
    # A lower precedence number wins even against higher authority.
    assert _winner(10, "contextual", 50, "canonical") == "a"


def test_superseded_never_wins_on_tie():
    assert _winner(50, "superseded", 50, "supporting") == "b"
