"""Canonicalisation and hashing tests."""
from __future__ import annotations

from tools.kb.hashing import canonical_bytes, canonicalise, sha256_text


def test_crlf_normalised_to_lf():
    assert canonicalise("a\r\nb\r\nc") == "a\nb\nc\n"


def test_nfc_normalisation():
    # e + combining acute -> precomposed é
    assert canonicalise("cafe\u0301") == canonicalise("café")


def test_trailing_newline_single():
    assert canonicalise("text\n\n\n") == "text\n"
    assert canonicalise("text") == "text\n"


def test_hash_stable_across_line_endings():
    assert sha256_text("a\r\nb") == sha256_text("a\nb")


def test_canonical_bytes_utf8_no_bom():
    assert canonical_bytes("héllo") == "héllo\n".encode("utf-8")
