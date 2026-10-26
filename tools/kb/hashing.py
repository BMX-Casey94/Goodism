"""Canonicalisation and hashing.

The same canonical form is used everywhere: integrity diffs, index rebuild
detection, manifest records, and (eventually) on-chain publication. If the
bytes on-chain must match the bytes here, there must be exactly one
canonicalisation function — this is it.

Canonical form: Unicode NFC, CRLF/CR -> LF, UTF-8, no BOM, with a single
trailing newline at end of file. Trailing-newline normalisation keeps
editors that add/remove a final newline from breaking hash equality with
published bytes; the on-chain publication step must apply the same rule.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from . import config


def canonicalise(text: str) -> str:
    """Return the canonical string form (NFC, LF line endings, one trailing \\n)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip("\n") + "\n"


def canonical_bytes(text: str) -> bytes:
    """Canonical string encoded as UTF-8 (no BOM)."""
    return canonicalise(text).encode(config.CANONICAL_ENCODING)


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of the canonical bytes of *text*."""
    return hashlib.sha256(canonical_bytes(text)).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's raw bytes (no canonicalisation).

    Used for provenance back to raw/ originals, where the untouched bytes
    are what matter.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
