"""Append-only on-chain provenance manifest.

The manifest (manifest/manifest.jsonl) is the one place that maps content
to proof. Published corpus files stay byte-identical to what is (or will
be) on-chain; the TXID and SPV proof data live only here. Records are
only ever appended — corrections add a new record with supersedes_txid.

One record per published section (a chunk heading-path). At publication
time a section's plaintext goes into a BSV transaction output; this
module records that mapping. The actual broadcast is deliberately out of
scope for the knowledge-base tooling (it happens once organisation is
complete, on testnet first); `kb manifest prepare` produces the exact
byte payloads to publish, and `kb manifest record` appends the resulting
TXIDs after broadcast.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .chunker import chunk_document
from .frontmatter import iter_corpus_files, load_document
from .hashing import canonical_bytes, sha256_text
import hashlib


@dataclass
class PreparedSection:
    doc_id: str
    version: int
    section_path: str
    file_path: str
    section_sha256: str
    file_sha256: str
    payload_hex: str  # exact canonical bytes to embed on-chain
    supersedes_txid: str | None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest() -> list[dict]:
    if not config.MANIFEST_PATH.exists():
        return []
    records = []
    for line in config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _latest_records_by_section() -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    for record in read_manifest():
        latest[(record["doc_id"], record["section_path"])] = record
    return latest


def prepare_publication(doc_id: str | None = None) -> list[PreparedSection]:
    """Build the exact payloads for sections of reviewed/published docs.

    A section is prepared when its document status is "reviewed" or
    "published" and its canonical bytes differ from the latest manifest
    record (new document or a correction).
    """
    latest = _latest_records_by_section()
    prepared: list[PreparedSection] = []

    for path in iter_corpus_files():
        doc = load_document(path, check_vocab=False)
        if doc.meta.status not in {"reviewed", "published"}:
            continue
        if doc_id and doc.meta.id != doc_id:
            continue

        chunks = chunk_document(doc.meta.id, doc.meta.title, doc.body)
        file_sha = _file_sha256(path)
        for chunk in chunks:
            section_bytes = canonical_bytes(chunk.text)
            section_sha = hashlib.sha256(section_bytes).hexdigest()
            key = (doc.meta.id, chunk.heading_path)
            previous = latest.get(key)
            if previous and previous["section_sha256"] == section_sha:
                continue  # unchanged since publication
            version = (previous["version"] + 1) if previous else 1
            prepared.append(
                PreparedSection(
                    doc_id=doc.meta.id,
                    version=version,
                    section_path=chunk.heading_path,
                    file_path=doc.rel_path,
                    section_sha256=section_sha,
                    file_sha256=file_sha,
                    payload_hex=section_bytes.hex(),
                    supersedes_txid=previous["txid"] if previous else None,
                )
            )
    return prepared


def record_publication(
    *,
    doc_id: str,
    version: int,
    section_path: str,
    file_path: str,
    section_sha256: str,
    file_sha256: str,
    txid: str,
    vout: int,
    block_hash: str | None = None,
    block_height: int | None = None,
    proof_file: str | None = None,
    supersedes_txid: str | None = None,
) -> dict:
    """Append one record after broadcast. Never edits existing lines."""
    record = {
        "doc_id": doc_id,
        "version": version,
        "section_path": section_path,
        "file_path": file_path,
        "section_sha256": section_sha256,
        "file_sha256": file_sha256,
        "txid": txid,
        "vout": vout,
        "block_hash": block_hash,
        "block_height": block_height,
        "proof_file": proof_file,
        "supersedes_txid": supersedes_txid,
        "status": "published",
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def anchor_manifest(txid: str, *, block_hash: str | None = None, block_height: int | None = None) -> dict:
    """Record an on-chain anchor of the manifest's own hash.

    Stored in manifest.anchor.json (separate file) to avoid the circularity
    of a hash appearing inside the data it hashes.
    """
    manifest_sha = hashlib.sha256(config.MANIFEST_PATH.read_bytes()).hexdigest()
    anchor = {
        "manifest_sha256": manifest_sha,
        "txid": txid,
        "block_hash": block_hash,
        "block_height": block_height,
        "anchored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    history = []
    if config.MANIFEST_ANCHOR_PATH.exists():
        history = json.loads(config.MANIFEST_ANCHOR_PATH.read_text("utf-8")).get("anchors", [])
    history.append(anchor)
    config.MANIFEST_ANCHOR_PATH.write_text(
        json.dumps({"anchors": history}, indent=2), encoding="utf-8"
    )
    return anchor


def provenance_for(doc_id: str, section_path: str) -> dict | None:
    """Latest record for one section (used by the MCP server)."""
    return _latest_records_by_section().get((doc_id, section_path))
