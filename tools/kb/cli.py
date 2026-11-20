"""Goodism KB command line interface.

Usage (after `pip install -e .`):
  kb ingest [--file NAME]        Ingest raw/inbox (or one file) into the corpus
  kb validate                    CI gate: schema, hashes, duplicates, links, manifest
  kb build-index [--full]        Build/incrementally update the hybrid index
  kb search "query"              Hybrid search against the pinned snapshot
  kb conflicts detect|list|confirm|reject
  kb manifest prepare            Emit payloads to publish on-chain (per section)
  kb manifest record ...         Append a TXID record after broadcast
  kb status                      Index snapshot pin + counts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import config

app = typer.Typer(add_completion=False, no_args_is_help=True)
conflicts_app = typer.Typer(no_args_is_help=True)
manifest_app = typer.Typer(no_args_is_help=True)
app.add_typer(conflicts_app, name="conflicts")
app.add_typer(manifest_app, name="manifest")


def _client_or_none(use_llm: bool):
    if not use_llm:
        return None
    from .llm import LLMClient

    config.load_dotenv_if_present()
    return LLMClient()


@app.command()
def ingest(
    file: Optional[str] = typer.Option(None, help="Single file name in raw/inbox"),
    no_llm: bool = typer.Option(False, help="Convert only; skip subagent steps"),
) -> None:
    """Ingest raw files into the corpus (staging anything that needs review)."""
    from .ingest import ingest_file

    config.ensure_dirs()
    client = _client_or_none(not no_llm)
    if file is not None:
        if Path(file).is_absolute() or ".." in Path(file).parts or "/" in file or "\\" in file:
            typer.echo("error: --file must be a plain filename inside raw/inbox", err=True)
            raise typer.Exit(2)
        target = (config.RAW_INBOX / file).resolve()
        if not target.is_relative_to(config.RAW_INBOX.resolve()):
            typer.echo("error: --file escapes raw/inbox", err=True)
            raise typer.Exit(2)
        targets = [target]
    else:
        targets = [p for p in sorted(config.RAW_INBOX.iterdir()) if p.is_file()]
    if not targets:
        typer.echo("raw/inbox is empty")
        raise typer.Exit(0)
    for path in targets:
        result = ingest_file(path, client)
        where = "staged" if result.staged else "corpus"
        typer.echo(
            f"{path.name} -> {result.corpus_path.relative_to(config.ROOT)} [{where}] "
            f"edits applied={result.edits_applied} rejected={result.edits_rejected}"
        )
        for flag in result.review_flags:
            typer.echo(f"  REVIEW: {flag}")


@app.command()
def validate() -> None:
    """Run the CI validation gate."""
    from .validate import run_validation

    report = run_validation()
    for warning in report.warnings:
        typer.echo(f"WARN  {warning}")
    for error in report.errors:
        typer.echo(f"ERROR {error}")
    typer.echo(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    raise typer.Exit(0 if report.ok else 1)


@app.command(name="build-index")
def build_index(
    full: bool = typer.Option(False, help="Rebuild every document from scratch"),
    note: Optional[str] = typer.Option(None),
) -> None:
    """Build or incrementally update the hybrid search index."""
    from .indexer import build_index as _build

    config.load_dotenv_if_present()
    report = _build(full=full, note=note)
    typer.echo(
        f"snapshot {report.snapshot_id}: {report.documents} docs, {report.chunks} chunks "
        f"(rebuilt {len(report.rebuilt_docs)}, carried {len(report.skipped_unchanged)}, "
        f"full={report.full_rebuild})"
    )


@app.command()
def search(
    query: str,
    top_k: int = typer.Option(config.DEFAULT_TOP_K),
    domain: Optional[str] = typer.Option(None),
    no_rerank: bool = typer.Option(False),
) -> None:
    """Hybrid search over the pinned snapshot."""
    from .indexer import search as _search

    config.load_dotenv_if_present()
    hits = _search(query, top_k=top_k, domain=domain, rerank=not no_rerank)
    for hit in hits:
        typer.echo(f"[{hit.score:.4f}] {hit.doc_id} :: {hit.heading_path} ({hit.source})")
        typer.echo(f"    {hit.text[:200].replace(chr(10), ' ')}")
        if hit.provenance:
            typer.echo(f"    txid: {hit.provenance['txid']} (v{hit.provenance['version']})")


@app.command(name="check-llm")
def check_llm() -> None:
    """Verify the LLM config (text call and vision call) before ingesting scans."""
    import base64
    import os
    import struct
    import zlib

    from .llm import LLMClient, LLMError

    config.load_dotenv_if_present()
    missing = [
        name
        for name in ("KB_LLM_BASE_URL", "KB_LLM_API_KEY", "KB_MODEL_BULK")
        if not os.environ.get(name)
    ]
    if missing:
        typer.echo(f"missing in .env: {', '.join(missing)} (see .env.example)", err=True)
        raise typer.Exit(1)

    client = LLMClient()
    typer.echo(f"endpoint: {client.base_url}")
    typer.echo(f"bulk model: {client.bulk_model} | strong model: {client.strong_model}")

    try:
        reply = client.chat(
            role="structurer", system="Reply with the single word OK.", user="ping", max_tokens=8
        )
        typer.echo(f"text call: ok ({reply.strip()[:20]!r})")
    except LLMError as exc:
        typer.echo(f"text call FAILED: {exc}", err=True)
        raise typer.Exit(1)

    # Vision call with a tiny generated 8x8 white PNG (no external files).
    def _png(width: int = 8, height: int = 8) -> bytes:
        raw = b"".join(b"\x00" + b"\xff" * (width * 3) for _ in range(height))

        def chunk(tag: bytes, data: bytes) -> bytes:
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )

    for role in ("extractor", "verifier"):
        try:
            client.chat(
                role=role,
                system="Describe the image in three words.",
                user="What is this?",
                images=[_png()],
                max_tokens=16,
            )
            typer.echo(f"vision call ({role}, {client.model_for(role)}): ok")
        except LLMError as exc:
            typer.echo(f"vision call ({role}) FAILED: {exc}", err=True)
            raise typer.Exit(1)
    typer.echo("LLM config OK: ready for `kb ingest` on scans")


@app.command()
def status() -> None:
    """Show the pinned index snapshot."""
    from .indexer import index_status

    typer.echo(json.dumps(index_status(), indent=2))


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------


@conflicts_app.command(name="detect")
def conflicts_detect() -> None:
    """Run the conflict subagent across topic clusters."""
    from .conflicts import detect_conflicts

    client = _client_or_none(True)
    added = detect_conflicts(client)
    typer.echo(f"{added} new conflict candidate(s) recorded for review")


@conflicts_app.command(name="list")
def conflicts_list() -> None:
    """List pending conflict candidates."""
    from .conflicts import list_candidates

    candidates = list_candidates()
    if not candidates:
        typer.echo("no pending candidates")
        return
    for item in candidates:
        typer.echo(f"{item['doc_id']} <-> {item['other_doc_id']}: {item['note']}")


@conflicts_app.command(name="confirm")
def conflicts_confirm(doc_id: str, other_id: str) -> None:
    """Confirm a candidate: promotes it to front matter conflicts_with."""
    from .conflicts import review_candidate

    if review_candidate(doc_id, other_id, confirm=True):
        typer.echo(f"confirmed: {doc_id} <-> {other_id}")
    else:
        typer.echo("candidate not found", err=True)
        raise typer.Exit(1)


@conflicts_app.command(name="reject")
def conflicts_reject(doc_id: str, other_id: str) -> None:
    """Reject a candidate."""
    from .conflicts import review_candidate

    if review_candidate(doc_id, other_id, confirm=False):
        typer.echo(f"rejected: {doc_id} <-> {other_id}")
    else:
        typer.echo("candidate not found", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@manifest_app.command(name="prepare")
def manifest_prepare(
    doc_id: Optional[str] = typer.Option(None),
    out: Optional[Path] = typer.Option(None, help="Write payloads JSON here"),
) -> None:
    """Prepare the exact byte payloads to publish on-chain."""
    from .manifest import prepare_publication

    prepared = prepare_publication(doc_id)
    payload = [p.__dict__ for p in prepared]
    if out:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"{len(payload)} section payload(s) written to {out}")
    else:
        typer.echo(json.dumps(payload, indent=2))


@manifest_app.command(name="record")
def manifest_record(
    doc_id: str,
    version: int,
    section_path: str,
    file_path: str,
    section_sha256: str,
    file_sha256: str,
    txid: str,
    vout: int = 0,
    block_hash: Optional[str] = None,
    block_height: Optional[int] = None,
    supersedes_txid: Optional[str] = None,
) -> None:
    """Append a manifest record after a section is broadcast on-chain."""
    from .manifest import record_publication

    record = record_publication(
        doc_id=doc_id,
        version=version,
        section_path=section_path,
        file_path=file_path,
        section_sha256=section_sha256,
        file_sha256=file_sha256,
        txid=txid,
        vout=vout,
        block_hash=block_hash,
        block_height=block_height,
        supersedes_txid=supersedes_txid,
    )
    typer.echo(f"recorded {record['doc_id']} v{record['version']} txid={txid}")


@manifest_app.command(name="anchor")
def manifest_anchor(
    txid: str,
    block_hash: Optional[str] = None,
    block_height: Optional[int] = None,
) -> None:
    """Record an on-chain anchor of the manifest's own hash."""
    from .manifest import anchor_manifest

    anchor = anchor_manifest(txid, block_hash=block_hash, block_height=block_height)
    typer.echo(f"manifest anchored at {anchor['anchored_at']} txid={txid}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
