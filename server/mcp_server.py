"""Goodism knowledge-base MCP server (read-only).

Exposes the corpus to an LLM through the Model Context Protocol. The
server is read-only by design: there are no write tools, so the LLM can
never alter the corpus, the index, or the manifest. The LLM copies
ready-made provenance records; it never generates a TXID.

Tools:
  research(queries, context, max_hits)     PREFERRED: multi-query, merged Markdown evidence
  read_around(doc_id, chunk, before, after) neighbouring chunks of one document
  list_documents()                         what the knowledge base covers
  search_knowledge(query, top_k, domain)   hybrid search -> passages + provenance
  get_document(doc_id)                     full document text + metadata
  get_section(doc_id, heading_path)        one section's verbatim text
  list_conflicts(topic)                    confirmed contradictions + prevailing doc
  list_topics()                            topics and domains for discovery
  get_index_status()                       pinned snapshot + embedding model
  get_provenance(doc_id, section_path)     on-chain provenance record

Content is returned as clearly delimited data with fence-escape protection.
"""
from __future__ import annotations

import asyncio
import functools
import hmac
import json
import os
import re
import sys
import threading
import weakref
from pathlib import Path
from typing import Any, Callable, TypeVar

import anyio

# Allow running both as `python -m server.mcp_server` and as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.kb import config  # noqa: E402
from tools.kb.conflicts import confirmed_conflicts  # noqa: E402
from tools.kb.frontmatter import FrontMatterError, iter_corpus_files, load_document  # noqa: E402
from tools.kb.hashing import sha256_text  # noqa: E402
from tools.kb.indexer import (  # noqa: E402
    get_neighbours,
    index_status,
    list_indexed_documents,
    search,
)
from tools.kb.manifest import provenance_for  # noqa: E402
from tools.kb.retrieval import DELIM_END, DELIM_START, escape_fenced, retrieve  # noqa: E402

try:
    from mcp.server.auth.provider import AccessToken, TokenVerifier
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("mcp package is required: pip install 'mcp>=1.2'") from exc

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_MIN_TOKEN_LEN = 32
_MAX_CHUNK_ORDINAL = 1_000_000

# Every tool implementation touches blocking resources (HTTP embedding and
# reranking calls, SQLite queries, file reads). Running them directly inside an
# `async def` would stall the event loop and serialise all concurrent clients.
# They are offloaded to a worker thread instead, bounded by a per-event-loop
# semaphore so one busy client cannot monopolise the thread pool.
_DEFAULT_MAX_CONCURRENCY = 4
_MAX_CONCURRENCY_CEILING = 32

_T = TypeVar("_T")

# Keyed by the running loop: an anyio.Semaphore is bound to the event loop that
# first awaits it, and older anyio/backend combinations raise "bound to a
# different event loop" when one crosses loops (current versions happen to
# tolerate it). Each `anyio.run` in a test session is a new loop, so per-loop
# keying is the safe contract rather than a shared singleton. Weak keys also
# drop the semaphore once its loop is collected.
_semaphores: "weakref.WeakKeyDictionary[Any, anyio.Semaphore]" = weakref.WeakKeyDictionary()
_semaphores_lock = threading.Lock()


def _max_concurrency() -> int:
    """Worker-thread cap for blocking tool work, clamped to 1..32."""
    raw = os.environ.get("KB_MCP_MAX_CONCURRENCY", "")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_MAX_CONCURRENCY
    return max(1, min(value, _MAX_CONCURRENCY_CEILING))


def _loop_semaphore() -> anyio.Semaphore:
    """Return the concurrency semaphore belonging to the running event loop."""
    try:
        key: Any = asyncio.get_running_loop()
    except RuntimeError:
        # Non-asyncio backend (trio): a thread runs one loop at a time, so the
        # thread is a stable enough key and nothing here depends on asyncio.
        key = threading.current_thread()
    with _semaphores_lock:
        semaphore = _semaphores.get(key)
        if semaphore is None:
            semaphore = anyio.Semaphore(_max_concurrency())
            _semaphores[key] = semaphore
        return semaphore


async def _run_blocking(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run a blocking callable in a worker thread without stalling the loop.

    The semaphore is held until the worker actually returns, so a cancelled
    request cannot free its slot while the thread is still busy; that keeps
    ``KB_MCP_MAX_CONCURRENCY`` a real cap rather than a target.
    """
    async with _loop_semaphore():
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


class StaticTokenVerifier(TokenVerifier):
    """Constant-time bearer-token verifier for hosted HTTP/SSE transport."""

    def __init__(self, expected_token: str) -> None:
        self._expected = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or not hmac.compare_digest(token, self._expected):
            return None
        return AccessToken(token=token, client_id="kb-client", scopes=["kb:read"])


def _build_mcp(transport: str | None = None) -> FastMCP:
    mode = transport or os.environ.get("KB_MCP_TRANSPORT", "stdio")
    host = os.environ.get("KB_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("KB_MCP_PORT", "8000"))

    token_verifier: TokenVerifier | None = None
    auth_settings: AuthSettings | None = None
    transport_security: TransportSecuritySettings | None = None
    if mode != "stdio":
        token = os.environ.get("KB_MCP_AUTH_TOKEN", "")
        if len(token) < _MIN_TOKEN_LEN:
            raise RuntimeError(
                f"KB_MCP_AUTH_TOKEN (>={_MIN_TOKEN_LEN} chars) is required for HTTP/SSE transport"
            )
        token_verifier = StaticTokenVerifier(token)
        public_url = os.environ.get("KB_MCP_PUBLIC_URL", f"http://{host}:{port}")
        auth_settings = AuthSettings(
            issuer_url=public_url,
            resource_server_url=public_url,
            required_scopes=["kb:read"],
        )
        allowed_hosts = [
            h.strip()
            for h in os.environ.get("KB_MCP_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*").split(",")
            if h.strip()
        ]
        allowed_origins = [
            o.strip()
            for o in os.environ.get("KB_MCP_ALLOWED_ORIGINS", "http://127.0.0.1:*,http://localhost:*").split(",")
            if o.strip()
        ]
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )

    return FastMCP(
        "goodism-kb",
        host=host,
        port=port,
        token_verifier=token_verifier,
        auth=auth_settings,
        transport_security=transport_security,
    )


mcp = _build_mcp("stdio")


def _format_provenance(
    record: dict | None,
    *,
    served_text: str | None = None,
) -> tuple[dict | None, bool]:
    """Validate provenance shape and optionally bind it to ``served_text``.

    Returns ``(formatted_record, hash_mismatch)``. Never returns a TXID when
    ``served_text`` does not match the manifest's ``section_sha256``.
    """
    if not record or not isinstance(record, dict):
        return None, False
    txid = record.get("txid")
    vout = record.get("vout")
    if not isinstance(txid, str) or not _HEX64_RE.match(txid):
        return None, False
    if not isinstance(vout, int) or vout < 0:
        return None, False

    if served_text is not None and record.get("section_sha256"):
        if sha256_text(served_text) != record["section_sha256"]:
            return None, True

    block_hash = record.get("block_hash")
    block_height = record.get("block_height")
    has_valid_block = (
        isinstance(block_hash, str)
        and bool(_HEX64_RE.match(block_hash))
        and isinstance(block_height, int)
        and block_height > 0
    )
    proof_file = record.get("proof_file")
    has_proof = bool(proof_file) and (config.ROOT / str(proof_file)).is_file()
    return (
        {
            "doc_id": record.get("doc_id"),
            "version": record.get("version"),
            "section_path": record.get("section_path"),
            "txid": txid,
            "vout": vout,
            "block_hash": block_hash if has_valid_block else None,
            "block_height": block_height if has_valid_block else None,
            "has_block_hash": has_valid_block,
            "verified": has_valid_block and has_proof,
            "published_at": record.get("published_at"),
        },
        False,
    )


def _status_caveat(status: str, needs_review: bool = False) -> str | None:
    if status == "superseded":
        return "superseded — newer version exists"
    if needs_review:
        return f"{status} — flagged needs_review"
    if status != "published":
        return f"{status} — not yet published on-chain"
    return None


def _search_knowledge(query: str, top_k: int, domain: str | None) -> str:
    config.load_dotenv_if_present()
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"results": [], "error": "query must be non-empty"}, indent=2)
    if len(query) > 512:
        return json.dumps({"results": [], "error": "query exceeds 512-character limit"}, indent=2)
    top_k = max(1, min(int(top_k), 25))
    try:
        hits = search(query, top_k=top_k, domain=domain)
    except Exception:
        return json.dumps({"results": [], "error": "search failed"}, indent=2)
    if not hits:
        return json.dumps({"results": [], "note": "no relevant passages found"}, indent=2)

    results = []
    for hit in hits:
        prov, mismatch = _format_provenance(hit.provenance, served_text=hit.text)
        entry = {
            "doc_id": hit.doc_id,
            "title": hit.title,
            "section": hit.heading_path,
            "authority": hit.authority,
            "precedence": hit.precedence,
            "status": hit.status,
            "caveat": _status_caveat(hit.status),
            "score": round(hit.score, 4),
            "passage": f"{DELIM_START}\n{escape_fenced(hit.text)}\n{DELIM_END}",
            "provenance": prov,
        }
        if mismatch:
            entry["provenance_mismatch"] = True
        results.append(entry)
    return json.dumps({"results": results}, indent=2)


@mcp.tool()
async def search_knowledge(query: str, top_k: int = 8, domain: str | None = None) -> str:
    """Hybrid search across the knowledge base.

    Returns ranked passages with their source document, section, authority
    and, when the section has been published on-chain, its provenance
    record (TXID etc). Quote passages verbatim; never invent citations.
    """
    return await _run_blocking(_search_knowledge, query, top_k, domain)


def _research(
    queries: list[str],
    context: int,
    max_hits: int,
    domain: str | None,
) -> str:
    config.load_dotenv_if_present()
    if not isinstance(queries, list) or not queries:
        return json.dumps({"error": "queries must be a non-empty list of strings"})
    if any(not isinstance(q, str) or len(q) > 512 for q in queries):
        return json.dumps({"error": "each query must be a string of at most 512 characters"})
    try:
        return retrieve(queries, context=context, max_hits=max_hits, domain=domain)
    except Exception:
        return json.dumps({"error": "retrieval failed"})


@mcp.tool()
async def research(
    queries: list[str],
    context: int = 1,
    max_hits: int = 16,
    domain: str | None = None,
) -> str:
    """PREFERRED retrieval tool. Search several focused queries at once.

    Use this for almost every user message. Split a compound message into one
    short query per question or topic (up to 8). For each, use the user's own
    words, and add a second query with likely wording from the texts (their
    vocabulary is archaic and idiosyncratic, for example "birth certificate"
    is also "certificate of an entry of a birth"). Questions that span several
    documents should get one query per document or angle.

    Returns one Markdown evidence pack: passages grouped by relevance, each
    with its document, section, authority, which query it answers, and any
    on-chain provenance. Every query keeps its best hits, so a narrow query
    is never crowded out. `context` adds neighbouring chunks either side
    (0 to 3) because the exact fact is often just beside the best match.
    Quote verbatim, cite document and section, and never invent a citation.
    """
    return await _run_blocking(_research, queries, context, max_hits, domain)


def _read_around(doc_id: str, chunk: int, before: int, after: int) -> str:
    if not isinstance(chunk, int) or isinstance(chunk, bool) or not (0 <= chunk <= _MAX_CHUNK_ORDINAL):
        return json.dumps({"error": "chunk must be an integer between 0 and 1000000"})
    before = max(0, min(int(before), 10))
    after = max(0, min(int(after), 10))
    try:
        rows = get_neighbours(doc_id, chunk, before=before, after=after)
    except Exception:
        return json.dumps({"error": "failed to read chunks"})
    if not rows:
        return json.dumps({"error": f"no chunks near {doc_id} #{chunk}"})
    parts = [f"# {doc_id}: chunks {rows[0]['ordinal']}–{rows[-1]['ordinal']}", ""]
    for row in rows:
        safe_heading = " ".join(escape_fenced(row["heading_path"]).split())
        parts += [
            f"## [{row['ordinal']}] {safe_heading}",
            "",
            DELIM_START,
            escape_fenced(row["text"].strip()),
            DELIM_END,
            "",
        ]
    return "\n".join(parts)


@mcp.tool()
async def read_around(doc_id: str, chunk: int, before: int = 2, after: int = 2) -> str:
    """Read the chunks around a position in one document.

    Use after `research` when a passage starts or stops mid-topic. `chunk` is
    a number from the `chunks a–b` line of a result. Returns verbatim text in
    document order.
    """
    return await _run_blocking(_read_around, doc_id, chunk, before, after)


def _list_documents() -> str:
    try:
        return json.dumps({"documents": list_indexed_documents()}, indent=2)
    except Exception:
        return json.dumps({"error": "failed to list documents"})


@mcp.tool()
async def list_documents() -> str:
    """List every indexed document with its summary, topics and section count.

    Call this first if you are unsure what the knowledge base covers, or to
    choose which document to read in full with `get_document`.
    """
    return await _run_blocking(_list_documents)


def _get_document(doc_id: str) -> str:
    for path in iter_corpus_files():
        try:
            doc = load_document(path, check_vocab=False)
        except FrontMatterError:
            continue
        if doc.meta.id == doc_id:
            return json.dumps(
                {
                    "doc_id": doc.meta.id,
                    "title": doc.meta.title,
                    "domain": doc.meta.domain,
                    "topics": doc.meta.topics,
                    "authority": doc.meta.authority,
                    "precedence": doc.meta.precedence,
                    "status": doc.meta.status,
                    "needs_review": doc.meta.needs_review,
                    "caveat": _status_caveat(doc.meta.status, doc.meta.needs_review),
                    "summary": doc.meta.summary,
                    "body": f"{DELIM_START}\n{escape_fenced(doc.body)}\n{DELIM_END}",
                },
                indent=2,
            )
    return json.dumps({"error": f"no document with id '{doc_id}'"})


@mcp.tool()
async def get_document(doc_id: str) -> str:
    """Fetch a full document's verbatim text and metadata by id."""
    return await _run_blocking(_get_document, doc_id)


def _get_section(doc_id: str, heading_path: str) -> str:
    for path in iter_corpus_files():
        try:
            doc = load_document(path, check_vocab=False)
        except FrontMatterError:
            continue
        if doc.meta.id != doc_id:
            continue
        from tools.kb.chunker import chunk_document

        chunks = [
            c
            for c in chunk_document(
                doc.meta.id, doc.meta.title, doc.body, doc.meta.chunk_max_tokens
            )
            if c.heading_path == heading_path
        ]
        if not chunks:
            return json.dumps({"error": f"no section '{heading_path}' in '{doc_id}'"})

        full_text = "\n\n".join(c.text.strip() for c in chunks)
        prov, mismatch = _format_provenance(
            provenance_for(doc_id, heading_path),
            served_text=chunks[0].text if len(chunks) == 1 else full_text,
        )
        payload = {
            "doc_id": doc_id,
            "section": heading_path,
            "status": doc.meta.status,
            "needs_review": doc.meta.needs_review,
            "caveat": _status_caveat(doc.meta.status, doc.meta.needs_review),
            "chunk_count": len(chunks),
            "text": f"{DELIM_START}\n{escape_fenced(full_text)}\n{DELIM_END}",
            "provenance": prov,
        }
        if mismatch:
            payload["provenance_mismatch"] = True
        return json.dumps(payload, indent=2)
    return json.dumps({"error": f"no document with id '{doc_id}'"})


@mcp.tool()
async def get_section(doc_id: str, heading_path: str) -> str:
    """Fetch one section's verbatim text plus its hash-verified provenance record."""
    return await _run_blocking(_get_section, doc_id, heading_path)


def _list_conflicts(topic: str | None) -> str:
    conflicts = confirmed_conflicts(topic)
    return json.dumps({"conflicts": conflicts, "count": len(conflicts)}, indent=2)


@mcp.tool()
async def list_conflicts(topic: str | None = None) -> str:
    """List confirmed contradictions and which document prevails."""
    return await _run_blocking(_list_conflicts, topic)


def _list_topics() -> str:
    import yaml

    domains = yaml.safe_load((config.VOCAB_DIR / "domains.yaml").read_text("utf-8"))["domains"]
    topics = yaml.safe_load((config.VOCAB_DIR / "topics.yaml").read_text("utf-8"))["topics"]
    return json.dumps(
        {
            "domains": {k: v["description"] for k, v in domains.items()},
            "topics": topics,
        },
        indent=2,
    )


@mcp.tool()
async def list_topics() -> str:
    """List domains and topics for discovery and query filtering."""
    return await _run_blocking(_list_topics)


def _get_index_status() -> str:
    try:
        return json.dumps(index_status(), indent=2)
    except Exception:
        return json.dumps({"error": "index unavailable"})


@mcp.tool()
async def get_index_status() -> str:
    """Return the pinned snapshot: version, embedding model, counts.

    Callers must check this before trusting results — if the configured
    embedding model differs from the pinned snapshot's model, a rebuild is
    pending and results may mix representations.
    """
    return await _run_blocking(_get_index_status)


def _get_provenance(doc_id: str, section_path: str) -> str:
    record = provenance_for(doc_id, section_path)
    prov, _ = _format_provenance(record)
    if prov is None:
        return json.dumps({"error": f"no valid provenance record for {doc_id} :: {section_path}"})
    return json.dumps(prov, indent=2)


@mcp.tool()
async def get_provenance(doc_id: str, section_path: str) -> str:
    """Return the on-chain provenance record for a published section.

    This is the record the bot copies verbatim when citing a source. The
    `verified` flag is true only once the transaction is confirmed in a
    block (valid 64-hex block_hash, positive height) AND a proof file exists.
    """
    return await _run_blocking(_get_provenance, doc_id, section_path)


def main() -> None:
    transport = os.environ.get("KB_MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Hosted mode: enforce bearer auth and DNS-rebinding protection in-process,
        # in addition to TLS termination at the reverse proxy / gateway.
        hosted = _build_mcp(transport)
        assert hosted._token_verifier is not None
        for tool in mcp._tool_manager.list_tools():
            hosted._tool_manager._tools[tool.name] = tool
        hosted.run(transport="sse")


if __name__ == "__main__":
    main()
