"""SQLite hybrid index: FTS5 (BM25) + dense vectors + rerank, with snapshots.

Design:
- Single-file database at index/kb.sqlite (git-ignored, rebuildable).
- Keyword leg: FTS5 with BM25 ranking over the display text (contextual
  prefix + verbatim chunk).
- Vector leg: embeddings via the configured provider, stored as float32
  blobs, exact cosine search in Python (fine at this scale; the schema is
  portable to Postgres/pgvector later).
- Fusion: reciprocal rank fusion, then an optional cross-encoder rerank
  of the top candidates.
- Snapshots: every build is versioned in the `snapshots` table; queries
  always run against the pinned snapshot so answers are reproducible.
- Incremental: only documents whose canonical hash changed are re-chunked
  and re-embedded. Changing EMBEDDING_MODEL forces a full rebuild.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config, lexicon
from .chunker import Chunk, chunk_document
from .frontmatter import CorpusDocument, iter_corpus_files, load_document
from .hashing import sha256_text

SCHEMA_VERSION = 1


_runtime_index_lock = threading.Lock()


def _connect(db_path: Path | None = None, *, readonly: bool = False) -> sqlite3.Connection:
    path = _resolve_index_path(db_path)
    if readonly:
        # Read-only so a query never creates a journal. Vercel's runtime
        # filesystem is read-only outside /tmp.
        if not path.is_file():
            raise RuntimeError("index is empty; run `kb build-index` first")
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        return conn
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # One file, so a build artifact does not depend on a sibling -wal file.
    if os.environ.get("VERCEL") == "1" or os.environ.get("KB_INDEX_PATH"):
        conn.execute("PRAGMA journal_mode=DELETE")
    else:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _resolve_index_path(db_path: Path | None) -> Path:
    if db_path is not None:
        return db_path.resolve()
    path = config.index_db_path().resolve()
    if path.is_file() or os.environ.get("VERCEL") != "1":
        return path
    return _build_vercel_index(path)


def _build_vercel_index(preferred: Path) -> Path:
    """Build a keyword index into /tmp when the bundled file was not shipped."""
    with _runtime_index_lock:
        if preferred.is_file():
            return preferred
        target = Path("/tmp/goodism-kb.sqlite")
        if not target.is_file():
            build_index(
                full=True,
                db_path=target,
                embedder=Embedder("fts-only"),
                note="vercel-runtime",
            )
        return target


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            doc_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            note TEXT
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT NOT NULL,
            snapshot_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            domain TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            authority TEXT NOT NULL,
            precedence INTEGER NOT NULL,
            status TEXT NOT NULL,
            body_sha256 TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            PRIMARY KEY (doc_id, snapshot_id)
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            doc_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            parent_ordinal INTEGER,
            heading_path TEXT NOT NULL,
            level INTEGER NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            text TEXT NOT NULL,
            display TEXT NOT NULL,
            embedding BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (snapshot_id, doc_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            display,
            content='',
            tokenize='unicode61'
        );
        """
    )


def _pack_embedding(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_embedding(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class Embedder:
    """Embedding provider wrapper. Model is pinned per snapshot.

    Set KB_EMBEDDING_MODEL=local:<name> to use a local sentence-transformers
    model instead of an API (keeps private content off third parties).
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.EMBEDDING_MODEL
        self._client = None
        self._local = None
        if self.model.startswith("local:"):
            from .embeddings_local import LocalEmbedder

            self._local = LocalEmbedder(self.model.removeprefix("local:"))

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package is required for embeddings") from exc
            self._client = OpenAI(
                base_url=os.environ.get("KB_EMBEDDING_BASE_URL") or None,
                api_key=os.environ.get("KB_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                timeout=60.0,
                max_retries=2,
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.model == "fts-only":
            return [[] for _ in texts]
        if self._local is not None:
            return self._local.embed(texts)
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


@lru_cache(maxsize=2)
def _load_manifest_cached(path: str, mtime: float, size: int) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or "doc_id" not in record or "section_path" not in record:
            continue
        key = (record["doc_id"], record["section_path"])
        prev = records.get(key)
        if prev is None or (record.get("version", 0), record.get("published_at", "")) >= (
            prev.get("version", 0),
            prev.get("published_at", ""),
        ):
            records[key] = record
    return records


def _load_manifest_records() -> dict[tuple[str, str], dict]:
    """Map (doc_id, section_path-normalised heading path) -> latest manifest record."""
    if not config.MANIFEST_PATH.exists():
        return {}
    try:
        st = config.MANIFEST_PATH.stat()
        return _load_manifest_cached(str(config.MANIFEST_PATH), st.st_mtime, st.st_size)
    except OSError:
        return {}


@dataclass
class BuildReport:
    snapshot_id: int
    documents: int
    chunks: int
    rebuilt_docs: list[str]
    skipped_unchanged: list[str]
    full_rebuild: bool


def _chunk_params() -> str:
    """Fingerprint of everything that decides how documents are cut into chunks."""
    overrides = []
    for path in iter_corpus_files():
        meta = load_document(path, check_vocab=False).meta
        if meta.chunk_max_tokens:
            overrides.append(f"{meta.id}={meta.chunk_max_tokens}")
    return (
        f"max_tokens={config.CHUNK_MAX_TOKENS};keywords={int(config.INDEX_KEYWORDS)};"
        f"overrides={','.join(sorted(overrides))}"
    )


def build_index(
    *,
    full: bool = False,
    note: str | None = None,
    db_path: Path | None = None,
    embedder: Embedder | None = None,
) -> BuildReport:
    """Build (or incrementally update) the index from the corpus."""
    config.ensure_dirs()
    conn = _connect(db_path)
    try:
        _create_schema(conn)
        embedder = embedder or get_embedder(config.EMBEDDING_MODEL)

        # Full rebuild triggers: explicit flag, or embedding model change.
        row = conn.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
        previous_model = row[0] if row else None
        full = full or (previous_model is not None and previous_model != embedder.model)
        # Chunking settings change what a chunk is, so unchanged bodies must be re-chunked.
        row = conn.execute("SELECT value FROM meta WHERE key='chunk_params'").fetchone()
        full = full or (previous_model is not None and (row[0] if row else None) != _chunk_params())

        if full:
            # chunks_fts is contentless, so rows cannot be deleted; it is
            # dropped here and recreated when the FTS index is rebuilt below.
            conn.execute("DROP TABLE IF EXISTS chunks_fts")
            for table in ("chunks", "documents", "snapshots"):
                conn.execute(f"DELETE FROM {table}")

        previous_hashes: dict[str, str] = {}
        if not full:
            for doc_id, body_sha in conn.execute(
                "SELECT doc_id, body_sha256 FROM documents WHERE snapshot_id = (SELECT MAX(id) FROM snapshots)"
            ):
                previous_hashes[doc_id] = body_sha

        cursor = conn.execute(
            "INSERT INTO snapshots (created_at, embedding_model, doc_count, chunk_count, note)"
            " VALUES (?, ?, 0, 0, ?)",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), embedder.model, note),
        )
        snapshot_id = cursor.lastrowid

        rebuilt: list[str] = []
        skipped: list[str] = []
        total_chunks = 0
        total_docs = 0

        manifest_records = _load_manifest_records()

        for path in iter_corpus_files():
            try:
                doc = load_document(path, check_vocab=True)
            except Exception as exc:
                raise RuntimeError(f"corpus validation failed for {path}: {exc}") from exc

            body_sha = sha256_text(doc.body)
            total_docs += 1

            if not full and previous_hashes.get(doc.meta.id) == body_sha:
                # Carry chunks forward unchanged into the new snapshot.
                prev_snapshot = conn.execute(
                    "SELECT MAX(id) FROM snapshots WHERE id < ?", (snapshot_id,)
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT doc_id, ordinal, parent_ordinal, heading_path, level,"
                    " start_offset, end_offset, text, display, embedding"
                    " FROM chunks WHERE snapshot_id = ? AND doc_id = ?",
                    (prev_snapshot, doc.meta.id),
                ).fetchall()
                for row in rows:
                    # row has 10 columns: doc_id..embedding. Prepend snapshot_id.
                    conn.execute(
                        "INSERT INTO chunks (snapshot_id, doc_id, ordinal, parent_ordinal,"
                        " heading_path, level, start_offset, end_offset, text, display, embedding)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (snapshot_id, *row),
                    )
                total_chunks += len(rows)
                conn.execute(
                    "INSERT INTO documents (doc_id, snapshot_id, path, title, domain, doc_type,"
                    " authority, precedence, status, body_sha256, meta_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc.meta.id,
                        snapshot_id,
                        doc.rel_path,
                        doc.meta.title,
                        doc.meta.domain,
                        doc.meta.doc_type,
                        doc.meta.authority,
                        doc.meta.precedence,
                        doc.meta.status,
                        body_sha,
                        json.dumps(doc.meta.model_dump()),
                    ),
                )
                skipped.append(doc.meta.id)
                continue

            chunks = chunk_document(
                doc.meta.id, doc.meta.title, doc.body, doc.meta.chunk_max_tokens
            )
            embeddings = embedder.embed([c.display for c in chunks]) if chunks else []
            for chunk, embedding in zip(chunks, embeddings):
                conn.execute(
                    "INSERT INTO chunks (snapshot_id, doc_id, ordinal, parent_ordinal,"
                    " heading_path, level, start_offset, end_offset, text, display, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot_id,
                        doc.meta.id,
                        chunk.ordinal,
                        chunk.parent_ordinal,
                        chunk.heading_path,
                        chunk.level,
                        chunk.start,
                        chunk.end,
                        chunk.text,
                        chunk.display,
                        _pack_embedding(embedding) if embedding else None,
                    ),
                )
            total_chunks += len(chunks)
            conn.execute(
                "INSERT INTO documents (doc_id, snapshot_id, path, title, domain, doc_type,"
                " authority, precedence, status, body_sha256, meta_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.meta.id,
                    snapshot_id,
                    doc.rel_path,
                    doc.meta.title,
                    doc.meta.domain,
                    doc.meta.doc_type,
                    doc.meta.authority,
                    doc.meta.precedence,
                    doc.meta.status,
                    body_sha,
                    json.dumps(doc.meta.model_dump()),
                ),
            )
            rebuilt.append(doc.meta.id)

        # Rebuild FTS for the new snapshot rows. A contentless FTS5 table
        # cannot be selectively deleted, so drop and recreate it fresh for
        # the pinned snapshot (search always targets the pinned snapshot).
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(display, content='', tokenize='unicode61')"
        )
        conn.execute(
            "INSERT INTO chunks_fts (rowid, display)"
            " SELECT id, display FROM chunks WHERE snapshot_id = ?",
            (snapshot_id,),
        )

        conn.execute(
            "UPDATE snapshots SET doc_count = ?, chunk_count = ? WHERE id = ?",
            (total_docs, total_chunks, snapshot_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_model', ?)",
            (embedder.model,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('chunk_params', ?)",
            (_chunk_params(),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('pinned_snapshot', ?)",
            (str(snapshot_id),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        return BuildReport(
            snapshot_id=snapshot_id,
            documents=total_docs,
            chunks=total_chunks,
            rebuilt_docs=rebuilt,
            skipped_unchanged=skipped,
            full_rebuild=full,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    doc_id: str
    title: str
    heading_path: str
    text: str
    score: float
    authority: str
    precedence: int
    status: str
    source: str  # "keyword" | "vector" | "both"
    provenance: dict | None  # manifest record when the section is published
    ordinal: int = -1  # position of the chunk within its document


@lru_cache(maxsize=4)
def get_embedder(model: str) -> "Embedder":
    """One Embedder per model name for the life of the process.

    Loading a sentence-transformers model takes seconds; doing it per query
    would make every MCP search slow.
    """
    return Embedder(model)


@lru_cache(maxsize=1)
def get_reranker() -> "Reranker":
    return Reranker()


class Reranker:
    """Optional cross-encoder reranker over fused candidates."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model
        self._model = None

    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def rerank(self, query: str, hits: list[SearchHit], keep: int) -> list[SearchHit]:
        if not self.available() or not hits:
            return hits[:keep]
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:keep]


MAX_FTS_QUERY_CHARS = 512
MAX_FTS_TERMS = 32


def _fts_escape(query: str, extra_phrases: list[str] | None = None) -> str:
    """Build a safe, bounded FTS5 MATCH expression from a free-text query.

    Splits into terms and double-quotes each one, joining with OR, so user
    punctuation (?, ", *, etc.) can never be parsed as FTS5 operators.
    Input and term count are capped to prevent superlinear FTS5 CPU DoS.
    ``extra_phrases`` (from the lexicon) are added as exact quoted phrases.
    """
    terms = re.findall(r"[\w'-]+", (query or "")[:MAX_FTS_QUERY_CHARS])[:MAX_FTS_TERMS]
    parts = [f'"{t[:64]}"' for t in terms]
    for phrase in extra_phrases or []:
        phrase = " ".join(phrase.split())[:120]
        if phrase:
            parts.append('"' + phrase.replace('"', '""') + '"')
    if not parts:
        return '""'
    return " OR ".join(parts)


def _pinned_snapshot(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='pinned_snapshot'").fetchone()
    if row:
        return int(row[0])
    row = conn.execute("SELECT MAX(id) FROM snapshots").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("index is empty; run `kb build-index` first")
    return int(row[0])


def search(
    query: str,
    *,
    top_k: int = config.DEFAULT_TOP_K,
    domain: str | None = None,
    include_superseded: bool = False,
    rerank: bool = True,
    db_path: Path | None = None,
    embedder: Embedder | None = None,
) -> list[SearchHit]:
    """Hybrid search over the pinned snapshot with RRF fusion + rerank."""
    conn = _connect(db_path, readonly=True)
    try:
        snapshot_id = _pinned_snapshot(conn)
        if embedder is None:
            # Query vectors must come from the model that built the snapshot,
            # whatever the environment says; mixing models gives meaningless scores.
            row = conn.execute(
                "SELECT embedding_model FROM snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
            embedder = get_embedder(row[0] if row and row[0] else config.EMBEDDING_MODEL)

        doc_filter = ""
        params: list = [snapshot_id]
        if domain:
            doc_filter += " AND d.domain = ?"
            params.append(domain)
        if not include_superseded:
            doc_filter += " AND d.status != 'superseded' AND d.authority != 'superseded'"

        # Keyword leg (BM25; lower is better in FTS5, so negate).
        match_expr = _fts_escape(query, lexicon.expansions(query))
        keyword_rows = conn.execute(
            f"""
            SELECT c.id, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.doc_id = c.doc_id AND d.snapshot_id = c.snapshot_id
            WHERE chunks_fts MATCH ? AND c.snapshot_id = ? {doc_filter}
            ORDER BY rank
            LIMIT ?
            """,
            (match_expr, *params, config.SEARCH_CANDIDATES_PER_LEG),
        ).fetchall()
        keyword_rank = {row[0]: i + 1 for i, row in enumerate(keyword_rows)}

        # Vector leg (exact cosine over the snapshot's chunks).
        vector_rank: dict[int, int] = {}
        if embedder.model != "fts-only":
            query_vec = embedder.embed([query])[0]
            rows = conn.execute(
                f"""
                SELECT c.id, c.embedding
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id AND d.snapshot_id = c.snapshot_id
                WHERE c.snapshot_id = ? {doc_filter}
                """,
                params,
            ).fetchall()
            scored = [
                (row[0], _cosine(query_vec, _unpack_embedding(row[1])))
                for row in rows
                if row[1] is not None
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            for i, (chunk_id, _) in enumerate(scored[: config.SEARCH_CANDIDATES_PER_LEG]):
                vector_rank[chunk_id] = i + 1

        # Reciprocal rank fusion.
        candidate_ids = set(keyword_rank) | set(vector_rank)
        fused: list[tuple[int, float, str]] = []
        for chunk_id in candidate_ids:
            score = 0.0
            source = []
            if chunk_id in keyword_rank:
                score += 1.0 / (config.RRF_K + keyword_rank[chunk_id])
                source.append("keyword")
            if chunk_id in vector_rank:
                score += 1.0 / (config.RRF_K + vector_rank[chunk_id])
                source.append("vector")
            fused.append((chunk_id, score, "+".join(source)))
        fused.sort(key=lambda item: item[1], reverse=True)
        fused = fused[: config.RERANK_KEEP]

        if not fused:
            return []

        ids = [int(cid) for cid, _, _ in fused]
        placeholders = ",".join("?" * len(ids))
        detail_rows = conn.execute(
            f"""
            SELECT c.id, c.doc_id, d.title, c.heading_path, c.text,
                   d.authority, d.precedence, d.status, c.ordinal
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id AND d.snapshot_id = c.snapshot_id
            WHERE c.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        details = {row[0]: row for row in detail_rows}

        manifest_records = _load_manifest_records()

        hits: list[SearchHit] = []
        for chunk_id, score, source in fused:
            row = details.get(chunk_id)
            if row is None:
                continue
            _, doc_id, title, heading_path, text, authority, precedence, status, ordinal = row
            provenance = manifest_records.get((doc_id, heading_path))
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    title=title,
                    heading_path=heading_path,
                    text=text,
                    score=score,
                    authority=authority,
                    precedence=precedence,
                    status=status,
                    source=source,
                    provenance=provenance,
                    ordinal=ordinal,
                )
            )

        # Authority precedence as a tie-break nudge before rerank.
        hits.sort(key=lambda h: (h.score, -h.precedence), reverse=True)

        if rerank:
            reranker = get_reranker()
            if reranker.available():
                hits = reranker.rerank(query, hits, keep=top_k)
                return hits
        return hits[:top_k]
    finally:
        conn.close()


def get_neighbours(
    doc_id: str,
    ordinal: int,
    before: int = 1,
    after: int = 1,
    db_path: Path | None = None,
) -> list[dict]:
    """Chunks around ``ordinal`` in the pinned snapshot, in document order."""
    conn = _connect(db_path, readonly=True)
    try:
        snapshot_id = _pinned_snapshot(conn)
        rows = conn.execute(
            "SELECT ordinal, heading_path, text FROM chunks"
            " WHERE snapshot_id = ? AND doc_id = ? AND ordinal BETWEEN ? AND ?"
            " ORDER BY ordinal",
            (snapshot_id, doc_id, ordinal - max(0, before), ordinal + max(0, after)),
        ).fetchall()
        return [{"ordinal": r[0], "heading_path": r[1], "text": r[2]} for r in rows]
    finally:
        conn.close()


def list_indexed_documents(db_path: Path | None = None) -> list[dict]:
    conn = _connect(db_path, readonly=True)
    try:
        snapshot_id = _pinned_snapshot(conn)
        rows = conn.execute(
            "SELECT d.doc_id, d.title, d.domain, d.authority, d.meta_json,"
            " (SELECT COUNT(*) FROM chunks c WHERE c.snapshot_id = d.snapshot_id AND c.doc_id = d.doc_id)"
            " FROM documents d WHERE d.snapshot_id = ? ORDER BY d.domain, d.title",
            (snapshot_id,),
        ).fetchall()
        out = []
        for doc_id, title, domain, authority, meta_json, n in rows:
            meta = json.loads(meta_json)
            out.append({"doc_id": doc_id, "title": title, "domain": domain,
                        "authority": authority, "summary": meta.get("summary"),
                        "topics": meta.get("topics", []), "sections": n})
        return out
    finally:
        conn.close()


def index_status(db_path: Path | None = None) -> dict:
    """Snapshot pin + counts, exposed via the MCP get_index_status tool."""
    conn = _connect(db_path, readonly=True)
    try:
        snapshot_id = _pinned_snapshot(conn)
        row = conn.execute(
            "SELECT created_at, embedding_model, doc_count, chunk_count, note"
            " FROM snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        model_row = conn.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
        return {
            "pinned_snapshot": snapshot_id,
            "created_at": row[0],
            "embedding_model": row[1],
            "configured_embedding_model": model_row[0] if model_row else None,
            "documents": row[2],
            "chunks": row[3],
            "note": row[4],
        }
    finally:
        conn.close()
