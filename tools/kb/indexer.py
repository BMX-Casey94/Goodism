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
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .chunker import Chunk, chunk_document
from .frontmatter import CorpusDocument, iter_corpus_files, load_document
from .hashing import sha256_text

SCHEMA_VERSION = 1


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.INDEX_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


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
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._local is not None:
            return self._local.embed(texts)
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def _load_manifest_records() -> dict[tuple[str, str], dict]:
    """Map (doc_id, section_path-normalised heading path) -> manifest record."""
    records: dict[tuple[str, str], dict] = {}
    if not config.MANIFEST_PATH.exists():
        return records
    for line in config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[(record["doc_id"], record["section_path"])] = record
    return records


@dataclass
class BuildReport:
    snapshot_id: int
    documents: int
    chunks: int
    rebuilt_docs: list[str]
    skipped_unchanged: list[str]
    full_rebuild: bool


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
        embedder = embedder or Embedder()

        # Full rebuild triggers: explicit flag, or embedding model change.
        row = conn.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
        previous_model = row[0] if row else None
        full = full or (previous_model is not None and previous_model != embedder.model)

        if full:
            for table in ("chunks_fts", "chunks", "documents", "snapshots"):
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

            chunks = chunk_document(doc.meta.id, doc.meta.title, doc.body)
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
                        _pack_embedding(embedding),
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

        # Rebuild FTS for the new snapshot rows. The FTS table is
        # contentless ('' content mode) so rows must be deleted by rowid.
        conn.execute("DELETE FROM chunks_fts WHERE rowid NOT IN (SELECT id FROM chunks WHERE snapshot_id = ?)", (snapshot_id,))
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


def _fts_escape(query: str) -> str:
    """Build a safe FTS5 MATCH expression from a free-text query.

    Splits into terms and double-quotes each one, joining with OR, so user
    punctuation (?, ", *, etc.) can never be parsed as FTS5 operators.
    """
    terms = re.findall(r"[\w'-]+", query)
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)


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
    conn = _connect(db_path)
    try:
        snapshot_id = _pinned_snapshot(conn)
        embedder = embedder or Embedder()

        doc_filter = ""
        params: list = [snapshot_id]
        if domain:
            doc_filter += " AND d.domain = ?"
            params.append(domain)
        if not include_superseded:
            doc_filter += " AND d.status != 'superseded' AND d.authority != 'superseded'"

        # Keyword leg (BM25; lower is better in FTS5, so negate).
        match_expr = _fts_escape(query)
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
        query_vec = embedder.embed([query])[0]
        vector_rank: dict[int, int] = {}
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

        id_list = ",".join(str(cid) for cid, _, _ in fused)
        detail_rows = conn.execute(
            f"""
            SELECT c.id, c.doc_id, d.title, c.heading_path, c.text,
                   d.authority, d.precedence, d.status
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id AND d.snapshot_id = c.snapshot_id
            WHERE c.id IN ({id_list})
            """
        ).fetchall()
        details = {row[0]: row for row in detail_rows}

        manifest_records = _load_manifest_records()

        hits: list[SearchHit] = []
        for chunk_id, score, source in fused:
            row = details.get(chunk_id)
            if row is None:
                continue
            _, doc_id, title, heading_path, text, authority, precedence, status = row
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
                )
            )

        # Authority precedence as a tie-break nudge before rerank.
        hits.sort(key=lambda h: (h.score, -h.precedence), reverse=True)

        if rerank:
            reranker = Reranker()
            if reranker.available():
                hits = reranker.rerank(query, hits, keep=top_k)
                return hits
        return hits[:top_k]
    finally:
        conn.close()


def index_status(db_path: Path | None = None) -> dict:
    """Snapshot pin + counts, exposed via the MCP get_index_status tool."""
    conn = _connect(db_path)
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
