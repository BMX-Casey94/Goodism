"""Central configuration: repository paths, tunables, environment.

Single source of truth so every tool (ingest, index, server, eval) agrees
on where things live. Nothing here holds secrets; secrets come from .env.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Repository root = three levels up from this file (tools/kb/config.py).
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "raw"
RAW_INBOX = RAW_DIR / "inbox"
CORPUS_DIR = ROOT / "corpus"
STAGING_DIR = CORPUS_DIR / ".staging"
SCHEMA_DIR = ROOT / "schema"
VOCAB_DIR = SCHEMA_DIR / "vocab"
TOOLS_DIR = ROOT / "tools"
PROMPTS_DIR = TOOLS_DIR / "kb" / "prompts"
SKILLS_DIR = TOOLS_DIR / "kb" / "skills"
EVAL_DIR = ROOT / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"
SERVER_DIR = ROOT / "server"
MANIFEST_DIR = ROOT / "manifest"
PROOFS_DIR = MANIFEST_DIR / "proofs"
MANIFEST_PATH = MANIFEST_DIR / "manifest.jsonl"
MANIFEST_ANCHOR_PATH = MANIFEST_DIR / "manifest.anchor.json"
INDEX_DIR = ROOT / "index"
INDEX_DB_PATH = INDEX_DIR / "kb.sqlite"

# Chunking tunables (structure-aware; never split mid-table or mid-list).
CHUNK_MIN_TOKENS = 300
CHUNK_MAX_TOKENS = int(os.environ.get("KB_CHUNK_MAX_TOKENS", "800"))
# Index-time keyword enrichment (distinctive terms appended to indexed text only).
INDEX_KEYWORDS = os.environ.get("KB_INDEX_KEYWORDS", "0") == "1"
# Rough token estimate: 1 token ~ 4 characters of English prose.
CHARS_PER_TOKEN = 4

# Retrieval tunables.
RRF_K = 60  # reciprocal rank fusion constant
SEARCH_CANDIDATES_PER_LEG = 60
RERANK_KEEP = 30  # candidates sent to reranker
DEFAULT_TOP_K = 8

# Embedding configuration. Model version is pinned in index metadata; a
# change triggers a full rebuild (see tools/kb/indexer.py).
EMBEDDING_MODEL = os.environ.get("KB_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS = 1536

# Canonicalisation (must match the on-chain publication step exactly).
CANONICAL_ENCODING = "utf-8"


@lru_cache(maxsize=1)
def load_dotenv_if_present() -> None:
    """Load .env for local runs. No-op when python-dotenv is absent."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def ensure_dirs() -> None:
    """Create runtime directories that are git-ignored or lazily used."""
    for path in (RAW_INBOX, STAGING_DIR, INDEX_DIR, EVAL_RESULTS_DIR, PROOFS_DIR):
        path.mkdir(parents=True, exist_ok=True)
