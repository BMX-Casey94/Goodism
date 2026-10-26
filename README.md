# Goodism knowledge base

A git-versioned corpus of verified Markdown, indexed by a rebuildable hybrid
search index, and served to an LLM via a read-only MCP server. Provenance is
anchored on BSV (plaintext, one TXID per section) via an append-only manifest.

## Layout

- `raw/inbox/` — drop source files here (PDF, DOCX, TXT, images). Never edited.
- `corpus/` — normalised Markdown, one file per source, plus `.meta.yaml` sidecars
  and `.edits.jsonl` logs. Files here are verbatim except for logged
  spelling/grammar fixes that passed the deterministic integrity gate.
- `corpus/.staging/` — OCR/vision output awaiting human review.
- `schema/` — front-matter JSON schema and controlled vocabularies.
- `tools/kb/` — ingest, chunker, indexer, validator, conflicts, manifest, MCP server.
- `manifest/` — append-only `manifest.jsonl` mapping sections to TXIDs.
- `eval/` — evaluation question set and CI.
- `index/` — generated SQLite index (git-ignored, rebuildable).

## Quick start

```powershell
pip install -e ".[ingest,dev]"
copy .env.example .env   # fill in keys
kb ingest                 # ingest raw/inbox into corpus/
kb validate               # CI gate: schema, hashes, duplicates, links
kb build-index            # build the hybrid search index
kb search "what is a cestui que vie trust"
kb status                 # pinned snapshot info
```

## Principles

- **Files are the source of truth; the index is disposable** and rebuildable
  with one command.
- **Text integrity is enforced by code, not an LLM.** Every change is
  classified; anything touching numbers, dates, names, negations or citations
  is rejected or sent to review.
- **Agents add structure around the text, never inside it.** Summaries and
  synonyms live in sidecars and index-time prefixes only.
- **Provenance is on-chain; the manifest is the single lookup.** The LLM copies
  a ready-made record; it never generates a TXID.

See the plan in `knowledge_base_architecture` for the full design.
