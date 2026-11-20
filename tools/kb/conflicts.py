"""Contradiction detection workflow with human review.

Flow:
  1. The conflict subagent (strong model) compares documents within the
     same topic cluster and records *candidates* in each sidecar's
     `conflict_candidates` list (status=pending). It never edits front
     matter and never edits text.
  2. A human reviews candidates (`kb conflicts list/confirm/reject`).
  3. Confirmed candidates are promoted to front matter `conflicts_with`
     by the review command — the only path that can write that field.
  4. The MCP `list_conflicts` tool exposes confirmed conflicts and which
     document prevails (lowest precedence number wins; authority breaks
     ties: canonical > supporting > contextual).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config
from .frontmatter import (
    dump_sidecar,
    iter_corpus_files,
    load_document,
    load_sidecar,
    split_front_matter,
)
from .llm import LLMClient, LLMError

_CONFLICT_SYSTEM = """You are a contradiction-detection engine for a private knowledge base.
Given several documents on the same topic, identify pairs that assert
incompatible claims about the same subject. Be conservative: only flag a
pair when the claims genuinely cannot both be true. Cite the specific
passage from each document.

Return JSON only:
{"conflicts": [{"doc_id": "...", "passage_a": "verbatim quote from the FIRST document",
"passage_b": "verbatim quote from the other document", "note": "one sentence"}]}
If none, return {"conflicts": []}."""


@dataclass
class ConflictCandidate:
    doc_id: str
    other_doc_id: str
    note: str
    passage_a: str
    passage_b: str


def detect_conflicts(client: LLMClient, *, max_docs_per_cluster: int = 6) -> int:
    """Run the conflict subagent across topic clusters. Returns candidate count."""
    # Cluster documents by shared topic.
    clusters: dict[str, list] = {}
    for path in iter_corpus_files():
        doc = load_document(path, check_vocab=False)
        for topic in doc.meta.topics:
            clusters.setdefault(topic, []).append(doc)

    added = 0
    for topic, docs in clusters.items():
        if len(docs) < 2:
            continue
        docs = sorted(docs, key=lambda d: d.meta.precedence)[:max_docs_per_cluster]
        base = docs[0]
        others = [d for d in docs[1:]]
        bundle = "\n\n".join(
            f"--- DOC {d.meta.id} ---\n{d.body[:8000]}" for d in [base, *others]
        )
        try:
            raw = client.chat(
                role="conflict",
                system=_CONFLICT_SYSTEM,
                user=f"First document id: {base.meta.id}\n\nDocuments:\n\n{bundle}",
                json_mode=True,
            )
            payload = json.loads(raw)
        except (LLMError, json.JSONDecodeError):
            continue

        for item in payload.get("conflicts", []):
            other_id = item.get("doc_id")
            if not other_id or other_id == base.meta.id:
                continue
            sidecar = load_sidecar(base.path) or {
                "doc_id": base.meta.id,
                "section_map": [],
                "conflict_candidates": [],
            }
            candidates = sidecar.setdefault("conflict_candidates", [])
            if any(c.get("doc_id") == other_id and c.get("status") == "pending" for c in candidates):
                continue
            candidates.append(
                {
                    "doc_id": other_id,
                    "note": item.get("note", ""),
                    "passage_a": item.get("passage_a", ""),
                    "passage_b": item.get("passage_b", ""),
                    "status": "pending",
                }
            )
            dump_sidecar(base.path, sidecar)
            added += 1
    return added


def list_candidates() -> list[dict]:
    """All pending conflict candidates across the corpus."""
    out: list[dict] = []
    for path in iter_corpus_files():
        sidecar = load_sidecar(path)
        if not sidecar:
            continue
        for candidate in sidecar.get("conflict_candidates", []):
            if candidate.get("status") == "pending":
                out.append(
                    {
                        "doc_id": sidecar["doc_id"],
                        "other_doc_id": candidate["doc_id"],
                        "note": candidate.get("note", ""),
                        "passage_a": candidate.get("passage_a", ""),
                        "passage_b": candidate.get("passage_b", ""),
                    }
                )
    return out


def _set_candidate_status(doc_path: Path, other_id: str, status: str) -> bool:
    sidecar = load_sidecar(doc_path)
    if not sidecar:
        return False
    changed = False
    for candidate in sidecar.get("conflict_candidates", []):
        if candidate.get("doc_id") == other_id and candidate.get("status") == "pending":
            candidate["status"] = status
            changed = True
    if changed:
        dump_sidecar(doc_path, sidecar)
    return changed


def _promote_to_front_matter(doc_path: Path, other_id: str) -> None:
    """Add other_id to the document's front matter conflicts_with list."""
    text = doc_path.read_text(encoding="utf-8")
    data, body = split_front_matter(text)
    conflicts = list(data.get("conflicts_with", []))
    if other_id not in conflicts:
        conflicts.append(other_id)
    data["conflicts_with"] = conflicts
    front = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    doc_path.write_text(f"---\n{front}---\n{body}", encoding="utf-8")


def review_candidate(doc_id: str, other_id: str, *, confirm: bool) -> bool:
    """Human review decision. Confirm promotes to front matter (both ways)."""
    target = None
    for path in iter_corpus_files():
        if path.stem == doc_id:
            target = path
            break
    if target is None:
        return False

    if confirm:
        _set_candidate_status(target, other_id, "confirmed")
        _promote_to_front_matter(target, other_id)
        # Record the reverse direction too, when the other document exists.
        for path in iter_corpus_files():
            if path.stem == other_id:
                _promote_to_front_matter(path, doc_id)
                break
        return True
    return _set_candidate_status(target, other_id, "rejected")


def confirmed_conflicts(topic: str | None = None) -> list[dict]:
    """Confirmed conflicts with the prevailing document, for the MCP tool."""
    authority_rank = {"canonical": 0, "supporting": 1, "contextual": 2, "superseded": 3}
    docs = {}
    for path in iter_corpus_files():
        doc = load_document(path, check_vocab=False)
        docs[doc.meta.id] = doc

    out: list[dict] = []
    for doc_id, doc in docs.items():
        if topic and topic not in doc.meta.topics:
            continue
        for other_id in doc.meta.conflicts_with:
            other = docs.get(other_id)
            if other is None:
                continue
            # Which prevails: precedence first, then authority rank.
            winner, loser = doc, other
            if (other.meta.precedence, authority_rank.get(other.meta.authority, 9)) < (
                doc.meta.precedence,
                authority_rank.get(doc.meta.authority, 9),
            ):
                winner, loser = other, doc
            # Collect any confirmed sidecar notes for this pair.
            notes: list[str] = []
            for d in (doc, other):
                sidecar = load_sidecar(d.path) or {}
                other_target = other_id if d is doc else doc_id
                for cand in sidecar.get("conflict_candidates", []):
                    if cand.get("doc_id") == other_target and cand.get("status") == "confirmed":
                        note = cand.get("note", "").strip()
                        if note and note not in notes:
                            notes.append(note)
            out.append(
                {
                    "doc_id": doc_id,
                    "conflicts_with": other_id,
                    "prevails": winner.meta.id,
                    "reason": (
                        f"'{winner.meta.title}' prevails over '{loser.meta.title}' "
                        f"(precedence {winner.meta.precedence} vs {loser.meta.precedence}, "
                        f"authority {winner.meta.authority} vs {loser.meta.authority})"
                    ),
                    "notes": notes,
                }
            )
    # De-duplicate symmetric pairs.
    seen = set()
    unique = []
    for item in out:
        key = tuple(sorted([item["doc_id"], item["conflicts_with"]]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
