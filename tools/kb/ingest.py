"""Ingestion pipeline: raw/inbox -> corpus.

Pipeline per file:
  1. Convert (deterministic converters; vision for images/scans).
  2. OCR/vision double-pass with comparison; disagreements -> review.
  3. Proofreader subagent proposes spelling/grammar edits as structured
     operations (never rewritten prose).
  4. Deterministic integrity gate classifies every change; code applies
     only ACCEPTed edits, logs them to <id>.edits.jsonl.
  5. Structurer subagent writes the sidecar (section map, summary, terms).
  6. Corpus file + front matter written to corpus/.staging for review
     (vision/OCR) or directly to corpus/ (native text with clean gate).

Agents propose; code applies and checks. No agent writes to raw/ or to
the corpus except through this module.
"""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .converters import Conversion, convert, convert_image_vision
from .frontmatter import dump_sidecar, validate_meta
from .hashing import canonicalise, sha256_file, sha256_text
from .integrity import GateResult, Verdict, compare
from .llm import LLMClient, LLMError

EDIT_CATEGORIES_AUTO = {"spelling", "punctuation", "case", "hyphenation", "whitespace"}


@dataclass
class ProposedEdit:
    """One structured edit proposed by the proofreader subagent."""

    before: str
    after: str
    category: str  # spelling | punctuation | grammar


@dataclass
class IngestResult:
    source: Path
    doc_id: str
    corpus_path: Path
    staged: bool  # True when written to .staging for human review
    gate: GateResult | None
    edits_applied: int
    edits_rejected: int
    review_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Proofreader subagent
# ---------------------------------------------------------------------------

_PROOFREAD_SYSTEM = """You are a proofreading engine for archival texts. You may ONLY
propose corrections for clear spelling mistakes, punctuation errors, and
simple grammar slips (agreement, doubled words, missing apostrophes).

Hard rules:
- NEVER reword, reorder, modernise, or "improve" phrasing.
- NEVER change numbers, dates, names, negations, citations, section
  references, or defined terms.
- Preserve archaic or unusual wording exactly; fix only clear typos.
- When unsure, propose nothing for that span.

Return JSON only: {"edits": [{"before": "exact original span",
"after": "corrected span", "category": "spelling|punctuation|grammar"}]}
Each "before" span must be copied character-for-character from the text
and must be short (a word to a sentence)."""


def propose_edits(client: LLMClient, text: str) -> list[ProposedEdit]:
    """Proofreader subagent: structured edit proposals, not rewritten prose."""
    # Process in windows so long documents stay within context.
    window = 12000
    edits: list[ProposedEdit] = []
    for start in range(0, len(text), window):
        chunk = text[start : start + window]
        try:
            raw = client.chat(
                role="proofreader",
                system=_PROOFREAD_SYSTEM,
                user=f"Text:\n\n{chunk}",
                json_mode=True,
            )
            payload = json.loads(raw)
        except (LLMError, json.JSONDecodeError):
            continue  # a failed window proposes nothing; safe default
        for item in payload.get("edits", []):
            before = str(item.get("before", ""))
            after = str(item.get("after", ""))
            category = str(item.get("category", ""))
            if before and after and before != after:
                edits.append(ProposedEdit(before=before, after=after, category=category))
    return edits


def apply_edits(
    text: str,
    edits: list[ProposedEdit],
    original: str,
) -> tuple[str, GateResult, list[dict]]:
    """Apply proposed edits through the deterministic gate.

    Each proposal is applied to a working copy; the working copy is then
    compared against the original in one pass, and only ACCEPT-category
    differences survive into the final text. This means a bad proposal is
    silently discarded rather than shipped.
    """
    working = text
    applied_log: list[dict] = []
    for edit in edits:
        if edit.before not in working:
            continue  # model hallucinated a span; ignore
        candidate = working.replace(edit.before, edit.after, 1)
        gate = compare(original, candidate)
        if gate.verdict == Verdict.ACCEPT:
            working = candidate
            applied_log.append(
                {
                    "before": edit.before,
                    "after": edit.after,
                    "category": edit.category,
                    "verdict": "accept",
                }
            )
        else:
            applied_log.append(
                {
                    "before": edit.before,
                    "after": edit.after,
                    "category": edit.category,
                    "verdict": gate.verdict.value,
                    "reasons": [c.reason for c in gate.changes if c.verdict != Verdict.ACCEPT][:5],
                }
            )
    final_gate = compare(original, working)
    return working, final_gate, applied_log


# ---------------------------------------------------------------------------
# Structurer subagent
# ---------------------------------------------------------------------------

_STRUCTURER_SYSTEM = """You are a document structuring engine. Given a markdown document,
produce a JSON object describing its structure and retrieval metadata.
You NEVER alter the document text; you only describe it.

Return JSON only:
{
  "summary": "2-4 sentence factual summary",
  "key_terms": ["important terms as written in the text"],
  "aliases": ["synonyms / plain-English phrasings a searcher might use"],
  "section_map": [{"heading": "exact heading text or 'Untitled section'",
                   "level": 1,
                   "start_paragraph": 0,
                   "summary": "one sentence"}]
}
Paragraphs are separated by blank lines; start_paragraph is the index of
the paragraph where the section begins. Headings must match the document
text exactly."""


def propose_sidecar(client: LLMClient, doc_id: str, body: str) -> dict:
    """Structurer subagent: sidecar metadata (never touches the text)."""
    try:
        raw = client.chat(
            role="structurer",
            system=_STRUCTURER_SYSTEM,
            user=f"Document:\n\n{body[:60000]}",
            json_mode=True,
        )
        payload = json.loads(raw)
    except (LLMError, json.JSONDecodeError):
        payload = {}
    return {
        "doc_id": doc_id,
        "summary": payload.get("summary"),
        "key_terms": payload.get("key_terms", []),
        "aliases": payload.get("aliases", []),
        "section_map": payload.get("section_map", []),
        "conflict_candidates": [],
        "notes": None,
    }


# ---------------------------------------------------------------------------
# Metadata proposal
# ---------------------------------------------------------------------------

_META_SYSTEM = """You are a cataloguing engine. Propose front matter metadata for the
document as JSON. Choose domain, topics and doc_type ONLY from the allowed
lists supplied. Suggest a kebab-case id. Be conservative: when unsure of
authority, use "supporting" and precedence 50.

Return JSON only:
{"id": "...", "title": "...", "domain": "...", "topics": ["..."],
 "doc_type": "...", "authority": "canonical|supporting|contextual",
 "precedence": 50, "summary": "...", "key_terms": ["..."], "aliases": ["..."]}"""


def propose_metadata(client: LLMClient, body: str, vocab: dict) -> dict:
    allowed = json.dumps(vocab)
    try:
        raw = client.chat(
            role="structurer",
            system=_META_SYSTEM,
            user=f"Allowed vocabularies:\n{allowed}\n\nDocument (truncated):\n\n{body[:20000]}",
            json_mode=True,
        )
        return json.loads(raw)
    except (LLMError, json.JSONDecodeError):
        return {}


def _load_vocabs() -> dict:
    import yaml

    return {
        "domains": list(yaml.safe_load((config.VOCAB_DIR / "domains.yaml").read_text("utf-8"))["domains"]),
        "doc_types": list(yaml.safe_load((config.VOCAB_DIR / "doc-types.yaml").read_text("utf-8"))["doc_types"]),
        "topics": yaml.safe_load((config.VOCAB_DIR / "topics.yaml").read_text("utf-8"))["topics"],
    }


# ---------------------------------------------------------------------------
# OCR double pass
# ---------------------------------------------------------------------------

def ocr_double_pass(path: Path, client: LLMClient) -> tuple[Conversion, list[str]]:
    """Two independent vision passes; disagreements become review flags."""
    first = convert_image_vision(path, client, role="extractor")
    second = convert_image_vision(path, client, role="verifier")
    flags: list[str] = []
    ratio = difflib.SequenceMatcher(
        None, first.text.split(), second.text.split(), autojunk=False
    ).ratio()
    if ratio < 0.98:
        flags.append(
            f"OCR passes disagree (similarity {ratio:.3f}); human review required"
        )
    # Keep the higher-confidence pass as the base text.
    chosen = first if (first.confidence or 0) >= (second.confidence or 0) else second
    chosen.notes += f"; double-pass similarity {ratio:.3f}"
    if ratio < 0.98:
        chosen.confidence = min(chosen.confidence or 1.0, 0.9)
    return chosen, flags


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return re.sub(r"-{2,}", "-", value) or "untitled"


def ingest_file(
    path: Path,
    client: LLMClient | None = None,
    *,
    overrides: dict | None = None,
) -> IngestResult:
    """Ingest one raw file into the corpus. Safe: writes only corpus outputs."""
    overrides = overrides or {}
    source_sha = sha256_file(path)
    review_flags: list[str] = []

    is_image = path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"}
    if is_image:
        if client is None:
            raise RuntimeError(f"{path.name}: images require an LLM client (vision path)")
        conversion, ocr_flags = ocr_double_pass(path, client)
        review_flags.extend(ocr_flags)
    else:
        conversion = convert(path, client)

    original_text = canonicalise(conversion.text)

    # Proofreading through the deterministic gate (native text only; vision
    # output already passed the double-pass check and is left verbatim).
    gate: GateResult | None = None
    edits_applied = edits_rejected = 0
    final_text = original_text
    edit_log: list[dict] = []
    if client is not None and conversion.extraction_method == "native-text":
        proposed = propose_edits(client, original_text)
        final_text, gate, edit_log = apply_edits(original_text, proposed, original_text)
        edits_applied = sum(1 for e in edit_log if e["verdict"] == "accept")
        edits_rejected = len(edit_log) - edits_applied
        if gate.verdict == Verdict.REVIEW:
            review_flags.append("integrity gate requires human review of some changes")

    # Metadata: LLM proposal, human overridable, then strict validation.
    vocab = _load_vocabs()
    proposed_meta = propose_metadata(client, final_text, vocab) if client else {}
    meta_dict = {
        "id": overrides.get("id") or _slugify(proposed_meta.get("id") or path.stem),
        "title": overrides.get("title") or proposed_meta.get("title") or path.stem,
        "domain": overrides.get("domain") or proposed_meta.get("domain"),
        "topics": overrides.get("topics") or proposed_meta.get("topics") or [],
        "doc_type": overrides.get("doc_type") or proposed_meta.get("doc_type"),
        "source_file": path.relative_to(config.RAW_DIR).as_posix()
        if path.is_relative_to(config.RAW_DIR)
        else f"inbox/{path.name}",
        "source_sha256": source_sha,
        "authority": overrides.get("authority") or proposed_meta.get("authority", "supporting"),
        "precedence": int(overrides.get("precedence") or proposed_meta.get("precedence", 50)),
        "status": "draft",
        "extraction_method": conversion.extraction_method,
        "ocr_confidence": conversion.confidence,
        "needs_review": bool(review_flags),
        "summary": proposed_meta.get("summary") or overrides.get("summary") or "Pending summary.",
        "key_terms": proposed_meta.get("key_terms", []),
        "aliases": proposed_meta.get("aliases", []),
    }
    meta = validate_meta(meta_dict, check_vocab=True, source_root=None)

    # Destination: staging for anything flagged or vision-derived.
    staged = bool(review_flags) or conversion.extraction_method in {"ocr", "vision"}
    base_dir = config.STAGING_DIR if staged else (config.CORPUS_DIR / meta.domain)
    base_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = base_dir / f"{meta.id}.md"

    import yaml

    front = yaml.safe_dump(
        {k: v for k, v in meta.model_dump().items() if v not in (None, [], False)},
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    corpus_path.write_text(f"---\n{front}---\n\n{final_text}", encoding="utf-8")

    # Edits log (auditable, revertable).
    if edit_log:
        edits_path = corpus_path.with_suffix(".edits.jsonl")
        with edits_path.open("w", encoding="utf-8") as handle:
            for entry in edit_log:
                entry = {**entry, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                handle.write(json.dumps(entry) + "\n")

    # Sidecar from the structurer subagent.
    if client is not None:
        sidecar = propose_sidecar(client, meta.id, final_text)
        dump_sidecar(corpus_path, sidecar)

    return IngestResult(
        source=path,
        doc_id=meta.id,
        corpus_path=corpus_path,
        staged=staged,
        gate=gate,
        edits_applied=edits_applied,
        edits_rejected=edits_rejected,
        review_flags=review_flags,
    )


def ingest_inbox(client: LLMClient | None = None) -> list[IngestResult]:
    """Ingest everything in raw/inbox/. Returns per-file results."""
    config.ensure_dirs()
    results: list[IngestResult] = []
    for path in sorted(config.RAW_INBOX.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            results.append(ingest_file(path, client))
    return results
