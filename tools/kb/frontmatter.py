"""Parse and validate corpus document front matter and sidecars.

Front matter lives at the top of each corpus file between ``---`` fences.
The metadata is validated against the JSON schema and the controlled
vocabularies so a bad document fails loudly at ingest time, not at query
time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import config

FENCE_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

# Valid ID characters for section_path segments (used by manifest lookups).
ID_RE = re.compile(r"\A[a-z0-9]+(-[a-z0-9]+)*\Z")


def _load_vocab(name: str, key: str) -> dict[str, str]:
    path = config.VOCAB_DIR / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data[key]


def _load_topic_set() -> set[str]:
    data = yaml.safe_load((config.VOCAB_DIR / "topics.yaml").read_text(encoding="utf-8"))
    return set(data["topics"])


class DocumentMeta(BaseModel):
    """Front matter model. Mirrors schema/document.schema.json."""

    model_config = {"extra": "forbid"}

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    domain: str
    topics: list[str] = Field(min_length=1)
    doc_type: str
    source_file: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority: str
    precedence: int = Field(ge=1, le=100)
    status: str
    effective_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    extraction_method: str
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool = False
    summary: str = Field(min_length=1)
    key_terms: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    chunk_max_tokens: int | None = Field(default=None, ge=40, le=800)

    @field_validator("authority")
    @classmethod
    def _authority_ok(cls, value: str) -> str:
        allowed = {"canonical", "supporting", "contextual", "superseded"}
        if value not in allowed:
            raise ValueError(f"authority must be one of {sorted(allowed)}")
        return value

    @field_validator("status")
    @classmethod
    def _status_ok(cls, value: str) -> str:
        allowed = {"draft", "reviewed", "published", "superseded"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value

    @field_validator("extraction_method")
    @classmethod
    def _method_ok(cls, value: str) -> str:
        allowed = {"native-text", "ocr", "vision", "manual"}
        if value not in allowed:
            raise ValueError(f"extraction_method must be one of {sorted(allowed)}")
        return value


@dataclass
class CorpusDocument:
    """A corpus file split into validated metadata and body text."""

    path: Path
    meta: DocumentMeta
    body: str  # markdown body below the front matter fence

    @property
    def rel_path(self) -> str:
        return self.path.relative_to(config.ROOT).as_posix()


class FrontMatterError(ValueError):
    """Raised when a corpus file fails parsing or validation."""


def split_front_matter(text: str) -> tuple[dict, str]:
    """Split raw file text into (front matter dict, body). Raises on missing fence."""
    match = FENCE_RE.match(text)
    if not match:
        raise FrontMatterError("file does not start with a '---' front matter fence")
    try:
        data = yaml.safe_load(match.group(1))
    except (yaml.YAMLError, RecursionError) as exc:
        raise FrontMatterError(f"invalid YAML front matter: {exc}") from exc
    if not isinstance(data, dict):
        raise FrontMatterError("front matter must be a YAML mapping")
    return data, text[match.end():]


def validate_meta(
    data: dict,
    *,
    check_vocab: bool = True,
    source_root: Path | None = None,
) -> DocumentMeta:
    """Validate a front matter dict into a DocumentMeta.

    When ``check_vocab`` is set, domain/topics/doc_type are checked against
    the controlled vocabularies. When ``source_root`` is given, ``source_file``
    must stay inside it. The original itself may be absent: ``raw/`` is not
    part of the repository, and indexing uses the committed Markdown.
    """
    try:
        meta = DocumentMeta.model_validate(data)
    except ValidationError as exc:
        raise FrontMatterError(str(exc)) from exc

    if meta.extraction_method in {"ocr", "vision"} and meta.ocr_confidence is None:
        raise FrontMatterError(
            "ocr_confidence is required when extraction_method is ocr or vision"
        )

    if check_vocab:
        domains = set(_load_vocab("domains", "domains"))
        doc_types = set(_load_vocab("doc-types", "doc_types"))
        topics = _load_topic_set()
        if meta.domain not in domains:
            raise FrontMatterError(
                f"unknown domain '{meta.domain}'; add it to schema/vocab/domains.yaml"
            )
        if meta.doc_type not in doc_types:
            raise FrontMatterError(
                f"unknown doc_type '{meta.doc_type}'; add it to schema/vocab/doc-types.yaml"
            )
        unknown_topics = [t for t in meta.topics if t not in topics]
        if unknown_topics:
            raise FrontMatterError(
                f"unknown topics {unknown_topics}; add them to schema/vocab/topics.yaml"
            )

    if source_root is not None:
        sf = Path(meta.source_file)
        if sf.is_absolute() or ".." in sf.parts:
            raise FrontMatterError(f"source_file '{meta.source_file}' must be a relative path inside raw/")
        source_path = (source_root / sf).resolve()
        if not source_path.is_relative_to(source_root.resolve()):
            raise FrontMatterError(f"source_file '{meta.source_file}' escapes raw/")

    return meta


def load_document(path: Path, *, check_vocab: bool = True) -> CorpusDocument:
    """Load and validate one corpus file."""
    text = path.read_text(encoding="utf-8")
    data, body = split_front_matter(text)
    meta = validate_meta(data, check_vocab=check_vocab, source_root=config.RAW_DIR)
    return CorpusDocument(path=path, meta=meta, body=body)


def iter_corpus_files(corpus_dir: Path | None = None) -> list[Path]:
    """All corpus markdown files, excluding staging and sidecars."""
    root = corpus_dir or config.CORPUS_DIR
    files: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        parts = path.relative_to(root).parts
        if parts and parts[0] == ".staging":
            continue
        files.append(path)
    return files


def sidecar_path_for(doc_path: Path) -> Path:
    """Sidecar lives beside the document as <stem>.meta.yaml."""
    return doc_path.with_suffix(".meta.yaml")


def load_sidecar(doc_path: Path) -> dict | None:
    """Load a document's sidecar if present, validated loosely (agents write it)."""
    path = sidecar_path_for(doc_path)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("doc_id") is None:
        raise FrontMatterError(f"invalid sidecar {path}: missing doc_id")
    return data


def dump_sidecar(doc_path: Path, data: dict) -> Path:
    """Write a sidecar with stable key order for clean diffs."""
    path = sidecar_path_for(doc_path)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    return path


def json_schema() -> dict:
    """The published JSON schema (kept as the human-readable contract)."""
    return json.loads((config.SCHEMA_DIR / "document.schema.json").read_text("utf-8"))
