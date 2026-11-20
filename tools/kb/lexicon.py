"""Phrase lexicon and user-wording triggers.

Connects what users type to the wording the documents use.

- ``schema/lexicon.yaml`` lists symmetric concepts: matching any term adds the
  others (e.g. "good faith" <-> "honesty in fact").
- ``schema/triggers.yaml`` lists one-way triggers: matching a user phrase
  (misspelling, UK/US variant, everyday synonym, acronym) adds verified
  verbatim corpus phrases (e.g. "sestui que vie" -> "cestui que vie").

Expansion only adds search terms to the FTS keyword leg; it never modifies a
document.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

from . import config

LEXICON_PATH = config.ROOT / "schema" / "lexicon.yaml"
TRIGGERS_PATH = config.ROOT / "schema" / "triggers.yaml"
MAX_EXPANSIONS = 8


def _compile(triggers: list[str]) -> re.Pattern:
    alternation = "|".join(re.escape(p) for p in sorted(triggers, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> list[tuple[re.Pattern, list[str]]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries: list[tuple[re.Pattern, list[str]]] = []
    for item in data.get("terms", []):
        phrases = [item["canonical"], *item.get("variants", [])]
        phrases = [p.strip() for p in phrases if p and p.strip()]
        if len(phrases) < 2:
            continue
        entries.append((_compile(phrases), phrases))
    for item in data.get("triggers", []):
        user = [p.strip() for p in item.get("user_phrases", []) if p and p.strip()]
        corpus = [p.strip() for p in item.get("corpus_phrases", []) if p and p.strip()]
        if not user or not corpus:
            continue
        entries.append((_compile(user), corpus))
    return entries


def expansions(
    query: str,
    path: Path | None = None,
    triggers_path: Path | None = None,
) -> list[str]:
    """Extra phrases to search for, given a user query."""
    if os.environ.get("KB_LEXICON", "1") == "0":
        return []
    files: list[Path] = []
    if path is not None:
        files.append(path)
        if triggers_path is not None:
            files.append(triggers_path)
    else:
        # Triggers first so misspellings/user-phrasing get priority slots.
        files.append(triggers_path or TRIGGERS_PATH)
        files.append(LEXICON_PATH)

    extra: list[str] = []
    lowered = query.lower()
    for f in files:
        if not f.exists():
            continue
        for pattern, phrases in _load(str(f), f.stat().st_mtime):
            if pattern.search(query):
                for phrase in phrases:
                    if phrase.lower() not in lowered and phrase not in extra:
                        extra.append(phrase)
            if len(extra) >= MAX_EXPANSIONS:
                return extra[:MAX_EXPANSIONS]
    return extra[:MAX_EXPANSIONS]
