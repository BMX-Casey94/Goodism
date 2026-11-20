"""Phrase lexicon: connects what users type to the wording the documents use.

``schema/lexicon.yaml`` lists concepts. Each has a plain canonical term and
variants that appear verbatim in the corpus. When a query mentions any term
of a concept, the other terms of that concept are added to the keyword
search, so "good faith" also finds "honesty in fact". Expansion only adds
search terms; it never changes a document.

Format::

    terms:
      - canonical: "good faith"
        variants: ["honesty in fact"]
        docs: [complete-ucc-1-9]
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

from . import config

LEXICON_PATH = config.ROOT / "schema" / "lexicon.yaml"
MAX_EXPANSIONS = 6


@lru_cache(maxsize=1)
def _load(path: str, mtime: float) -> list[tuple[re.Pattern, list[str]]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries: list[tuple[re.Pattern, list[str]]] = []
    for item in data.get("terms", []):
        phrases = [item["canonical"], *item.get("variants", [])]
        phrases = [p.strip() for p in phrases if p and p.strip()]
        if len(phrases) < 2:
            continue
        alternation = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
        entries.append((re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE), phrases))
    return entries


def expansions(query: str, path: Path | None = None) -> list[str]:
    """Extra phrases to search for, given a user query. Empty if no lexicon."""
    lexicon = path or LEXICON_PATH
    if os.environ.get("KB_LEXICON", "1") == "0" or not lexicon.exists():
        return []
    extra: list[str] = []
    lowered = query.lower()
    for pattern, phrases in _load(str(lexicon), lexicon.stat().st_mtime):
        if pattern.search(query):
            for phrase in phrases:
                if phrase.lower() not in lowered and phrase not in extra:
                    extra.append(phrase)
        if len(extra) >= MAX_EXPANSIONS:
            break
    return extra[:MAX_EXPANSIONS]
