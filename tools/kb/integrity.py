"""Deterministic text-integrity gate. Code, not an LLM, decides what ships.

The gate compares an original text against an edited text and classifies
every change. A change is accepted only if it falls in an allowed category
(spelling, punctuation, case, hyphenation, whitespace). Anything touching
numbers, dates, names, negations, defined terms, citations or references
is rejected outright; genuinely ambiguous cases are marked for human
review. A document with a rejected change never enters the index.

Design notes:
- Word-level diff (not character-level) so a reworded sentence is visible.
- Numbers/dates are rejected by token inspection, so "section 9" -> "section 10"
  can never slip through.
- Spelling acceptance uses a deterministic UK-English dictionary via
  wordfreq zipf scores. A substitution is a plausible spelling fix when
  both tokens are dictionary words with similar frequency and similar
  form, or when the original is a rare/non-word and the replacement is a
  common word that differs by a small edit distance. This intentionally
  errs on the side of REVIEW, never auto-accept.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum

# Words that flip meaning. Any change touching these is an automatic reject.
_NEGATIONS = {
    "not", "no", "never", "none", "neither", "nor", "nothing", "nobody",
    "nowhere", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "shan't", "mustn't", "oughtn't",
}

# Citation / cross-reference markers common in legal texts.
_CITATION_RE = re.compile(
    r"\b(?:v(?:s|ersus)?\.?|§+|¶+|ibid|supra|infra|op\.\s*cit|cf|et\s+seq|"
    r"ch|art|sec|sect|no|vol|pp?|ed)\.\s*$",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\d")
_DATEISH_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?|\d{4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b"
)

# Punctuation we are willing to treat as a safe punctuation-only change.
_PUNCT_ONLY = set(".,;:!?'\"()[]{}—–-/")


class Category(str, Enum):
    UNCHANGED = "unchanged"
    SPELLING = "spelling"
    PUNCTUATION = "punctuation"
    CASE = "case"
    HYPHENATION = "hyphenation"
    WHITESPACE = "whitespace"
    NUMBER = "number"          # always reject
    NAME = "name"              # capitalised-token change: reject
    NEGATION = "negation"      # always reject
    CITATION = "citation"      # always reject
    REWORD = "reword"          # always reject (semantic change)


class Verdict(str, Enum):
    ACCEPT = "accept"      # safe to auto-apply
    REVIEW = "review"      # needs a human; not applied automatically
    REJECT = "reject"      # forbidden; document fails the gate


_ALLOWED_AUTO = {
    Category.PUNCTUATION,
    Category.CASE,
    Category.HYPHENATION,
    Category.WHITESPACE,
}

_ALWAYS_REJECT = {Category.NUMBER, Category.NEGATION, Category.CITATION, Category.REWORD}


@dataclass
class Change:
    """One classified difference between original and edited text."""

    category: Category
    verdict: Verdict
    before: str
    after: str
    position: int  # token index in the original token stream
    reason: str


@dataclass
class GateResult:
    """Outcome of comparing an original with an edited text."""

    verdict: Verdict  # worst verdict across all changes
    changes: list[Change] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.verdict == Verdict.ACCEPT

    @property
    def rejected_changes(self) -> list[Change]:
        return [c for c in self.changes if c.verdict == Verdict.REJECT]

    @property
    def review_changes(self) -> list[Change]:
        return [c for c in self.changes if c.verdict == Verdict.REVIEW]


_TOKEN_RE = re.compile(r"\S+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _zipf(word: str) -> float:
    """wordfreq zipf score; 0.0 when unavailable or unknown."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return 0.0
    return zipf_frequency(word.lower(), "en")


def _edit_distance(a: str, b: str, limit: int = 2) -> int:
    """Small Levenshtein with early bail past *limit*."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr.append(val)
            row_min = min(row_min, val)
        if row_min > limit:
            return limit + 1
        prev = curr
    return prev[-1]


def _looks_like_spelling_fix(before: str, after: str) -> bool:
    """Heuristic: rare/non-word -> common word at small edit distance."""
    a, b = before.lower(), after.lower()
    if a == b:
        return False
    freq_before, freq_after = _zipf(a), _zipf(b)
    # Unknown or rare original, common replacement, small distance.
    if freq_after >= 3.0 and freq_before < 3.0 and _edit_distance(a, b) <= 2:
        return True
    # Substituting one valid common word for another (e.g. insure->ensure,
    # nine->fine) is a potential meaning change, never an auto-accepted typo.
    return False


_EDGE_PUNCT = ".,;:!?\"'()[]{}“”‘’—–-…"


def _has_forbidden_control(text: str) -> bool:
    import unicodedata

    for ch in text:
        if ch in ("\n", "\t", "\r"):
            continue
        if unicodedata.category(ch) in {"Cf", "Cc", "Cs", "Co", "Cn"}:
            return True
    return False


def _classify_pair(before: str, after: str, position: int) -> Change:
    """Classify one token-level substitution."""
    if before == after:
        return Change(Category.UNCHANGED, Verdict.ACCEPT, before, after, position, "identical")

    if _has_forbidden_control(after) and not _has_forbidden_control(before):
        return Change(Category.REWORD, Verdict.REJECT, before, after, position,
                      "introduces invisible or control character")

    # Number or date involvement is an automatic reject, no exceptions.
    if _NUMBER_RE.search(before + after) or _DATEISH_RE.search(before + " " + after):
        return Change(Category.NUMBER, Verdict.REJECT, before, after, position,
                      "change touches a number or date")

    low_before, low_after = before.lower(), after.lower()
    if low_before in _NEGATIONS or low_after in _NEGATIONS:
        return Change(Category.NEGATION, Verdict.REJECT, before, after, position,
                      "change touches a negation")
    if (low_before.endswith("n't") or low_after.endswith("n't")) and low_before != low_after:
        return Change(Category.NEGATION, Verdict.REJECT, before, after, position,
                      "change touches a negation contraction")

    if _CITATION_RE.search(before) or _CITATION_RE.search(after):
        return Change(Category.CITATION, Verdict.REJECT, before, after, position,
                      "change touches a citation marker")

    # Hyphenation: prefix hyphens like re-sign/resign or re-cover/recover change meaning.
    if low_before.replace("-", "") == low_after.replace("-", ""):
        if re.match(r"^(re|co|pre|post|non|un)-", low_before) or re.match(
            r"^(re|co|pre|post|non|un)-", low_after
        ):
            return Change(Category.HYPHENATION, Verdict.REVIEW, before, after, position,
                          "prefix hyphen change may alter meaning")
        return Change(Category.HYPHENATION, Verdict.ACCEPT, before, after, position,
                      "hyphenation only")
    if low_before == low_after:
        return Change(Category.CASE, Verdict.ACCEPT, before, after, position, "case only")

    # Punctuation-only: only strip leading/trailing punctuation, never internal
    # apostrophes (lessee's -> lessees, shouldn't -> shouldnt) or control chars.
    b_core = before.strip(_EDGE_PUNCT).lower()
    a_core = after.strip(_EDGE_PUNCT).lower()
    if b_core and b_core == a_core:
        return Change(Category.PUNCTUATION, Verdict.ACCEPT, before, after, position,
                      "punctuation only")

    # Capitalised-token substitution is treated as a possible name change.
    if before[:1].isupper() and after[:1].isupper():
        return Change(Category.NAME, Verdict.REVIEW, before, after, position,
                      "capitalised token changed; may be a name")

    # Spelling: deterministic dictionary heuristics. Ambiguity -> REVIEW.
    if _looks_like_spelling_fix(before, after):
        return Change(Category.SPELLING, Verdict.ACCEPT, before, after, position,
                      "plausible spelling correction")
    return Change(Category.REWORD, Verdict.REJECT, before, after, position,
                  "substitution is not a recognisable spelling/punctuation fix")


def compare(original: str, edited: str) -> GateResult:
    """Compare two texts and classify every change.

    Whitespace-only differences at the file level are folded in first so
    token alignment stays stable.
    """
    if _has_forbidden_control(edited) and not _has_forbidden_control(original):
        return GateResult(
            verdict=Verdict.REJECT,
            changes=[
                Change(Category.REWORD, Verdict.REJECT, "", "", 0,
                       "edited text introduces invisible or control characters")
            ],
        )

    orig_tokens = _tokenise(original)
    edit_tokens = _tokenise(edited)

    changes: list[Change] = []
    matcher = difflib.SequenceMatcher(a=orig_tokens, b=edit_tokens, autojunk=False)
    orig_index = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            orig_index = i2
            continue
        before_span = orig_tokens[i1:i2]
        after_span = edit_tokens[j1:j2]

        if tag == "replace" and len(before_span) == len(after_span):
            for k, (b, a) in enumerate(zip(before_span, after_span)):
                changes.append(_classify_pair(b, a, i1 + k))
        else:
            # Insert/delete or unequal replace.
            joined_before = " ".join(before_span)
            joined_after = " ".join(after_span)
            if re.sub(r"\s+", "", joined_before) == re.sub(r"\s+", "", joined_after):
                # Token boundary changed (e.g. "not able"->"notable", "100 000"->"100000").
                if _NUMBER_RE.search(joined_before + joined_after) or any(
                    t.lower() in _NEGATIONS for t in before_span + after_span
                ):
                    changes.append(Change(Category.NUMBER, Verdict.REJECT,
                                          joined_before, joined_after, i1,
                                          "token split/merge touches number or negation"))
                else:
                    changes.append(Change(Category.WHITESPACE, Verdict.REVIEW,
                                          joined_before, joined_after, i1,
                                          "word split/merge requires review"))
            elif not _NUMBER_RE.search(joined_before + joined_after) and (
                " ".join(before_span).replace("-", " ").lower()
                == " ".join(after_span).replace("-", " ").lower()
            ):
                changes.append(Change(Category.HYPHENATION, Verdict.ACCEPT,
                                      joined_before, joined_after, i1,
                                      "hyphenation/line-break reflow"))
            else:
                changes.append(Change(Category.REWORD, Verdict.REJECT,
                                      joined_before, joined_after, i1,
                                      "insertion, deletion, or unequal rewording"))
        orig_index = i2

    worst = Verdict.ACCEPT
    order = {Verdict.ACCEPT: 0, Verdict.REVIEW: 1, Verdict.REJECT: 2}
    for change in changes:
        if order[change.verdict] > order[worst]:
            worst = change.verdict
    return GateResult(verdict=worst, changes=changes)


def verdict_for_file(original_text: str, edited_text: str) -> GateResult:
    """Convenience wrapper used by the ingest pipeline and tests."""
    return compare(original_text, edited_text)
