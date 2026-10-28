"""DOCX artefact cleaning must never alter wording or intentional hyphenation."""
from __future__ import annotations

import re

from tools.kb.converters import _clean_docx_artefacts


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text)


def test_matched_underline_becomes_bold():
    assert _clean_docx_artefacts("__Heading text__") == "**Heading text**"


def test_dangling_underline_removed():
    assert _clean_docx_artefacts("__dangling start only") == "dangling start only"


def test_backslash_escapes_removed():
    assert _clean_docx_artefacts(r"number\-\(5\)\-five\.") == "number-(5)-five."


def test_intra_word_hyphenation_preserved():
    text = "governs all move-ment of vessel on this plane-t, owner-ship, Corp-oration"
    assert _clean_docx_artefacts(text) == text


def test_word_stream_unchanged_on_realistic_passage():
    source = (
        r"__6\. Persons are also divided into citizens, \(q\. v\.\) and aliens\.__" "\n\n"
        "With this page eleven-(11)-of only eighty-(80)-page[s]."
    )
    cleaned = _clean_docx_artefacts(source)
    assert _words(cleaned) == _words(source)
    assert "__" not in cleaned
    assert "eleven-(11)-of" in cleaned
