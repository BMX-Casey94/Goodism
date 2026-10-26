"""Integrity gate tests: the gate must never let meaning change through."""
from __future__ import annotations

from tools.kb.integrity import Category, Verdict, compare


def test_identical_text_accepted():
    result = compare("The trust endures.", "The trust endures.")
    assert result.verdict == Verdict.ACCEPT
    assert not result.changes


def test_spelling_fix_accepted():
    result = compare("The governement holds title.", "The government holds title.")
    assert result.verdict == Verdict.ACCEPT
    assert any(c.category == Category.SPELLING for c in result.changes)


def test_punctuation_only_accepted():
    result = compare("Wait; the rule applies", "Wait, the rule applies.")
    assert result.verdict == Verdict.ACCEPT


def test_number_change_rejected():
    result = compare("see section 9", "see section 10")
    assert result.verdict == Verdict.REJECT
    assert any(c.category == Category.NUMBER for c in result.changes)


def test_negation_change_rejected():
    result = compare("The trustee may act.", "The trustee may not act.")
    assert result.verdict == Verdict.REJECT


def test_rewording_rejected():
    result = compare(
        "The beneficiary holds equitable title.",
        "The beneficiary owns the property outright.",
    )
    assert result.verdict == Verdict.REJECT


def test_hyphenation_accepted():
    result = compare("a well-known principle", "a well known principle")
    assert result.verdict == Verdict.ACCEPT


def test_case_only_accepted():
    result = compare("the Act applies", "the act applies")
    assert result.verdict == Verdict.ACCEPT


def test_whitespace_reflow_accepted():
    result = compare("word1 word2 word3", "word1  word2   word3")
    assert result.verdict == Verdict.ACCEPT


def test_capitalised_substitution_goes_to_review():
    result = compare("Blackstone wrote this.", "Blackwell wrote this.")
    assert result.verdict in {Verdict.REVIEW, Verdict.REJECT}
    assert result.verdict != Verdict.ACCEPT
