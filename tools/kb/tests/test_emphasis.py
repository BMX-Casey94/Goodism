import re

from tools.kb.converters import repair_emphasis_line as fix

WORDS = re.compile(r"[A-Za-z0-9]+")


def _same_words(a: str, b: str) -> bool:
    return WORDS.findall(a) == WORDS.findall(b)


def test_space_before_closing_moves_outside():
    assert fix("**PART 1. GENERAL **") == "**PART 1. GENERAL**"
    assert fix("**one ** two") == "**one** two"


def test_space_after_opening_moves_outside():
    assert fix("say ** word** now") == "say **word** now"


def test_nbsp_handled_and_no_leading_indent_added():
    assert fix("**\u00a0Bouvier\u2019s Law**") == "**Bouvier\u2019s Law**"


def test_four_star_join_removed():
    assert fix("**For the****-\u201cdisclosure**") == "**For the-\u201cdisclosure**"


def test_whitespace_only_span_loses_markers():
    assert fix("** **") == ""


def test_odd_marker_count_left_alone():
    assert fix("**open only ") == "**open only "


def test_words_never_change():
    for s in ["a ** b ** c", "**x** **y **", "** lone** and **pair **"]:
        assert _same_words(s, fix(s))
