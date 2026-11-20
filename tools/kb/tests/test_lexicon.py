from tools.kb import lexicon
from tools.kb.indexer import _fts_escape


def _write(tmp_path, text):
    path = tmp_path / "lexicon.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_no_lexicon_means_no_expansion(tmp_path):
    assert lexicon.expansions("good faith", tmp_path / "missing.yaml") == []


def test_variants_added_when_any_term_matches(tmp_path):
    path = _write(
        tmp_path,
        'terms:\n  - canonical: "good faith"\n    variants: ["honesty in fact"]\n',
    )
    assert lexicon.expansions("What is Good Faith?", path) == ["honesty in fact"]
    assert lexicon.expansions("honesty in fact", path) == ["good faith"]
    assert lexicon.expansions("goodfaith", path) == []


def test_terms_already_in_the_query_are_not_repeated(tmp_path):
    path = _write(tmp_path, 'terms:\n  - canonical: "a b"\n    variants: ["c d"]\n')
    assert lexicon.expansions("a b and c d", path) == []


def test_phrases_become_quoted_fts_phrases():
    expr = _fts_escape("good faith", extra_phrases=["honesty in fact", 'say "hi"'])
    assert '"good"' in expr and '"honesty in fact"' in expr
    assert '"say ""hi"""' in expr


def test_one_way_triggers_expand_misspellings_and_synonyms(tmp_path):
    trig = tmp_path / "triggers.yaml"
    trig.write_text(
        'triggers:\n  - user_phrases: ["sestui", "birth cert"]\n'
        '    corpus_phrases: ["cestui que vie", "certificate of an entry of a birth"]\n',
        encoding="utf-8",
    )
    assert lexicon.expansions("whats sestui about", path=tmp_path / "none.yaml", triggers_path=trig) == [
        "cestui que vie",
        "certificate of an entry of a birth",
    ]
    # One-way: corpus phrase alone does not trigger user phrases
    assert lexicon.expansions("cestui que vie", path=tmp_path / "none.yaml", triggers_path=trig) == []

