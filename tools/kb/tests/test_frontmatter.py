"""Front matter validation tests."""
from __future__ import annotations

import pytest

from tools.kb.frontmatter import (
    FrontMatterError,
    split_front_matter,
    validate_meta,
)

GOOD = {
    "id": "test-doc",
    "title": "Test",
    "domain": "trusts-equity",
    "topics": ["trust-law"],
    "doc_type": "commentary",
    "source_file": "inbox/x.docx",
    "source_sha256": "a" * 64,
    "authority": "supporting",
    "precedence": 50,
    "status": "draft",
    "extraction_method": "native-text",
    "summary": "A test document.",
}


def test_valid_meta_passes():
    meta = validate_meta(dict(GOOD), check_vocab=True)
    assert meta.id == "test-doc"


def test_bad_id_rejected():
    data = dict(GOOD, id="Bad ID!")
    with pytest.raises(FrontMatterError):
        validate_meta(data, check_vocab=False)


def test_unknown_domain_rejected():
    data = dict(GOOD, domain="not-a-domain")
    with pytest.raises(FrontMatterError, match="unknown domain"):
        validate_meta(data, check_vocab=True)


def test_unknown_topic_rejected():
    data = dict(GOOD, topics=["not-a-topic"])
    with pytest.raises(FrontMatterError, match="unknown topics"):
        validate_meta(data, check_vocab=True)


def test_ocr_requires_confidence():
    data = dict(GOOD, extraction_method="ocr", ocr_confidence=None)
    with pytest.raises(FrontMatterError, match="ocr_confidence"):
        validate_meta(data, check_vocab=False)


def test_bad_hash_rejected():
    data = dict(GOOD, source_sha256="zz")
    with pytest.raises(FrontMatterError):
        validate_meta(data, check_vocab=False)


def test_split_front_matter():
    text = "---\nid: a\ntitle: T\n---\n\nBody here.\n"
    data, body = split_front_matter(text)
    assert data["id"] == "a"
    assert body.strip() == "Body here."


def test_split_front_matter_missing_fence():
    with pytest.raises(FrontMatterError):
        split_front_matter("no fence here")
