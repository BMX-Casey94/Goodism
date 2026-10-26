"""Chunker tests: structure preserved, tables/lists never split."""
from __future__ import annotations

from tools.kb.chunker import chunk_document


def test_heading_sections_become_chunks():
    body = (
        "# Part One\n\nFirst section text.\n\n"
        "# Part Two\n\nSecond section text.\n"
    )
    chunks = chunk_document("doc", "Title", body)
    paths = [c.heading_path for c in chunks]
    assert any("Part One" in p for p in paths)
    assert any("Part Two" in p for p in paths)


def test_no_headings_single_chunk():
    body = "Just a paragraph of prose without headings."
    chunks = chunk_document("doc", "Title", body)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Title"


def test_tables_never_split():
    rows = "\n".join(f"| col{i} | value{i} |" for i in range(40))
    body = f"# Section\n\nIntro.\n\n{rows}\n\nAfter table."
    chunks = chunk_document("doc", "Title", body)
    for chunk in chunks:
        if "| col0 |" in chunk.text:
            assert "| col39 |" in chunk.text, "table was split across chunks"


def test_lists_never_split():
    items = "\n".join(f"- item number {i} with some text" for i in range(30))
    body = f"# Section\n\nLead in.\n\n{items}\n\nLead out."
    chunks = chunk_document("doc", "Title", body)
    for chunk in chunks:
        if "- item number 0 " in chunk.text:
            assert "- item number 29 " in chunk.text, "list was split across chunks"


def test_contextual_prefix_present_not_in_text():
    body = "# Equity\n\nSome text about trusts."
    chunks = chunk_document("doc", "My Title", body)
    assert chunks[0].display.startswith("My Title — ")
    assert "My Title" not in chunks[0].text  # verbatim text stays clean


def test_offsets_point_at_verbatim_text():
    body = "# A\n\nalpha beta gamma.\n\n# B\n\ndelta epsilon."
    chunks = chunk_document("doc", "T", body)
    for chunk in chunks:
        assert body[chunk.start : chunk.end].strip("\n") == chunk.text
