"""Structure-aware chunking with parent-child ranges.

Rules (from the plan):
- Split on markdown headings; each chunk carries its full heading path.
- Target 300-800 tokens per chunk; never split mid-table or mid-list.
- Parent-child: small child chunks for search precision, each pointing at
  its parent section (heading block) which is returned for context.
- Chunk text is the verbatim corpus text; contextual prefixes (title +
  heading path) are prepended at index time only and are never written
  back to the corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


@dataclass
class Chunk:
    doc_id: str
    heading_path: str  # "Equity / Section 9"
    text: str  # verbatim corpus text for this chunk
    start: int  # char offset in the body
    end: int
    level: int
    ordinal: int  # chunk order within the document
    parent_ordinal: int | None  # parent section chunk ordinal, None for roots
    display: str = ""  # text + contextual prefix, used for embedding/FTS


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // config.CHARS_PER_TOKEN)


@dataclass
class _Block:
    kind: str  # "para" | "table" | "list" | "code"
    text: str
    start: int
    end: int


def _split_blocks(section_text: str, section_start: int) -> list[_Block]:
    """Split a section into atomic blocks, keeping tables/lists/code intact."""
    lines = section_text.split("\n")
    blocks: list[_Block] = []
    current: list[str] = []
    current_kind = "para"
    pos = section_start
    block_start = pos

    def flush(end_pos: int) -> None:
        nonlocal current, block_start, current_kind
        if current:
            blocks.append(_Block(current_kind, "\n".join(current), block_start, end_pos))
            current = []

    in_code = False
    for line in lines:
        line_end = pos + len(line)
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                current.append(line)
                flush(line_end)
                current_kind = "para"
                in_code = False
            else:
                flush(pos - 1)
                current_kind = "code"
                block_start = pos
                current.append(line)
                in_code = True
            pos = line_end + 1
            continue
        if in_code:
            current.append(line)
            pos = line_end + 1
            continue

        if not stripped:
            flush(pos - 1)
            current_kind = "para"
        elif _TABLE_ROW_RE.match(line):
            if current_kind != "table":
                flush(pos - 1)
                current_kind = "table"
                block_start = pos
            current.append(line)
        elif _LIST_ITEM_RE.match(line):
            if current_kind not in {"list"}:
                flush(pos - 1)
                current_kind = "list"
                block_start = pos
            current.append(line)
        else:
            if current_kind in {"table", "list"}:
                flush(pos - 1)
                current_kind = "para"
            if not current:
                block_start = pos
            current.append(line)
        pos = line_end + 1
    flush(pos - 1)
    return blocks


def chunk_document(doc_id: str, title: str, body: str) -> list[Chunk]:
    """Chunk one document body. Offsets are relative to *body*."""
    lines = body.split("\n")
    # Find heading line positions.
    headings: list[tuple[int, int, str]] = []  # (line_index, level, text)
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            headings.append((i, len(match.group(1)), match.group(2).strip()))

    # Build section ranges (heading line -> before next heading of any level).
    sections: list[tuple[int, int, int, str]] = []  # (start_line, end_line, level, text)
    if not headings:
        sections.append((0, len(lines), 1, title))
    else:
        if headings[0][0] > 0:
            sections.append((0, headings[0][0], 1, title))
        for idx, (line_no, level, text) in enumerate(headings):
            end_line = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
            sections.append((line_no, end_line, level, text))

    # Compute char offset of each line start.
    line_starts: list[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    # Maintain the heading-path stack for contextual prefixes.
    path_stack: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    ordinal = 0
    parent_for_level: dict[int, int] = {}

    for start_line, end_line, level, heading_text in sections:
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, heading_text))
        heading_path = " / ".join(t for _, t in path_stack)

        start_char = line_starts[start_line]
        end_char = line_starts[end_line - 1] + len(lines[end_line - 1]) if end_line > start_line else start_char
        section_text = body[start_char:end_char].strip("\n")

        parent_ordinal = parent_for_level.get(level - 1)

        # Section fits in one chunk?
        if _estimate_tokens(section_text) <= config.CHUNK_MAX_TOKENS:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    heading_path=heading_path,
                    text=section_text,
                    start=start_char,
                    end=end_char,
                    level=level,
                    ordinal=ordinal,
                    parent_ordinal=parent_ordinal,
                )
            )
            parent_for_level[level] = ordinal
            ordinal += 1
            continue

        # Split by atomic blocks, packing into token-bounded chunks.
        blocks = _split_blocks(section_text, start_char)
        buffer: list[_Block] = []
        buffer_tokens = 0
        parent_ordinal_of_run = parent_ordinal

        def flush_buffer() -> None:
            nonlocal buffer, buffer_tokens, ordinal, parent_ordinal_of_run
            if not buffer:
                return
            text = "\n\n".join(b.text for b in buffer)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    heading_path=heading_path,
                    text=text,
                    start=buffer[0].start,
                    end=buffer[-1].end,
                    level=level,
                    ordinal=ordinal,
                    parent_ordinal=parent_ordinal_of_run,
                )
            )
            parent_for_level[level] = ordinal
            parent_ordinal_of_run = ordinal  # continuation chunks chain to the first
            ordinal += 1
            buffer = []
            buffer_tokens = 0

        for block in blocks:
            block_tokens = _estimate_tokens(block.text)
            if buffer and buffer_tokens + block_tokens > config.CHUNK_MAX_TOKENS:
                flush_buffer()
            buffer.append(block)
            buffer_tokens += block_tokens
        flush_buffer()

    # Contextual prefix (index-time only; never written to the corpus).
    for chunk in chunks:
        chunk.display = f"{title} — {chunk.heading_path}\n\n{chunk.text}"
    return chunks
