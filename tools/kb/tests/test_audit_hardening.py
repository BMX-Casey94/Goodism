"""Regression tests for the audit hardening work (H5, lexicon validation, DOCX guard).

Three independent concerns:

- ``server/mcp_server.py`` tools must be async and offload blocking work to a
  worker thread, bounded by a semaphore that survives repeated event loops.
- ``tools/kb/validate.py`` must catch malformed query-expansion vocabulary.
- ``tools/kb/converters.py`` must reject zip-bomb DOCX files and stream scanned
  PDF pages instead of buffering every bitmap.
"""
from __future__ import annotations

import inspect
import json
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
import yaml

from server import mcp_server
from tools.kb import converters, lexicon, validate

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# DOCX archive guard
# ---------------------------------------------------------------------------


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_docx_guard_accepts_a_small_well_formed_archive(tmp_path):
    docx = _write_zip(
        tmp_path / "small.docx",
        {
            "[Content_Types].xml": b"<Types/>",
            "word/document.xml": b"<w:document>hello</w:document>",
        },
    )
    converters._check_docx_archive(docx)  # must not raise


def test_docx_guard_rejects_excessive_entry_count(tmp_path, monkeypatch):
    monkeypatch.setattr(converters, "DOCX_MAX_ENTRIES", 2)
    docx = _write_zip(
        tmp_path / "many.docx",
        {f"word/part{i}.xml": b"<x/>" for i in range(5)},
    )
    with pytest.raises(ValueError, match="entries"):
        converters._check_docx_archive(docx)


def test_docx_guard_rejects_excessive_uncompressed_size(tmp_path, monkeypatch):
    monkeypatch.setattr(converters, "DOCX_MAX_UNCOMPRESSED_BYTES", 1024)
    monkeypatch.setattr(converters, "DOCX_MAX_COMPRESSION_RATIO", 10_000.0)
    docx = _write_zip(tmp_path / "big.docx", {"word/document.xml": b"a" * 4096})
    with pytest.raises(ValueError, match="uncompressed"):
        converters._check_docx_archive(docx)


def test_docx_guard_rejects_highly_compressible_member(tmp_path):
    # 2 MiB of zeros deflates to a couple of kilobytes: a classic bomb member.
    docx = _write_zip(
        tmp_path / "zeros.docx",
        {"word/document.xml": b"\x00" * (2 * 1024 * 1024)},
    )
    with pytest.raises(ValueError, match="compression ratio"):
        converters._check_docx_archive(docx)


def test_docx_guard_rejects_member_without_compressed_size(tmp_path, monkeypatch):
    # A member claiming a large uncompressed size but recording no compressed
    # size cannot be ratio-checked, so it is treated as a bomb outright. A real
    # archive cannot be built in this shape, so the member list is stubbed.
    docx = tmp_path / "zerosize.docx"
    docx.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    class _Info:
        filename = "word/document.xml"
        file_size = 4 * 1024 * 1024
        compress_size = 0

    class _Archive:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def infolist(self):
            return [_Info()]

    monkeypatch.setattr(
        converters, "zipfile", SimpleNamespace(ZipFile=_Archive, BadZipFile=zipfile.BadZipFile)
    )
    with pytest.raises(ValueError, match="no compressed size"):
        converters._check_docx_archive(docx)


@pytest.mark.parametrize(
    "member", ["../evil.xml", "word/../../evil.xml", "/etc/passwd", "\\windows\\evil.xml", "C:/evil.xml"]
)
def test_docx_guard_rejects_path_traversal_and_absolute_members(tmp_path, member):
    docx = _write_zip(tmp_path / "slip.docx", {member: b"<x/>"})
    with pytest.raises(ValueError, match="unsafe member"):
        converters._check_docx_archive(docx)


def test_docx_guard_rejects_non_zip_input(tmp_path):
    bogus = tmp_path / "not-a-docx.docx"
    bogus.write_bytes(b"this is plainly not a zip archive")
    with pytest.raises(ValueError, match="not a valid DOCX"):
        converters._check_docx_archive(bogus)


def test_convert_docx_checks_the_archive_before_parsing(tmp_path, monkeypatch):
    def _boom(_path):
        raise ValueError("guard ran")

    monkeypatch.setattr(converters, "_check_docx_archive", _boom)
    with pytest.raises(ValueError, match="guard ran"):
        converters.convert_docx(tmp_path / "anything.docx")


# ---------------------------------------------------------------------------
# Streaming PDF page rendering
# ---------------------------------------------------------------------------


def test_pdf_page_images_is_a_generator():
    assert inspect.isgeneratorfunction(converters.pdf_page_images)
    produced = converters.pdf_page_images(Path("unused.pdf"))
    assert inspect.isgenerator(produced)
    produced.close()


def test_pdf_page_images_yields_lazily_page_by_page(monkeypatch):
    import pymupdf

    rendered: list[int] = []

    class _FakePixmap:
        def __init__(self, page_no: int) -> None:
            self.page_no = page_no

        def tobytes(self, fmt: str) -> bytes:
            assert fmt == "png"
            return f"page-{self.page_no}".encode()

    class _FakePage:
        def __init__(self, page_no: int) -> None:
            self.page_no = page_no

        def get_pixmap(self, dpi: int) -> _FakePixmap:
            assert dpi == 150
            rendered.append(self.page_no)
            return _FakePixmap(self.page_no)

    class _FakeDoc:
        def __init__(self) -> None:
            self.pages = [_FakePage(1), _FakePage(2), _FakePage(3)]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            return iter(self.pages)

    monkeypatch.setattr(pymupdf, "open", lambda _path: _FakeDoc())

    stream = converters.pdf_page_images(Path("scan.pdf"), dpi=150)
    assert rendered == []  # nothing rendered until the first pull
    assert next(stream) == b"page-1"
    assert rendered == [1]  # exactly one page materialised
    assert list(stream) == [b"page-2", b"page-3"]
    assert rendered == [1, 2, 3]


# ---------------------------------------------------------------------------
# Lexicon and triggers validation
# ---------------------------------------------------------------------------


def _point_lexicon_at(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    lex = tmp_path / "lexicon.yaml"
    trig = tmp_path / "triggers.yaml"
    monkeypatch.setattr(lexicon, "LEXICON_PATH", lex)
    monkeypatch.setattr(lexicon, "TRIGGERS_PATH", trig)
    return lex, trig


def test_validate_lexicon_passes_on_the_real_schema_files():
    assert lexicon.LEXICON_PATH.is_file()
    assert lexicon.TRIGGERS_PATH.is_file()
    report = validate.ValidationReport()
    validate.validate_lexicon(report)
    assert report.errors == [], report.errors
    assert report.warnings == []


def test_validate_lexicon_passes_on_well_formed_fixtures(tmp_path, monkeypatch):
    lex, trig = _point_lexicon_at(monkeypatch, tmp_path)
    lex.write_text(
        'terms:\n  - canonical: "good faith"\n    variants: ["honesty in fact"]\n',
        encoding="utf-8",
    )
    trig.write_text(
        "triggers:\n  - user_phrases: [\"sestui\"]\n"
        '    corpus_phrases: ["cestui que vie"]\n',
        encoding="utf-8",
    )
    report = validate.ValidationReport()
    validate.validate_lexicon(report)
    assert report.errors == [], report.errors


def test_validate_lexicon_is_a_no_op_when_files_are_absent(tmp_path, monkeypatch):
    _point_lexicon_at(monkeypatch, tmp_path)
    report = validate.ValidationReport()
    validate.validate_lexicon(report)
    assert report.errors == []


def test_validate_lexicon_catches_duplicate_canonical_case_insensitively(tmp_path, monkeypatch):
    lex, _ = _point_lexicon_at(monkeypatch, tmp_path)
    lex.write_text(
        'terms:\n'
        '  - canonical: "Good Faith"\n    variants: ["honesty in fact"]\n'
        '  - canonical: "good faith"\n    variants: ["fair dealing"]\n',
        encoding="utf-8",
    )
    report = validate.ValidationReport()
    validate.validate_lexicon(report)
    assert any("duplicate canonical" in e for e in report.errors)


@pytest.mark.parametrize(
    "body, expected",
    [
        ('terms:\n  - canonical: "good faith"\n    variants: []\n', "non-empty list of strings"),
        ('terms:\n  - canonical: ""\n    variants: ["x"]\n', "'canonical' must be a non-empty string"),
        ('terms:\n  - canonical: "good faith"\n    variants: [""]\n', "non-empty string"),
        ('terms:\n  - canonical: "good faith"\n', "'variants' must be a non-empty list"),
        ("terms: not-a-list\n", "'terms' must be a list"),
        ("terms:\n  - just a string\n", "must be a mapping"),
        ("- a\n- b\n", "must be a YAML mapping"),
        ("", "is empty"),
    ],
)
def test_validate_lexicon_rejects_malformed_terms(tmp_path, monkeypatch, body, expected):
    lex, _ = _point_lexicon_at(monkeypatch, tmp_path)
    lex.write_text(body, encoding="utf-8")
    report = validate.ValidationReport()
    validate.validate_lexicon(report)
    assert any(expected in e for e in report.errors), report.errors


def test_validate_lexicon_catches_invalid_yaml(tmp_path, monkeypatch):
    lex, _ = _point_lexicon_at(monkeypatch, tmp_path)
    lex.write_text('terms: [\n  - "unclosed\n', encoding="utf-8")
    report = validate.ValidationReport()
    validate.validate_lexicon(report)
    assert any("invalid YAML" in e for e in report.errors)


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            "triggers:\n  - user_phrases: []\n    corpus_phrases: [\"x\"]\n",
            "'user_phrases' must be a non-empty list",
        ),
        (
            "triggers:\n  - user_phrases: [\"x\"]\n    corpus_phrases: []\n",
            "'corpus_phrases' must be a non-empty list",
        ),
        (
            "triggers:\n  - user_phrases: [\"x\"]\n",
            "'corpus_phrases' must be a non-empty list",
        ),
        (
            "triggers:\n  - user_phrases: [\"\"]\n    corpus_phrases: [\"x\"]\n",
            "non-empty string",
        ),
        ("triggers: nope\n", "'triggers' must be a list"),
    ],
)
def test_validate_lexicon_rejects_malformed_triggers(tmp_path, monkeypatch, body, expected):
    _, trig = _point_lexicon_at(monkeypatch, tmp_path)
    trig.write_text(body, encoding="utf-8")
    report = validate.ValidationReport()
    validate.validate_lexicon(report)
    assert any(expected in e for e in report.errors), report.errors


def test_run_validation_includes_lexicon_checks(tmp_path, monkeypatch):
    lex, trig = _point_lexicon_at(monkeypatch, tmp_path)
    lex.write_text("terms: not-a-list\n", encoding="utf-8")
    trig.write_text("triggers: []\n", encoding="utf-8")
    monkeypatch.setattr(validate, "validate_corpus", lambda report: {})
    monkeypatch.setattr(validate, "validate_manifest", lambda report, hashes: None)
    report = validate.run_validation()
    assert any("'terms' must be a list" in e for e in report.errors)


# ---------------------------------------------------------------------------
# MCP server async offloading
# ---------------------------------------------------------------------------


def test_every_mcp_tool_is_async():
    tools = mcp_server.mcp._tool_manager.list_tools()
    assert len(tools) == 10
    for tool in tools:
        assert inspect.iscoroutinefunction(tool.fn), f"{tool.name} is still synchronous"


def test_read_around_and_index_status_survive_repeated_event_loops():
    for _ in range(3):
        res = json.loads(anyio.run(mcp_server.read_around, "complete-ucc-1-9", 10**30))
        assert "error" in res

        status = json.loads(anyio.run(mcp_server.get_index_status))
        assert "error" not in status
        assert "pinned_snapshot" in status


def test_blocking_work_runs_off_the_event_loop_thread():
    main_thread = threading.current_thread().ident
    seen: list[int | None] = []

    def _record() -> str:
        seen.append(threading.current_thread().ident)
        return "ok"

    assert anyio.run(mcp_server._run_blocking, _record) == "ok"
    assert seen and seen[0] != main_thread


def test_blocking_work_propagates_exceptions():
    def _fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        anyio.run(mcp_server._run_blocking, _fail)


def test_semaphore_caps_actual_concurrency(monkeypatch):
    monkeypatch.setenv("KB_MCP_MAX_CONCURRENCY", "2")
    active = 0
    peak = 0

    def _work() -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.05)
        active -= 1

    async def _burst() -> None:
        async with anyio.create_task_group() as tg:
            for _ in range(8):
                tg.start_soon(mcp_server._run_blocking, _work)

    anyio.run(_burst)
    assert peak == 2


@pytest.mark.parametrize(
    "raw, expected",
    [("", 4), ("1", 1), ("8", 8), ("32", 32), ("999", 32), ("0", 1), ("-5", 1), ("junk", 4)],
)
def test_max_concurrency_is_clamped(monkeypatch, raw, expected):
    monkeypatch.setenv("KB_MCP_MAX_CONCURRENCY", raw)
    assert mcp_server._max_concurrency() == expected


def test_semaphore_is_reused_within_a_loop_and_rebuilt_per_loop():
    async def _twice():
        return mcp_server._loop_semaphore(), mcp_server._loop_semaphore()

    first_a, first_b = anyio.run(_twice)
    second_a, second_b = anyio.run(_twice)
    assert first_a is first_b  # one semaphore per loop
    assert first_a is not second_a  # a new loop gets a fresh semaphore
    assert second_a is second_b


def test_research_tool_awaits_through_the_offload_path(monkeypatch):
    calls: list[tuple[str, int, int, str | None]] = []

    def _fake_research(queries, context, max_hits, domain):
        calls.append((queries, context, max_hits, domain))
        return json.dumps({"ok": True})

    monkeypatch.setattr(mcp_server, "_research", _fake_research)
    out = json.loads(anyio.run(mcp_server.research, ["good faith"]))
    assert out == {"ok": True}
    assert calls == [(["good faith"], 1, 16, None)]


# ---------------------------------------------------------------------------
# Environment sanity: the fixtures above must not depend on a broken schema
# ---------------------------------------------------------------------------


def test_schema_files_parse_as_mappings():
    for path in (REPO_ROOT / "schema" / "lexicon.yaml", REPO_ROOT / "schema" / "triggers.yaml"):
        assert path.is_file(), path
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
