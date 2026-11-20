"""Format converters: DOCX, PDF, TXT, and images -> text/markdown.

Text-native formats are converted locally and deterministically. Images
and scanned PDFs go through a vision model (two independent passes; the
pipeline compares them). Converters return plain markdown-ish text with
headings preserved; they never "improve" the content.
"""
from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"}

# A DOCX is a ZIP, so an untrusted file can be a decompression bomb: a few
# kilobytes on disk that expands to gigabytes of XML once mammoth reads it.
# These limits bound the work done before any parsing starts.
DOCX_MAX_ENTRIES = 2000
DOCX_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024  # 100 MiB
DOCX_MAX_COMPRESSION_RATIO = 100.0
# The ratio check only applies to members large enough for a ratio to mean
# anything; small XML parts compress enormously without being a threat.
_DOCX_RATIO_MIN_BYTES = 1_048_576


@dataclass
class Conversion:
    text: str
    extraction_method: str  # native-text | ocr | vision
    confidence: float | None  # None for native text
    notes: str = ""


def convert_txt(path: Path) -> Conversion:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Conversion(text=text, extraction_method="native-text", confidence=None)


# Mammoth escapes Markdown special characters with backslashes, which some
# viewers render literally. These escapes are conversion artefacts, not
# content: removing the backslash renders identically. Intra-word hyphens
# (e.g. "plane-t") and all wording are left strictly alone.
_BACKSLASH_ESCAPES = {
    "\\-": "-",
    "\\(": "(",
    "\\)": ")",
    "\\[": "[",
    "\\]": "]",
    "\\.": ".",
    "\\!": "!",
    "\\#": "#",
    "\\+": "+",
    "\\_": "_",
}

# Empty bookmark anchors mammoth emits for Word bookmarks, e.g. <a id="_Toc1"></a>.
# They contain no words and render as noise in some viewers.
_EMPTY_ANCHOR_RE = re.compile(r'<a id="[^"]*"></a>')


_WS = " \t\u00a0"


def repair_emphasis_line(line: str) -> str:
    """Make ``**bold**`` spans valid CommonMark without touching any word.

    CommonMark ignores a bold span if whitespace follows the opening ``**`` or
    precedes the closing ``**``, so the asterisks print literally. Word keeps
    such whitespace inside the run. This moves the whitespace outside the
    markers, merges ``****`` joins, and drops markers around whitespace-only
    spans. Lines with an odd number of ``**`` are left alone.
    """
    if "**" not in line:
        return line
    line = line.replace("****", "")
    if line.count("**") % 2:
        return line
    parts = line.split("**")
    res: list[str] = []
    carry = ""
    for i, piece in enumerate(parts):
        if i % 2 == 0:
            if carry and piece[:1] in tuple(_WS):
                carry = ""  # the following text already starts with a space
            res.append(carry + piece)
            carry = ""
            continue
        core = piece.strip(_WS)
        if not core:
            res.append(piece)  # whitespace-only span: drop the markers
            continue
        lead = piece[: len(piece) - len(piece.lstrip(_WS))]
        trail = piece[len(piece.rstrip(_WS)):]
        before = "".join(res)
        if lead and before and before[-1] not in _WS:
            res.append(lead)  # never add indentation at the start of a line
        res.append("**" + core + "**")
        carry = trail
    out = "".join(res)
    if not line.endswith(tuple(_WS)):
        out = out.rstrip(_WS)
    return out


def repair_emphasis(text: str) -> str:
    return "\n".join(repair_emphasis_line(l) for l in text.split("\n"))


def _clean_docx_artefacts(text: str) -> str:
    """Remove DOCX-conversion artefacts without touching wording.

    - Unescape mammoth's over-escaped punctuation (renders identically).
    - Convert matched __text__ underline pairs to **text** (underline is not
      Markdown; __ reads as bold). Dangling __ markers (no closing pair) are
      dropped because they render as literal underscores.
    - Intra-word hyphens, numbers and every word are preserved verbatim.
    """
    for esc, plain in _BACKSLASH_ESCAPES.items():
        text = text.replace(esc, plain)

    text = _EMPTY_ANCHOR_RE.sub("", text)
    text = repair_emphasis(text)

    # Matched underline pairs first.
    text = re.sub(r"__(.+?)__", r"**\1**", text)
    # Remaining dangling __ (line-initial emphasis mammoth could not close).
    text = text.replace("__", "")
    return text


def _docx_member_is_safe(name: str) -> bool:
    """Reject absolute member names and any ``..`` component (zip-slip)."""
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:", name):  # Windows drive-qualified name
        return False
    parts = name.replace("\\", "/").split("/")
    return ".." not in parts


def _check_docx_archive(path: Path) -> None:
    """Refuse DOCX files that could exhaust memory or escape the archive root.

    Called before mammoth opens the file: mammoth expands the whole package, so
    a zip bomb would already be in memory by the time its own limits apply.
    """
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name}: not a valid DOCX/ZIP archive: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{path.name}: cannot open DOCX archive: {exc}") from exc

    if len(infos) > DOCX_MAX_ENTRIES:
        raise ValueError(
            f"{path.name}: DOCX has {len(infos)} entries, exceeding the limit of "
            f"{DOCX_MAX_ENTRIES}"
        )

    total_uncompressed = 0
    for info in infos:
        if not _docx_member_is_safe(info.filename):
            raise ValueError(
                f"{path.name}: DOCX contains an unsafe member name: {info.filename!r}"
            )
        total_uncompressed += info.file_size
        if total_uncompressed > DOCX_MAX_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"{path.name}: DOCX expands to more than "
                f"{DOCX_MAX_UNCOMPRESSED_BYTES} bytes uncompressed"
            )
        if info.file_size <= _DOCX_RATIO_MIN_BYTES:
            continue
        if info.compress_size == 0:
            raise ValueError(
                f"{path.name}: DOCX member {info.filename!r} is {info.file_size} bytes "
                "with no compressed size (zip bomb)"
            )
        ratio = info.file_size / info.compress_size
        if ratio > DOCX_MAX_COMPRESSION_RATIO:
            raise ValueError(
                f"{path.name}: DOCX member {info.filename!r} has compression ratio "
                f"{ratio:.0f}:1, exceeding the limit of {DOCX_MAX_COMPRESSION_RATIO:.0f}:1"
            )


def convert_docx(path: Path) -> Conversion:
    """DOCX -> markdown preserving headings, lists and tables via mammoth.

    Conversion artefacts (backslash escapes, dangling underline markers) are
    cleaned; the author's wording, hyphenation and punctuation choices are
    preserved verbatim.
    """
    _check_docx_archive(path)
    try:
        import mammoth
    except ImportError as exc:
        raise RuntimeError(
            "mammoth is required for DOCX conversion: pip install mammoth"
        ) from exc
    with path.open("rb") as handle:
        result = mammoth.convert_to_markdown(handle)
    cleaned = _clean_docx_artefacts(result.value)
    notes = "; ".join(m.message for m in result.messages) if result.messages else ""
    return Conversion(
        text=cleaned, extraction_method="native-text", confidence=None, notes=notes
    )


def convert_pdf(path: Path) -> Conversion:
    """PDF -> text. Uses pdfplumber for layout-faithful extraction.

    Pages with no extractable text indicate a scan; the caller should route
    those through the vision path instead.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required for PDF conversion: pip install pdfplumber"
        ) from exc

    pages: list[str] = []
    empty_pages = 0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not text.strip():
                empty_pages += 1
            pages.append(text)
    body = "\n\n".join(p for p in pages if p.strip())
    notes = ""
    if empty_pages:
        notes = f"{empty_pages} page(s) had no extractable text (likely scans)"
    return Conversion(text=body, extraction_method="native-text", confidence=None, notes=notes)


# Shared transcription instruction. main_text_only drops running heads,
# footers, page numbers and reference footnotes at transcription time so
# they never reach the corpus (per project decision: only the main text is
# kept from scanned pages).
def _transcription_system(main_text_only: bool) -> str:
    base = (
        "You are a meticulous transcription engine. Transcribe the document image "
        "VERBATIM: preserve the original spelling (including errors), punctuation, "
        "capitalisation, hyphenation and line structure. Do not correct, modernise, "
        "summarise or comment. Mark any illegible word as [illegible]."
    )
    if main_text_only:
        base += (
            " Transcribe ONLY the main body text of the page. EXCLUDE running heads, "
            "page headers and footers, page numbers, and reference footnotes/endnotes "
            "or citation footers. Do not transcribe those marginal regions at all. "
            "If you are unsure whether a region is body text or a footer/reference, "
            "include it and wrap it in [marginal]...[/marginal] so a human can decide."
        )
    return base + " Output only the transcription."


def convert_image_vision(
    path: Path, client, *, role: str = "extractor", main_text_only: bool = True
) -> Conversion:
    """Image -> text via a vision model. One pass; the pipeline runs two.

    The model is instructed to transcribe verbatim, preserving spelling,
    punctuation and layout, and to mark illegible spans.
    """
    from .llm import LLMClient

    if not isinstance(client, LLMClient):
        raise TypeError("convert_image_vision requires an LLMClient")

    system = _transcription_system(main_text_only)
    text = client.chat(
        role=role,
        system=system,
        user="Transcribe this document page verbatim.",
        images=[path.read_bytes()],
        max_tokens=8192,
    )
    # Confidence is coarse: presence of [illegible] marks lowers it. The
    # pipeline refines this by comparing two independent passes.
    illegible = text.count("[illegible]")
    words = max(len(text.split()), 1)
    confidence = max(0.5, 1.0 - (illegible / words) * 10)
    return Conversion(
        text=text,
        extraction_method="vision",
        confidence=round(confidence, 3),
        notes=f"vision pass, {illegible} illegible mark(s)",
    )


def pdf_page_images(path: Path, dpi: int = 200) -> Iterator[bytes]:
    """Render a PDF's pages to PNG bytes, one page at a time.

    Yields rather than returning a list: a 400-page scan at 200 dpi is well
    over a gigabyte of bitmaps, which must never be held in memory at once.
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to render scanned PDFs: pip install PyMuPDF"
        ) from exc
    with pymupdf.open(path) as doc:
        for page in doc:
            pixmap = page.get_pixmap(dpi=dpi)
            yield pixmap.tobytes("png")


def _count_marginal(text: str) -> int:
    return text.count("[marginal]")


def convert_pdf_scanned_vision(
    path: Path, client, *, role: str = "extractor", main_text_only: bool = True
) -> Conversion:
    """Scanned PDF -> text by rendering each page and running the vision path.

    Returns the concatenated page transcriptions. Confidence is lowered by
    [illegible] and [marginal] marks; the pipeline runs a second pass and
    flags disagreements.
    """
    from .llm import LLMClient

    if not isinstance(client, LLMClient):
        raise TypeError("convert_pdf_scanned_vision requires an LLMClient")

    page_images = pdf_page_images(path)
    system = _transcription_system(main_text_only)
    pages: list[str] = []
    illegible = 0
    marginal = 0
    page_count = 0
    for image in page_images:
        page_count += 1
        text = client.chat(
            role=role,
            system=system,
            user="Transcribe this document page verbatim.",
            images=[image],
            max_tokens=8192,
        )
        illegible += text.count("[illegible]")
        marginal += _count_marginal(text)
        pages.append(text)
    body = "\n\n".join(pages)
    words = max(len(body.split()), 1)
    confidence = max(0.5, 1.0 - (illegible / words) * 10 - (marginal * 0.02))
    return Conversion(
        text=body,
        extraction_method="vision",
        confidence=round(confidence, 3),
        notes=(
            f"scanned PDF, {page_count} page(s), {illegible} illegible, "
            f"{marginal} marginal region(s)"
        ),
    )


def convert(path: Path, client=None) -> Conversion:
    """Dispatch on file type. Images/scans require an LLMClient.

    A PDF with no extractable text is a scan: it is rendered to images and
    transcribed via the vision path automatically.
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return convert_txt(path)
    if suffix == ".docx":
        return convert_docx(path)
    if suffix == ".pdf":
        conversion = convert_pdf(path)
        if not conversion.text.strip():
            if client is None:
                raise RuntimeError(
                    f"{path.name} appears to be a scanned PDF; an LLMClient is required"
                )
            return convert_pdf_scanned_vision(path, client)
        return conversion
    if suffix in IMAGE_SUFFIXES:
        if client is None:
            raise RuntimeError(f"{path.name} is an image; an LLMClient is required")
        return convert_image_vision(path, client)
    raise RuntimeError(f"unsupported file type: {suffix}")
