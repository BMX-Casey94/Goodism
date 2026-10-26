"""Format converters: DOCX, PDF, TXT, and images -> text/markdown.

Text-native formats are converted locally and deterministically. Images
and scanned PDFs go through a vision model (two independent passes; the
pipeline compares them). Converters return plain markdown-ish text with
headings preserved; they never "improve" the content.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class Conversion:
    text: str
    extraction_method: str  # native-text | ocr | vision
    confidence: float | None  # None for native text
    notes: str = ""


def convert_txt(path: Path) -> Conversion:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Conversion(text=text, extraction_method="native-text", confidence=None)


def convert_docx(path: Path) -> Conversion:
    """DOCX -> markdown preserving headings, lists and tables via mammoth."""
    try:
        import mammoth
    except ImportError as exc:
        raise RuntimeError(
            "mammoth is required for DOCX conversion: pip install mammoth"
        ) from exc
    with path.open("rb") as handle:
        result = mammoth.convert_to_markdown(handle)
    notes = "; ".join(m.message for m in result.messages) if result.messages else ""
    return Conversion(
        text=result.value, extraction_method="native-text", confidence=None, notes=notes
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


def convert_image_vision(path: Path, client, *, role: str = "extractor") -> Conversion:
    """Image -> text via a vision model. One pass; the pipeline runs two.

    The model is instructed to transcribe verbatim, preserving spelling,
    punctuation and layout, and to mark illegible spans.
    """
    from .llm import LLMClient

    if not isinstance(client, LLMClient):
        raise TypeError("convert_image_vision requires an LLMClient")

    system = (
        "You are a meticulous transcription engine. Transcribe the document image "
        "VERBATIM: preserve the original spelling (including errors), punctuation, "
        "capitalisation, hyphenation and line structure. Do not correct, modernise, "
        "summarise or comment. Mark any illegible word as [illegible]. Output only "
        "the transcription."
    )
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


def convert(path: Path, client=None) -> Conversion:
    """Dispatch on file type. Images/scans require an LLMClient."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return convert_txt(path)
    if suffix == ".docx":
        return convert_docx(path)
    if suffix == ".pdf":
        conversion = convert_pdf(path)
        if not conversion.text.strip() and client is not None:
            raise RuntimeError(
                f"{path.name} appears to be a scanned PDF; render pages to images "
                "and use the vision path"
            )
        return conversion
    if suffix in IMAGE_SUFFIXES:
        if client is None:
            raise RuntimeError(f"{path.name} is an image; an LLMClient is required")
        return convert_image_vision(path, client)
    raise RuntimeError(f"unsupported file type: {suffix}")
