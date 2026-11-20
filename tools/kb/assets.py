"""Image assets: stored as real files, referenced from the corpus text.

Documents must never contain base64 blobs. Images are written to
``corpus/assets/<doc_id>/`` as ordinary files and referenced with a relative
Markdown link, so a reader (or an LLM tool) can load an image only when
needed. ``corpus/assets/index.jsonl`` records every asset with its SHA-256,
size, and (for PDFs) the source page. Alt text is a neutral label
("Image 3"); it is metadata, not part of the author's wording.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from . import config

ASSETS_DIR = config.CORPUS_DIR / "assets"
INDEX_FILE = ASSETS_DIR / "index.jsonl"
_DATA_URI = re.compile(r"!\[([^\]]*)\]\(data:image/([a-zA-Z0-9.+-]+);base64,([^)]*)\)")
_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "gif": "gif", "bmp": "bmp", "webp": "webp"}
_REF = re.compile(r"!\[[^\]]*\]\(((?:\.\./)+assets/[^)\s]+)\)")


def _load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    return [json.loads(l) for l in INDEX_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def _append_index(rows: list[dict]) -> None:
    if not rows:
        return
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {(r["doc_id"], r["file"]) for r in _load_index()}
    with INDEX_FILE.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            if (row["doc_id"], row["file"]) not in existing:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _dimensions(data: bytes) -> tuple[int | None, int | None]:
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.width, im.height
    except Exception:  # pragma: no cover - dimensions are optional metadata
        return None, None


def _write_asset(doc_id: str, name: str, data: bytes) -> Path:
    target = ASSETS_DIR / doc_id / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def externalise_images(text: str, doc_id: str, doc_path: Path, source: str = "docx") -> str:
    """Replace inline base64 images with relative file links; return new text."""
    rows: list[dict] = []
    seen: dict[str, str] = {}  # sha -> filename (identical images share one file)
    counter = 0

    def repl(match: re.Match) -> str:
        nonlocal counter
        subtype, payload = match.group(2).lower(), match.group(3)
        data = base64.b64decode(payload)
        sha = hashlib.sha256(data).hexdigest()
        if sha not in seen:
            counter += 1
            name = f"img-{counter:03d}.{_EXT.get(subtype, subtype)}"
            _write_asset(doc_id, name, data)
            w, h = _dimensions(data)
            rows.append(
                {
                    "doc_id": doc_id, "file": name, "sha256": sha, "bytes": len(data),
                    "width": w, "height": h, "source": source, "page": None,
                }
            )
            seen[sha] = name
        name = seen[sha]
        rel = Path(
            *([".."] * len(doc_path.relative_to(config.CORPUS_DIR).parts[:-1])),
            "assets", doc_id, name,
        ).as_posix()
        return f"![Image {int(Path(name).stem.split('-')[1])}]({rel})"

    new_text = _DATA_URI.sub(repl, text)
    _append_index(rows)
    return new_text


def extract_pdf_figures(pdf_path: Path, doc_id: str, min_edge: int = 150) -> list[dict]:
    """Save embedded raster figures from a native PDF; record page numbers.

    Figures are not inserted into the text (the PDF text has no reliable
    anchor position); they are discoverable through ``index.jsonl``.
    """
    import pymupdf

    rows: list[dict] = []
    seen: set[str] = set()
    with pymupdf.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            for info in page.get_images(full=True):
                xref = info[0]
                img = pdf.extract_image(xref)
                data, ext = img["image"], _EXT.get(img["ext"].lower(), img["ext"].lower())
                if min(img["width"], img["height"]) < min_edge:
                    continue
                sha = hashlib.sha256(data).hexdigest()
                if sha in seen:
                    continue
                seen.add(sha)
                name = f"fig-p{page_no:04d}-{len(rows) + 1:02d}.{ext}"
                _write_asset(doc_id, name, data)
                rows.append(
                    {
                        "doc_id": doc_id, "file": name, "sha256": sha, "bytes": len(data),
                        "width": img["width"], "height": img["height"],
                        "source": "pdf", "page": page_no,
                    }
                )
    _append_index(rows)
    return rows


def check_references(doc_path: Path, text: str) -> list[str]:
    """Return problems: unresolved links or hashes that differ from the index."""
    problems: list[str] = []
    index = {(r["doc_id"], r["file"]): r for r in _load_index()}
    for ref in _REF.findall(text):
        target = (doc_path.parent / ref).resolve()
        if not target.exists():
            problems.append(f"missing image: {ref}")
            continue
        row = index.get((target.parent.name, target.name))
        if row is None:
            problems.append(f"image not in assets index: {ref}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
            problems.append(f"image hash mismatch: {ref}")
    return problems
