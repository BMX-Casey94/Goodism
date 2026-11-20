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
import binascii
import hashlib
import json
import os
import re
from pathlib import Path

from . import config

ASSETS_DIR = config.CORPUS_DIR / "assets"
INDEX_FILE = ASSETS_DIR / "index.jsonl"
# Bounded to prevent quadratic ReDoS on repeated '![' and oversized payloads.
_DATA_URI = re.compile(
    r"!\[[^\]\n]{0,500}\]\(data:image/([a-zA-Z0-9.+-]{1,20});base64,([A-Za-z0-9+/=\s]{0,14000000})\)",
    re.IGNORECASE,
)
_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "gif": "gif", "bmp": "bmp", "webp": "webp"}
_DOC_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ASSET_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_REF = re.compile(r"!\[[^\]]*\]\(((?:\.\./)+assets/[^)\s]+)\)")
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB per extracted image


def _sniff_image_ext(data: bytes) -> str | None:
    """Return canonical raster extension from magic bytes, or None if unsupported."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"
    return None


def _load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    rows: list[dict] = []
    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _append_index(rows: list[dict]) -> None:
    if not rows:
        return
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        existing = {(r["doc_id"], r["file"]) for r in _load_index()}
    except (OSError, json.JSONDecodeError, KeyError):
        existing = set()
    with INDEX_FILE.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            if (row["doc_id"], row["file"]) not in existing:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                existing.add((row["doc_id"], row["file"]))


def _dimensions(data: bytes) -> tuple[int | None, int | None]:
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.width, im.height
    except Exception:  # pragma: no cover - dimensions are optional metadata
        return None, None


def _write_asset(doc_id: str, name: str, data: bytes) -> Path:
    if not _DOC_ID_RE.match(doc_id):
        raise ValueError(f"invalid asset doc_id: {doc_id!r}")
    if not _ASSET_NAME_RE.match(name) or ".." in name:
        raise ValueError(f"invalid asset filename: {name!r}")
    root = ASSETS_DIR.resolve()
    root.mkdir(parents=True, exist_ok=True)
    doc_dir = (root / doc_id)
    if doc_dir.exists() and (doc_dir.is_symlink() or Path(os.path.realpath(doc_dir)) != doc_dir.resolve()):
        raise ValueError("asset directory is a symlink/junction")
    target = (doc_dir / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError("asset path escapes corpus/assets")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_symlink():
        raise ValueError("asset target is a symlink")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def externalise_images(text: str, doc_id: str, doc_path: Path, source: str = "docx") -> str:
    """Replace inline base64 images with relative file links; return new text."""
    if not _DOC_ID_RE.match(doc_id):
        raise ValueError(f"invalid asset doc_id: {doc_id!r}")
    rows: list[dict] = []
    seen: dict[str, str] = {}  # sha -> filename (identical images share one file)
    counter = 0

    def repl(match: re.Match) -> str:
        nonlocal counter
        subtype, payload = match.group(1).lower(), match.group(2)
        # Strict allowlist: reject svg, php, xhtml, or any non-raster subtype.
        if subtype not in _EXT:
            return match.group(0)
        try:
            data = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError):
            return match.group(0)
        if not data or len(data) > MAX_IMAGE_BYTES:
            return match.group(0)
        sniffed = _sniff_image_ext(data)
        # Allow synthetic tiny test PNGs/bytes only if magic matches; reject spoofed scripts/SVGs.
        if sniffed is None:
            return match.group(0)
        sha = hashlib.sha256(data).hexdigest()
        if sha not in seen:
            counter += 1
            name = f"img-{counter:03d}.{sniffed}"
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

    if not _DOC_ID_RE.match(doc_id):
        raise ValueError(f"invalid asset doc_id: {doc_id!r}")
    rows: list[dict] = []
    seen: set[str] = set()
    with pymupdf.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            for info in page.get_images(full=True):
                xref = info[0]
                img = pdf.extract_image(xref)
                data = img["image"]
                if not data or len(data) > MAX_IMAGE_BYTES:
                    continue
                ext = _sniff_image_ext(data) or _EXT.get(img["ext"].lower())
                if ext is None:
                    continue
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
    """Return problems: escaping links, unresolved links, or hash mismatches."""
    problems: list[str] = []
    try:
        index = {(r["doc_id"], r["file"]): r for r in _load_index()}
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return [f"corrupt assets index: {type(exc).__name__}"]
    assets_root = ASSETS_DIR.resolve()
    for ref in _REF.findall(text):
        target = (doc_path.parent / ref).resolve()
        if not target.is_relative_to(assets_root):
            problems.append(f"image path escapes corpus/assets: {ref}")
            continue
        if not target.exists() or not target.is_file() or target.is_symlink():
            problems.append(f"missing image: {ref}")
            continue
        row = index.get((target.parent.name, target.name))
        if row is None:
            problems.append(f"image not in assets index: {ref}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
            problems.append(f"image hash mismatch: {ref}")
    return problems
