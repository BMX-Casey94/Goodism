import base64
import io
import re

from PIL import Image

from tools.kb import assets, config


def _png(colour: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), colour).save(buf, "PNG")
    return buf.getvalue()


def _uri(data: bytes) -> str:
    return f"![](data:image/png;base64,{base64.b64encode(data).decode()})"


def test_images_become_files_and_identical_images_are_shared(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "domain").mkdir(parents=True)
    monkeypatch.setattr(config, "CORPUS_DIR", corpus)
    monkeypatch.setattr(assets, "ASSETS_DIR", corpus / "assets")
    monkeypatch.setattr(assets, "INDEX_FILE", corpus / "assets" / "index.jsonl")

    red, blue = _png("red"), _png("blue")
    text = f"Before {_uri(red)} middle {_uri(blue)} and again {_uri(red)} after."
    doc = corpus / "domain" / "doc.md"
    out = assets.externalise_images(text, "doc", doc)

    assert "base64" not in out
    refs = re.findall(r"\]\((\.\./assets/doc/[^)]+)\)", out)
    assert len(refs) == 3 and refs[0] == refs[2] != refs[1]
    # Only the words around the images remain, unchanged.
    assert re.sub(r"!\[[^\]]*\]\([^)]*\)", "", out).split() == "Before middle and again after.".split()
    assert len(list((corpus / "assets" / "doc").iterdir())) == 2
    assert assets.check_references(doc, out) == []


def test_check_references_reports_missing_and_tampered(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "domain").mkdir(parents=True)
    monkeypatch.setattr(config, "CORPUS_DIR", corpus)
    monkeypatch.setattr(assets, "ASSETS_DIR", corpus / "assets")
    monkeypatch.setattr(assets, "INDEX_FILE", corpus / "assets" / "index.jsonl")

    doc = corpus / "domain" / "doc.md"
    out = assets.externalise_images(_uri(_png("red")), "doc", doc)
    target = corpus / "assets" / "doc" / "img-001.png"
    target.write_bytes(_png("green"))
    assert any("hash mismatch" in p for p in assets.check_references(doc, out))
    target.unlink()
    assert any("missing image" in p for p in assets.check_references(doc, out))
