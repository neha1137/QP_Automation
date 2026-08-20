"""
test_ocr_fallback.py — a synthetic, image-only (no vector text at all)
PDF must fall back to OCR. Built fresh at test time via PyMuPDF + Pillow
(both already project dependencies) — no binary fixture checked in, so it
can't go stale relative to the code.
"""

from __future__ import annotations

import os
import tempfile

import fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from extractor import check_tesseract_available, extract_document

pytestmark = pytest.mark.skipif(
    not check_tesseract_available()[0],
    reason="Tesseract OCR engine not installed in this environment",
)


def _text_image(lines: list[str], size=(1000, 400)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    y = 30
    for line in lines:
        draw.text((30, y), line, fill="black", font=font)
        y += 60
    return img


def _make_image_only_pdf(path: str, lines: list[str]):
    """A PDF page with ONLY a rendered image — page.get_text() returns
    empty, forcing the OCR fallback path."""
    img = _text_image(lines)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        img.save(tmp_img.name)
        img_path = tmp_img.name
    try:
        doc = fitz.open()
        page = doc.new_page(width=img.width, height=img.height)
        page.insert_image(fitz.Rect(0, 0, img.width, img.height), filename=img_path)
        doc.save(path)
        doc.close()
    finally:
        os.remove(img_path)


def test_image_only_page_triggers_ocr(tmp_path):
    pdf_path = str(tmp_path / "image_only.pdf")
    _make_image_only_pdf(pdf_path, ["HELLO WORLD", "QUESTION ONE"])

    result = extract_document(pdf_path)
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.extraction_method == "ocr"
    assert result.ocr_page_count == 1
    assert result.native_page_count == 0
    # Loose, noise-tolerant check — OCR output on a synthetic render isn't
    # guaranteed pixel-perfect, so this only checks recognizable content
    # made it through, not an exact match.
    assert "HELLO" in page.text.upper() or "WORLD" in page.text.upper()
