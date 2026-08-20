"""
test_ocr_mixed.py — a document with one native-text page and one
image-only page must be handled per-page: native stays native, only the
image-only page pays the OCR cost.
"""

from __future__ import annotations

import fitz
import pytest

from extractor import check_tesseract_available, extract_document
from test_ocr_fallback import _make_image_only_pdf

pytestmark = pytest.mark.skipif(
    not check_tesseract_available()[0],
    reason="Tesseract OCR engine not installed in this environment",
)


def _add_native_text_page(doc: "fitz.Document"):
    page = doc.new_page(width=612, height=792)
    text = (
        "1.\tWhich word is the odd one out?\n"
        "1.\tApple\n2.\tBanana\n3.\tCarrot\n4.\tMango\n"
    ) * 8  # enough real text to clear the quality-score threshold
    page.insert_text((36, 36), text, fontsize=11)


def test_mixed_native_and_ocr_pages(tmp_path):
    pdf_path = str(tmp_path / "mixed.pdf")

    doc = fitz.open()
    _add_native_text_page(doc)  # page 1: native
    doc.save(pdf_path)
    doc.close()

    # Append an image-only page (page 2) via a second temp PDF, then merge.
    image_only_path = str(tmp_path / "image_only_part.pdf")
    _make_image_only_pdf(image_only_path, ["SCANNED QUESTION PAGE"])

    merged = fitz.open(pdf_path)
    to_insert = fitz.open(image_only_path)
    merged.insert_pdf(to_insert)
    merged.saveIncr()
    merged.close()
    to_insert.close()

    result = extract_document(pdf_path)
    assert len(result.pages) == 2
    assert result.native_page_count == 1
    assert result.ocr_page_count == 1
    assert result.pages[0].extraction_method == "native"
    assert result.pages[1].extraction_method == "ocr"
