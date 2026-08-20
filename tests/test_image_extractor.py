"""
tests/test_image_extractor.py — image_extractor.py: bbox -> real bytes,
lossless-raster-vs-render selection, hashing.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import pytest
from PIL import Image

from image_locator import RegionCandidate, locate_image_regions
from image_extractor import build_artifact, compute_sha256, hash_and_type_for_upload, render_region

ROOT = Path(__file__).resolve().parent.parent
SSC_PDF = str(ROOT / "SSC Stenographer MTP-2.pdf")
SYNTHETIC_RASTER_PDF = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_raster.pdf")


def test_render_region_produces_valid_png_for_vector_content():
    doc = fitz.open(SSC_PDF)
    try:
        page = doc[0]
        data = render_region(page, (99.0, 402.0, 298.0, 476.0))
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        img = Image.open(io.BytesIO(data))
        img.verify()
    finally:
        doc.close()


def test_build_artifact_for_vector_diagram_uses_render_path():
    doc = fitz.open(SSC_PDF)
    try:
        page = doc[0]
        candidates = locate_image_regions(page, 5, 6)
        question_cand = next(c for c in candidates if c.destination == "question")
        artifact = build_artifact(page, question_cand)
        assert artifact is not None
        assert artifact.source_type == "pdf_region_render"
        assert artifact.content_type == "image/png"
        assert len(artifact.image_bytes) > 0
        assert artifact.sha256 == compute_sha256(artifact.image_bytes)
        # bytes really are a valid image
        Image.open(io.BytesIO(artifact.image_bytes)).verify()
    finally:
        doc.close()


def test_build_artifact_for_embedded_raster_uses_lossless_path():
    doc = fitz.open(SYNTHETIC_RASTER_PDF)
    try:
        page = doc[0]
        candidates = locate_image_regions(page, 1, 2)
        artifact = build_artifact(page, candidates[0])
        assert artifact is not None
        assert artifact.source_type == "embedded_raster"
        assert artifact.content_type == "image/png"
        Image.open(io.BytesIO(artifact.image_bytes)).verify()
    finally:
        doc.close()


def test_build_artifact_returns_none_for_degenerate_bbox():
    doc = fitz.open(SSC_PDF)
    try:
        page = doc[0]
        bad = RegionCandidate(
            destination="question", bbox=(100.0, 100.0, 100.0, 100.0),  # zero area
            confidence="high", source_type="pdf_region_render", reason="",
        )
        assert build_artifact(page, bad) is None
    finally:
        doc.close()


def test_compute_sha256_is_deterministic_and_content_sensitive():
    a = compute_sha256(b"hello")
    b = compute_sha256(b"hello")
    c = compute_sha256(b"world")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_hash_and_type_for_upload_detects_content_type_from_filename():
    sha_png, ct_png = hash_and_type_for_upload(b"\x89PNG\r\n\x1a\nfake", "diagram.png")
    sha_jpg, ct_jpg = hash_and_type_for_upload(b"\xff\xd8\xff fake", "diagram.jpg")
    sha_jpeg, ct_jpeg = hash_and_type_for_upload(b"\xff\xd8\xff fake", "diagram.jpeg")
    assert ct_png == "image/png"
    assert ct_jpg == "image/jpeg"
    assert ct_jpeg == "image/jpeg"
    assert sha_png != sha_jpg  # different bytes -> different hash, never blindly labeled the same
