"""
tests/test_image_locator.py — Regression tests for image_locator.py against
real PDF coordinates (SSC/AFCAT samples, already used by the rest of the
suite via conftest.py's session-scoped fixtures) plus the synthetic raster
fixture for the one code path neither real sample PDF actually exercises
(an embedded raster image belonging to a real question — both real PDFs'
image questions are vector-drawn, confirmed by direct inspection; see
image_locator.py's module docstring).
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from image_locator import locate_image_regions

ROOT = Path(__file__).resolve().parent.parent
SSC_PDF = str(ROOT / "SSC Stenographer MTP-2.pdf")
AFCAT_PDF = str(ROOT / "templates" / "AFCAT_Mock_1-5_SP_2026_Final_File.pdf")
SYNTHETIC_RASTER_PDF = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_raster.pdf")


@pytest.fixture(scope="module")
def ssc_pdf():
    doc = fitz.open(SSC_PDF)
    yield doc
    doc.close()


@pytest.fixture(scope="module")
def afcat_pdf():
    doc = fitz.open(AFCAT_PDF)
    yield doc
    doc.close()


# -- vector diagram detection (real SSC PDF) --------------------------------

def test_ssc_q5_detected_on_correct_page_with_clean_option_split(ssc_pdf):
    page = ssc_pdf[0]  # Q5 is on page 1
    candidates = locate_image_regions(page, 5, 6)
    dests = {c.destination for c in candidates}
    assert dests == {"question", "option_a", "option_b", "option_c", "option_d"}
    for c in candidates:
        assert c.confidence == "high"
        assert c.source_type == "pdf_region_render"  # vector diagram, not embedded raster


def test_ssc_q5_region_is_within_expected_page_band(ssc_pdf):
    page = ssc_pdf[0]
    candidates = locate_image_regions(page, 5, 6)
    question_region = next(c for c in candidates if c.destination == "question")
    x0, y0, x1, y1 = question_region.bbox
    # Q5's diagram sits between its own marker (y~377) and the option row
    # (y~446-476) — a real, verified band from direct PDF inspection.
    assert 370 <= y0 <= 480
    assert 370 <= y1 <= 480
    assert x1 > x0


@pytest.mark.parametrize("number,next_number", [(39, 40), (44, 45), (45, 46), (46, 47), (47, 48)])
def test_ssc_page2_image_questions_all_detected(ssc_pdf, number, next_number):
    page = ssc_pdf[1]  # all these are on page 2
    candidates = locate_image_regions(page, number, next_number)
    assert candidates, f"Q{number} should have detected at least one region"
    dests = {c.destination for c in candidates}
    assert dests == {"question", "option_a", "option_b", "option_c", "option_d"}


def test_multiple_adjacent_image_questions_get_distinct_non_overlapping_regions(ssc_pdf):
    """Q44-47 sit back-to-back on the same page — their 'question' regions
    must not bleed into each other (a real risk this locator specifically
    guards against via the next-marker boundary)."""
    page = ssc_pdf[1]
    regions = {}
    for num, nxt in [(44, 45), (45, 46), (46, 47)]:
        cands = locate_image_regions(page, num, nxt)
        q = next(c for c in cands if c.destination == "question")
        regions[num] = q.bbox

    # Sorted by y0, each question's region must end before the next begins.
    ordered = sorted(regions.items(), key=lambda kv: kv[1][1])
    for (n1, r1), (n2, r2) in zip(ordered, ordered[1:]):
        assert r1[3] <= r2[1] + 1.0, f"Q{n1} region overlaps Q{n2}'s region"


# -- non-image questions never produce candidates ---------------------------

def test_text_only_question_returns_no_candidates(ssc_pdf):
    page = ssc_pdf[0]
    assert locate_image_regions(page, 1, 2) == []


# -- two-column layout doesn't bleed content from the other column ---------

def test_column_aware_band_excludes_other_column_content(ssc_pdf):
    """Real bug caught during implementation: Q5's y-band, if taken at
    full page width, overlaps Q11/Q12's UNRELATED content in the right
    column at the same y-range. The 'question' region's right edge must
    stay within the left column, nowhere near the right column's ~330+
    x-position content confirmed present in that same y-band."""
    page = ssc_pdf[0]
    candidates = locate_image_regions(page, 5, 6)
    question_region = next(c for c in candidates if c.destination == "question")
    assert question_region.bbox[2] < 310  # right edge stays in the left column


# -- AFCAT (different page size, different option-marker scheme) -----------

def test_afcat_letter_mode_options_split_into_separate_destinations(afcat_pdf):
    """AFCAT uses letter-mode options ('(a)'/'(b)'/'(c)'/'(d)'). The
    locator recognizes this scheme too (not just SSC's digit-dot scheme),
    so an AFCAT image question splits into 5 independent regions — the
    question's own diagram plus one crop per option — visually confirmed
    correct during implementation (each option crop contains exactly one
    of the four answer-choice symbols, nothing more)."""
    page = afcat_pdf[3]  # Q52 is on page 4
    candidates = locate_image_regions(page, 52, 53)
    dests = {c.destination for c in candidates}
    assert dests == {"question", "option_a", "option_b", "option_c", "option_d"}
    for c in candidates:
        assert c.confidence == "high"


def test_afcat_grid_layout_text_question_never_fabricates_an_image(afcat_pdf):
    """Real finding during Fix 1 implementation: AFCAT Q49 has a genuine
    2x2 grid of (a)/(b)/(c)/(d) markers (confirmed via direct PDF
    inspection) but its options are plain text ('30 years', '30.5 years',
    ...), not images. The grid-partition logic must correctly find NO
    raster/drawing content in any of the 4 cells and return [] — proving
    grid detection doesn't fabricate visual content just because a grid
    layout is present."""
    page = afcat_pdf[3]
    candidates = locate_image_regions(page, 49, 50)
    assert candidates == []


# -- synthetic fixture: embedded raster image (real PDFs don't exercise this) --

def test_synthetic_fixture_detects_embedded_raster_image():
    doc = fitz.open(SYNTHETIC_RASTER_PDF)
    try:
        page = doc[0]
        assert len(page.get_images(full=True)) == 1  # sanity: fixture really has a raster image
        candidates = locate_image_regions(page, 1, 2)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.destination == "question"
        assert c.source_type == "embedded_raster"
        assert c.raster_xref is not None
        # bbox matches the exact rect the fixture placed the image at.
        assert c.bbox == pytest.approx((60.0, 80.0, 180.0, 200.0), abs=0.5)
    finally:
        doc.close()


def test_question_numbered_one_does_not_collide_with_its_own_option_search():
    """Real bug caught by this fixture during implementation: a question
    numbered "1" has an opening marker "1." that, without a fix, was being
    mistaken for option 1's own marker (same literal text). Must resolve
    to 0 option markers found (no options exist in this fixture's Q1 band
    at all), not 1."""
    doc = fitz.open(SYNTHETIC_RASTER_PDF)
    try:
        page = doc[0]
        candidates = locate_image_regions(page, 1, 2)
        # If the bug were present, this would come back "ambiguous" — the
        # fixed behavior is a clean single "question" region.
        assert {c.destination for c in candidates} == {"question"}
    finally:
        doc.close()
