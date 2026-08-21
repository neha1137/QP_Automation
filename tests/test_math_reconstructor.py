"""
tests/test_math_reconstructor.py — regression tests for the mathematical-
content cleanup/reconstruction stage (math_reconstructor.py) that now runs
between raw PDF text extraction and everything downstream (parser.py,
latex_processor.py).

Two kinds of fixtures are used:
  - Hand-built "dict_page"-shaped plain dicts, for unit-testing the pure
    matching/substitution helpers in isolation without needing real font
    rendering (this is how the PUA stretchy-parenthesis-glyph tests work —
    those exact codepoints have no glyph in an ordinary installed font, so
    they can't be reliably rendered through a synthetic fitz page).
  - Real, in-memory fitz pages (built with fitz.open() + insert_text() +
    draw_line()), for testing the geometry-reading code paths that need an
    actual fitz.Page (get_drawings(), get_text("dict")) end to end — this
    is how the fraction-bar and superscript tests work.

The real-PDF, full-pipeline case (the actual tank question that motivated
this feature) is covered separately in tests/test_latex_real_afcat_pdf.py.
"""

from __future__ import annotations

import fitz
import pytest

import math_reconstructor as mr


# ---------------------------------------------------------------------------
# Helpers to build synthetic fitz pages
# ---------------------------------------------------------------------------

def _blank_page():
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    return doc, page


def _draw_bar(page, x0, x1, y):
    shape = page.new_shape()
    shape.draw_line((x0, y), (x1, y))
    shape.finish(width=0.5)
    shape.commit()


def _stacked_fraction_page(num="22", den="7", with_bar=True, fontsize=12):
    """A synthetic page with `num` positioned directly above `den`,
    optionally connected by a drawn fraction bar."""
    doc, page = _blank_page()
    page.insert_text((50, 50), num, fontsize=fontsize)
    page.insert_text((50, 68), den, fontsize=fontsize)
    if with_bar:
        _draw_bar(page, 48, 65, 54.3)
    return doc, page


# ---------------------------------------------------------------------------
# 1. Fraction reconstruction — real geometry, end to end via reconstruct_math
# ---------------------------------------------------------------------------

def test_vertically_stacked_fraction_with_bar_becomes_frac_marker():
    """The core real-world case: 22 positioned above 7, connected by an
    actual drawn fraction bar -> confidently reconstructed as \\frac{22}{7},
    ready for latex_processor.py to wrap in $...$."""
    doc, page = _stacked_fraction_page("22", "7", with_bar=True)
    try:
        raw = page.get_text()
        assert raw == "22\n7\n"
        cleaned = mr.reconstruct_math(page, raw)
        assert cleaned == r"\frac{22}{7}" + "\n"
    finally:
        doc.close()


def test_stacked_numbers_without_a_bar_are_never_fabricated_into_a_fraction():
    """Two numbers that merely sit on consecutive lines — e.g. the
    "Question 22 / Option 7" shape — must NEVER become a fraction just
    because they're vertically adjacent. Only a confirmed drawn fraction
    bar makes it confident. No bar here -> text passes through unchanged."""
    doc, page = _stacked_fraction_page("22", "7", with_bar=False)
    try:
        raw = page.get_text()
        cleaned = mr.reconstruct_math(page, raw)
        assert cleaned == raw
        assert "\\frac" not in cleaned
    finally:
        doc.close()


def test_fraction_bar_with_non_atomic_content_is_not_converted():
    """A bar exists, but what's above/below it isn't a simple digit/letter
    atom (e.g. a whole word) — never guessed at as a fraction."""
    doc, page = _blank_page()
    try:
        page.insert_text((50, 50), "Given", fontsize=12)
        page.insert_text((50, 68), "here", fontsize=12)
        _draw_bar(page, 48, 90, 54.3)
        raw = page.get_text()
        cleaned = mr.reconstruct_math(page, raw)
        assert cleaned == raw
    finally:
        doc.close()


def test_repeating_grid_bars_are_never_treated_as_fractions():
    """A coding-decoding matrix / answer table draws many bars sharing the
    same x-span (repeating grid lines) — these must be excluded from
    fraction detection, never just because a digit happens to sit above
    and below one of them."""
    doc, page = _blank_page()
    try:
        # Three "rows" of a grid, each bar at the same x-range — a fraction
        # bar for a genuine equation never repeats like this.
        for row, y in enumerate([40, 60, 80]):
            page.insert_text((50, y), "1", fontsize=10)
            page.insert_text((50, y + 16), "2", fontsize=10)
            _draw_bar(page, 48, 65, y + 8)
        raw = page.get_text()
        cleaned = mr.reconstruct_math(page, raw)
        assert cleaned == raw
        assert "\\frac" not in cleaned
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 2. Superscript / subscript reconstruction — real geometry
# ---------------------------------------------------------------------------

def test_geometric_superscript_becomes_unicode_superscript():
    """A smaller, raised digit glued to the preceding base character (no
    gap) — exactly how a genuine exponent like r² is encoded in the real
    PDF — is converted to a real unicode superscript digit so the
    (unmodified) latex_processor.py superscript detector picks it up."""
    doc, page = _blank_page()
    try:
        page.insert_text((50, 100), "r", fontsize=9)
        # raised (smaller y = higher up) and reduced size, touching 'r'
        # exactly (no gap) — matches how the real PDF encodes πr²h.
        page.insert_text((52.997, 96.8), "2", fontsize=7)
        raw = page.get_text()
        assert raw.strip() == "r2"  # sanity: confirms the fixture is glued
        cleaned = mr.reconstruct_math(page, raw)
        assert cleaned.strip() == "r²"
    finally:
        doc.close()


def test_same_size_adjacent_digit_is_not_treated_as_superscript():
    """A same-size, baseline-aligned digit right after a letter (e.g. an
    ordinary alphanumeric code like 'A1') must NOT be reinterpreted as an
    exponent — no size/position evidence supports it."""
    doc, page = _blank_page()
    try:
        page.insert_text((50, 100), "A", fontsize=9)
        page.insert_text((56.003, 100), "1", fontsize=9)
        raw = page.get_text()
        assert raw.strip() == "A1"  # sanity: confirms the fixture is glued
        cleaned = mr.reconstruct_math(page, raw)
        assert cleaned == raw
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 3. Decimal-point reconstruction (hand-built dict fixtures)
# ---------------------------------------------------------------------------

def _span(text, x0, y0, x1, y1, size=9.0, font="Test"):
    return {"text": text, "bbox": (x0, y0, x1, y1), "size": size, "font": font}


def _dict_page(spans):
    return {"blocks": [{"lines": [{"spans": spans}]}]}


def test_decimal_point_merge_from_split_span_and_orphaned_period():
    dict_page = _dict_page(
        [
            _span("3 5", 174.0, 485.8, 185.1, 497.0),
            _span(".", 178.4, 485.8, 180.6, 497.0),
        ]
    )
    merges = mr._find_decimal_merges(dict_page)
    assert merges == [("3", "5")]
    text = "Volume = 3 5\nsome other text\n.\nmore text"
    result = mr._apply_decimal_merges(text, merges)
    assert "3.5" in result
    assert not result.count("\n.\n")


def test_decimal_merge_not_applied_without_a_confirming_period_span():
    dict_page = _dict_page([_span("3 5", 174.0, 485.8, 185.1, 497.0)])
    assert mr._find_decimal_merges(dict_page) == []


# ---------------------------------------------------------------------------
# 4. Stretchy-parenthesis PUA glyph collapse (hand-built dict fixtures)
# ---------------------------------------------------------------------------

def test_pua_bracket_pieces_collapse_to_literal_parens_when_font_confirms():
    open_pieces = ""
    close_pieces = ""
    dict_page = _dict_page(
        [{"text": c, "bbox": (0, 0, 1, 1), "size": 9.0, "font": "EuclidSymbol"} for c in open_pieces]
        + [{"text": c, "bbox": (0, 0, 1, 1), "size": 9.0, "font": "EuclidSymbol"} for c in close_pieces]
    )
    assert mr._page_uses_paren_piece_font(dict_page) is True
    text = f"= {open_pieces}\n22\n7\n{close_pieces}×(3.5) × 8"
    cleaned = mr._collapse_bracket_pieces(text)
    assert open_pieces not in cleaned
    assert close_pieces not in cleaned
    assert "(" in cleaned and ")" in cleaned
    assert "×(3.5) × 8" in cleaned  # trailing content preserved verbatim


def test_pua_like_chars_left_untouched_without_font_confirmation():
    """The exact same codepoints, but attributed to an ordinary font (no
    Euclid/Symbol/MTExtra hint) — never blindly collapsed; per the
    "don't delete real content you can't identify" rule, an unrecognized
    PUA usage must be left completely alone."""
    dict_page = _dict_page(
        [{"text": "", "bbox": (0, 0, 1, 1), "size": 9.0, "font": "Arial"}]
    )
    assert mr._page_uses_paren_piece_font(dict_page) is False


# ---------------------------------------------------------------------------
# 4b. Full equation-block reconstruction (hand-built dict/bar fixtures) —
#     the "Volume = (22/7) × (3.5)² × 8" multi-part rebuild, tested via
#     precise, deterministic fixtures rather than fragile synthetic-font
#     positioning (real end-to-end proof against the actual PDF lives in
#     tests/test_latex_real_afcat_pdf.py).
# ---------------------------------------------------------------------------

def _rect(x0, y0, x1, y1):
    return fitz.Rect(x0, y0, x1, y1)


def test_confident_equation_text_reconstructs_multi_term_row():
    """Volume = 22/7 (bar-confirmed) × 5 — spread across two dict
    "blocks" exactly like the real PDF (numerator sharing a block with
    the preceding label text), reconstructed into one coherent row."""
    dict_page = {
        "blocks": [
            {"lines": [{"spans": [
                _span("Volume = ", 50.0, 60.0, 90.0, 72.0),
                _span("22", 100.0, 50.0, 110.0, 62.0),
            ]}]},
            {"lines": [{"spans": [
                _span("7", 100.0, 64.0, 108.0, 76.0),
                _span("× 5", 115.0, 60.0, 135.0, 72.0),
            ]}]},
        ]
    }
    bars = [_rect(98.0, 64.5, 112.0, 64.5)]
    result = mr._confident_equation_text(dict_page, bars, [0, 1])
    assert result == r"Volume = \frac{22}{7} × 5"


def test_confident_equation_text_glues_superscript_and_drops_redundant_parens():
    """(3.5)² glued correctly, and a stretchy-paren pair that directly
    wraps nothing but a lone \\frac{}{} is dropped as purely decorative."""
    dict_page = {
        "blocks": [
            {"lines": [{"spans": [
                _span("X = ", 100.0, 60.0, 120.0, 72.0),
                _span("\uf8eb", 140.0, 60.0, 144.0, 72.0, font="EuclidSymbol"),
                _span("22", 148.0, 50.0, 157.0, 62.0),
            ]}]},
            {"lines": [{"spans": [
                _span("7", 150.0, 64.0, 155.0, 76.0),
                _span("\uf8f6", 158.0, 60.0, 162.0, 72.0, font="EuclidSymbol"),
                _span("(3.5)", 170.0, 60.0, 190.0, 72.0),
                _span("2", 190.0, 57.0, 193.0, 64.0, size=6.3),  # raised, small
            ]}]},
        ]
    }
    bars = [_rect(148.0, 64.5, 161.0, 64.5)]
    result = mr._confident_equation_text(dict_page, bars, [0, 1])
    assert result == r"X = \frac{22}{7} (3.5)²"


def test_confident_equation_text_rejects_row_overlap_ambiguity():
    """Two unrelated atoms whose x-ranges genuinely overlap after being
    clustered into the same visual row — a sign the y-proximity row
    clustering conflated two distinct lines. Must reject, never guess an
    ordering between them."""
    dict_page = {
        "blocks": [
            {"lines": [{"spans": [
                _span("22", 100.0, 50.0, 110.0, 62.0),
                _span("foo", 102.0, 50.0, 130.0, 62.0),  # overlaps "22" in x
            ]}]},
            {"lines": [{"spans": [_span("7", 100.0, 64.0, 108.0, 76.0)]}]},
        ]
    }
    bars = [_rect(98.0, 64.5, 112.0, 64.5)]
    assert mr._confident_equation_text(dict_page, bars, [0, 1]) is None


def test_confident_equation_text_rejects_when_a_row_lacks_an_equals_sign():
    """A row with no "=" doesn't stand on its own as a complete equation —
    the real-PDF case this guards against left a dangling '(7x + 8)' line
    from a ratio-equation layout this module doesn't confidently handle."""
    dict_page = {
        "blocks": [
            {"lines": [{"spans": [_span("(7x + 8)", 50.0, 60.0, 100.0, 72.0)]}]},
        ]
    }
    assert mr._confident_equation_text(dict_page, [], [0]) is None


# ---------------------------------------------------------------------------
# 4c. Evidence-block detection and clustering
# ---------------------------------------------------------------------------

def test_evidence_block_indices_flags_only_pua_and_bar_touching_blocks():
    pua_font_span = _span("", 0.0, 0.0, 1.0, 1.0, font="EuclidSymbol")
    ordinary_prose_span = _span("This is an ordinary sentence.", 0.0, 100.0, 200.0, 112.0)
    bar_touching_span = _span("22", 100.0, 50.0, 110.0, 62.0)
    dict_page = {
        "blocks": [
            {"lines": [{"spans": [pua_font_span]}]},        # block 0: evidence (PUA)
            {"lines": [{"spans": [ordinary_prose_span]}]},   # block 1: NOT evidence
            {"lines": [{"spans": [bar_touching_span]}]},     # block 2: evidence (bar)
        ]
    }
    bars = [_rect(98.0, 64.5, 112.0, 64.5)]
    evidence = mr._evidence_block_indices(dict_page, bars)
    assert evidence == {0, 2}


def test_cluster_consecutive_merges_only_adjacent_indices():
    assert mr._cluster_consecutive({15, 16, 17, 18, 19}) == [[15, 16, 17, 18, 19]]
    assert mr._cluster_consecutive({2, 5, 6, 9}) == [[2], [5, 6], [9]]
    assert mr._cluster_consecutive(set()) == []


# ---------------------------------------------------------------------------
# 4d. Block-text-range location (splicing target) — count-aware matching
# ---------------------------------------------------------------------------

def test_find_span_occurrences_is_count_aware_for_duplicate_text():
    """A span text ('.') that legitimately appears twice in the block must
    be located twice, not the same first occurrence reused — the exact
    real-PDF bug (two orphaned decimal-point spans) this guards against."""
    text = "a.b.c"
    result = mr._find_span_occurrences(text, [".", "."], start=0, window=10)
    assert result == (1, 4)  # covers both "." occurrences (indices 1 and 3)


def test_find_span_occurrences_returns_none_when_not_found():
    result = mr._find_span_occurrences("hello world", ["xyz"], start=0, window=20)
    assert result is None


# ---------------------------------------------------------------------------
# 4e. Full orchestration: splice a confident cluster, leave the rest alone
# ---------------------------------------------------------------------------

class _FakePage:
    """A minimal stand-in for fitz.Page exposing only what
    reconstruct_equation_blocks actually calls: get_drawings()."""

    def __init__(self, bar_rects):
        self._bars = bar_rects

    def get_drawings(self):
        items = []
        for r in self._bars:
            items.append({"items": [("l", fitz.Point(r.x0, r.y0), fitz.Point(r.x1, r.y1))]})
        return items


def test_reconstruct_equation_blocks_splices_confident_region_only():
    text = "Before text.\nVolume = 22\n7\n× 5\nAfter, unrelated 22\nQuestion 7 follows."
    dict_page = {
        "blocks": [
            # "Volume = " sharing a block with the numerator "22", exactly
            # as PyMuPDF grouped it in the real PDF (block17 there).
            {"lines": [{"spans": [
                _span("Volume = ", 50.0, 60.0, 90.0, 72.0),
                _span("22", 100.0, 50.0, 110.0, 62.0),
            ]}]},
            {"lines": [{"spans": [
                _span("7", 100.0, 64.0, 108.0, 76.0),
                _span("× 5", 115.0, 60.0, 135.0, 72.0),
            ]}]},
        ]
    }
    page = _FakePage([_rect(98.0, 64.5, 112.0, 64.5)])
    result = mr.reconstruct_equation_blocks(page, dict_page, text)
    assert r"Volume = \frac{22}{7} × 5" in result
    # The unrelated trailing "22" / "Question 7" text is completely
    # untouched — the cluster's splice range must not overreach.
    assert "After, unrelated 22" in result
    assert "Question 7 follows." in result


def test_reconstruct_equation_blocks_is_noop_with_no_evidence():
    text = "Just an ordinary paragraph with no math in it at all."
    dict_page = {"blocks": [{"lines": [{"spans": [_span("Just an ordinary paragraph with no math in it at all.", 0, 0, 300, 12)]}]}]}
    page = _FakePage([])
    assert mr.reconstruct_equation_blocks(page, dict_page, text) == text


# ---------------------------------------------------------------------------
# 5. Fail-open behavior
# ---------------------------------------------------------------------------

def test_reconstruct_math_returns_original_text_on_any_internal_error():
    class ExplodingPage:
        def get_text(self, *_a, **_k):
            raise RuntimeError("boom")

        def get_drawings(self):
            raise RuntimeError("boom")

    assert mr.reconstruct_math(ExplodingPage(), "22\n7\n") == "22\n7\n"


def test_reconstruct_math_passes_through_empty_and_none_like_inputs():
    class DummyPage:
        def get_text(self, *_a, **_k):
            return {"blocks": []}

        def get_drawings(self):
            return []

    assert mr.reconstruct_math(DummyPage(), "") == ""
