"""
image_locator.py — Locates the PDF-page region belonging to a specific
image question (and, where determinable, which part of that region is the
question's own diagram vs. an individual option's image).

Why this exists as a separate module from parser.py: parser.py works
entirely on the unified, page-tagged TEXT stream — it has no notion of
on-page coordinates at all (confirmed by inspection: extractor.py only
ever calls page.get_text(), never page.get_text("words")/get_images()/
get_drawings()). This module adds a second, independent read of the same
PDF page via PyMuPDF's coordinate APIs, but never re-derives question
BOUNDARIES from scratch — it only asks "where on the page does the
already-known question `number` start, and where does the already-known
next question start", using data parser.py already got right.

Empirical grounding (both real sample PDFs, inspected directly):
    SSC PDF:   every image question (Q5, Q39, Q44-47) is a VECTOR drawing
               region (page.get_images() == 0 on those pages; get_drawings()
               has 33-227 path entries). No embedded raster images at all.
    AFCAT PDF: same pattern on every paper (52/59/60/72/... etc across
               pages 4-78) — one single embedded raster image exists in the
               entire 89-page document (page 15), and it does NOT belong to
               any image question (likely a logo/page furniture).
So: page.get_images() is tried first (cheap, lossless when it applies) but
page.get_pixmap(clip=...) — rendering the region as a bitmap — is the
mechanism that actually has to carry this project's real PDFs, not an
edge-case fallback.

Never fabricates a region: if the target question's own marker word can't
be found on its page, this returns no candidates for that question at all
(the caller then shows "automatic detection failed" + manual upload —
never a guessed crop).

Per-option sub-band splitting (see `_sub_bands`) recognizes three option-
marker spellings — digit-dot ("1."/"2."/"3."/"4.", SSC's own scheme),
paren-letter ("(a)"/"(b)"/"(c)"/"(d)", AFCAT's scheme, matching parser.py's
own `LETTER_MARKER_RE`), and dot-letter ("a."/"b."/"c."/"d.") — tried in
that order per question, never mixed within one question. Once 4 markers
are resolved (in any one scheme), `_partition_grid` splits the region into
a true 2D grid of option bands (row-clustered by y, then column-split by x
within each row), which correctly handles a horizontal row, a vertical
stack, AND a 2-per-row grid — confirmed present in both real sample PDFs
(SSC: horizontal row; AFCAT: both a horizontal row and a genuine 2x2 grid
elsewhere in the same document).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Left-margin tolerance (points) for treating two word x0 positions as
# "the same indent column" — small enough that a question-number column
# and an option-number column (confirmed ~17pt apart on the SSC sample,
# 72pt vs 89pt) are never confused with each other, generous enough to
# absorb normal sub-pixel/kerning variation within the same column.
_SAME_COLUMN_TOLERANCE = 6.0

# A drawing/image bbox smaller than this (in points, either dimension) is
# treated as page furniture (a rule line, a bullet glyph, a border stroke)
# rather than a real diagram — avoids flagging every plain-text question
# that merely has a stray decorative line near it. This module is only
# ever invoked for questions parser.py already flagged `is_image`, so it
# doesn't need to be conservative about false positives on ordinary text
# questions — only about mistaking page furniture for the actual figure.
_MIN_VISUAL_DIMENSION = 8.0

# How far below the page's bottom margin we're willing to render when a
# question is the last one on its page and there's no "next marker" to
# bound the band — a generous fixed margin rather than the literal page
# edge, since footers/page numbers sometimes sit there.
_BOTTOM_MARGIN = 24.0

DESTINATIONS = ["question", "option_a", "option_b", "option_c", "option_d"]
_OPTION_DESTS = DESTINATIONS[1:]


@dataclass
class RegionCandidate:
    destination: str            # "question" | "option_a".."option_d" | "ambiguous"
    bbox: tuple                 # (x0, y0, x1, y1) in page points
    confidence: str              # "high" | "medium" | "needs_review"
    source_type: str             # "embedded_raster" | "pdf_region_render"
    reason: str = ""             # short human-readable note for the "why" of the confidence
    raster_xref: int | None = None


def _words_sorted(page) -> list[tuple]:
    """(x0,y0,x1,y1,text,block,line,word_no), already in PyMuPDF reading
    order (top-to-bottom, left-to-right within a line)."""
    return page.get_text("words")


def _find_marker(words: list[tuple], number: int, y_min: float) -> tuple | None:
    """
    Finds the word `"{number}."` on the page, at or after `y_min` (so a
    caller searching for "the NEXT question's marker" never matches
    something above the current question's own marker).

    Disambiguation from same-shaped option markers (parser.py's own
    docstring: "2." can be option B of one question OR question 2's own
    marker — numeric value alone never decides it): among all matches at
    or after y_min, the true question-number marker is the LEFTMOST one
    (real exam layouts flush question numbers to the page's own left
    margin and indent every option relative to it — confirmed on the SSC
    sample: question markers x0~=72, option markers x0~=89). If two
    candidates tie within `_SAME_COLUMN_TOLERANCE` of that leftmost x0,
    the match is genuinely ambiguous — returns None rather than guessing,
    same as parser.py's own "never fabricate" precedent for broken option
    sequences.
    """
    target = f"{number}."
    candidates = [w for w in words if w[4] == target and w[1] >= y_min - 0.5]
    if not candidates:
        return None
    candidates.sort(key=lambda w: (w[1], w[0]))  # top-to-bottom, then left-to-right
    best = min(candidates, key=lambda w: w[0])
    tied = [w for w in candidates if abs(w[0] - best[0]) <= _SAME_COLUMN_TOLERANCE and w is not best]
    if tied:
        return None
    return best


_DEGENERATE_EPS = 1.0  # points of slack when testing a zero-width/height rect's fixed coordinate against a band edge


def _rect_overlap_fraction(rect: tuple, band: tuple) -> float:
    """
    Fraction of `rect` that falls inside `band` (both (x0,y0,x1,y1)).

    Real finding from the SSC Q49/Q50 sample: PyMuPDF's get_drawings()
    represents a straight line stroke (e.g. one border of a hand-drawn
    grid/table diagram) as a rect with ZERO width or height — its own
    "area" is 0. The original area-ratio formula (intersection / rect's
    own area) divides by zero for every such stroke and always returns
    0.0, so EVERY line making up a thin-line diagram was silently
    discarded before ever being considered — a real diagram built from
    36+ hairline strokes (Q50's number grid) or 168+ strokes (Q49's two
    matrices) produced zero candidates, not a merged/oversized one.

    Degenerate rects are handled explicitly instead of by area ratio:
      - a point (zero width AND height): inside the band or not.
      - a vertical line (zero width): the line's fixed x must fall within
        the band's x-range; overlap is the FRACTION OF ITS LENGTH that
        falls within the band's y-range.
      - a horizontal line (zero height): symmetric.
    A normal 2D rect (real image/vector shape) still uses the original
    intersection-area / rect's-own-area ratio, unchanged — this is what
    every already-validated real case (SSC Q5, AFCAT Q52, the option-grid
    fixtures) relies on, so none of that behavior changes.
    """
    x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
    bx0, by0, bx1, by1 = band[0], band[1], band[2], band[3]
    w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)

    if w <= 0 and h <= 0:
        return 1.0 if (bx0 - _DEGENERATE_EPS <= x0 <= bx1 + _DEGENERATE_EPS
                        and by0 - _DEGENERATE_EPS <= y0 <= by1 + _DEGENERATE_EPS) else 0.0
    if w <= 0:
        if not (bx0 - _DEGENERATE_EPS <= x0 <= bx1 + _DEGENERATE_EPS):
            return 0.0
        iy0, iy1 = max(y0, by0), min(y1, by1)
        return max(0.0, iy1 - iy0) / h
    if h <= 0:
        if not (by0 - _DEGENERATE_EPS <= y0 <= by1 + _DEGENERATE_EPS):
            return 0.0
        ix0, ix1 = max(x0, bx0), min(x1, bx1)
        return max(0.0, ix1 - ix0) / w

    area = w * h
    ix0, iy0 = max(x0, bx0), max(y0, by0)
    ix1, iy1 = min(x1, bx1), min(y1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / area


def _union_bbox(rects: list[tuple]) -> tuple | None:
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


def _is_significant_bbox(bbox: tuple | None) -> bool:
    """
    Applied to the UNIONED bbox of everything attributed to one band, NOT
    to each individual rect beforehand — a real thin-line diagram (Q49/
    Q50) is made entirely of individually-degenerate strokes, none of
    which is "significant" alone, but whose UNION spans a real 2D area.
    A single genuinely stray decorative rule line's "union" is just that
    one line, still thin in one dimension, so it's still correctly
    excluded — this preserves the original page-furniture protection
    while adding thin-line-diagram detection.
    """
    if bbox is None:
        return False
    return (bbox[2] - bbox[0]) >= _MIN_VISUAL_DIMENSION and (bbox[3] - bbox[1]) >= _MIN_VISUAL_DIMENSION


def _detect_column_bounds(
    page, words: list[tuple], marker_x0: float, top: float, bottom: float
) -> tuple[float, float]:
    """
    Real finding from the SSC sample (page 1, Q5): this document uses a
    two-column layout, and a question's vertical band (top marker y to
    next marker y) can coincidentally overlap an ENTIRELY UNRELATED
    question's content in the other column at the same y-range (Q5's
    diagram band overlapped Q11/Q12's options one column over). A y-only
    band is therefore not safe by itself — this restricts it to the
    horizontal column that actually contains the target question's own
    marker.

    Heuristic: merge word horizontal extents into coverage intervals,
    SCOPED TO THIS QUESTION'S OWN [top, bottom] band only (not the whole
    page — across the full page height, some line somewhere almost always
    touches every x position, which would hide a real per-row gutter
    entirely), then look for the widest genuinely empty gap between two
    intervals that falls within the page's middle half. If no such gap is
    found, the page/region is treated as single-column and the full page
    width is returned unchanged.
    """
    page_w = page.rect.x1
    local = [w for w in words if top - 0.5 <= w[1] < bottom + 0.5]
    if not local:
        return (0.0, page_w)

    intervals = sorted((w[0], w[2]) for w in local)
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    mid_lo, mid_hi = page_w * 0.25, page_w * 0.75
    best_gap = None
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        width = next_start - prev_end
        mid = (prev_end + next_start) / 2
        if width >= 10.0 and mid_lo <= mid <= mid_hi:
            if best_gap is None or width > (best_gap[1] - best_gap[0]):
                best_gap = (prev_end, next_start)

    if best_gap is None:
        return (0.0, page_w)

    gutter = (best_gap[0] + best_gap[1]) / 2
    return (0.0, gutter) if marker_x0 < gutter else (gutter, page_w)


_ROW_EPS = 2.5  # y-difference (points) still considered "the same printed row"

# Option-marker spellings tried in this order — digit-dot first (SSC's own
# scheme, already validated against real data), then the two letter-mode
# spellings parser.py itself recognizes for letter-option papers (see
# parser.py's LETTER_MARKER_RE for "(a)".."(d)"; "a.".."d." covers the
# same convention without parens, per the user's explicit "or A./B./C./D."
# requirement). Matching is case-insensitive (word text is lowercased
# before comparison), so "(A)" and "(a)" are the same target. A paper only
# ever uses ONE of these consistently — schemes are tried in order and the
# first one that resolves all 4 markers wins; this never mixes markers
# from two different schemes within one question.
_OPTION_MARKER_SCHEMES: list[list[str]] = [
    ["1.", "2.", "3.", "4."],
    ["(a)", "(b)", "(c)", "(d)"],
    ["a.", "b.", "c.", "d."],
]


def _after(y: float, x: float, cursor_y: float, cursor_x: float) -> bool:
    if y > cursor_y + _ROW_EPS:
        return True
    if y < cursor_y - _ROW_EPS:
        return False
    return x > cursor_x


def _find_four_markers(in_band: list[tuple], top: float, marker_x: float, scheme: list[str]) -> dict:
    """
    Finds up to 4 option markers for ONE scheme, by pure READING-ORDER
    monotonicity (same-row = increasing x, next-row = increasing y beyond
    `_ROW_EPS`) — covers vertical stacks, horizontal rows, 2D grids, and
    ordinary 2-per-line wrapping uniformly, without needing to know in
    advance which layout a given page uses.

    Cursor starts AT the question's own opening marker position (not an
    unbounded "before everything" sentinel) — otherwise, when a question
    happens to be numbered "1" (or any small number that coincidentally
    looks like an option marker too), its own marker word would
    immediately satisfy the search for option marker "1." on the very
    first iteration (same y, same word) — a real bug caught by the
    synthetic raster fixture, where question 1's own "1." was being
    mistaken for option 1's marker.
    """
    found = {}
    cursor_y, cursor_x = top, marker_x
    for i, target in enumerate(scheme, start=1):
        candidates = [w for w in in_band if w[4].lower() == target and _after(w[1], w[0], cursor_y, cursor_x)]
        if not candidates:
            break
        candidates.sort(key=lambda w: (w[1], w[0]))
        m = candidates[0]
        found[i] = (m[1], m[0])
        cursor_y, cursor_x = m[1], m[0]
    return found


def _find_four_markers_two_pass(
    in_band: list[tuple], top: float, marker_x: float, scheme: list[str], x_lo: float, x_hi: float
) -> dict:
    """
    Pass 1: search restricted to the detected column (x_lo, x_hi) —
    this is what protects against the real SSC bug (a question's y-band
    coincidentally overlapping an UNRELATED question's content one page
    column over — see `_detect_column_bounds`'s docstring). If that alone
    resolves all 4 markers, it's used as-is; every already-validated real
    PDF case (SSC, AFCAT) resolves in this pass, so behavior there is
    completely unchanged.

    Pass 2 (only tried if pass 1 leaves the search incomplete but not
    empty): retries with markers OUTSIDE the column ALSO allowed, but
    ONLY if they sit on the same printed row (within `_ROW_EPS`) as a
    marker pass 1 already found. This is what makes a genuine SAME-
    question multi-column option GRID resolvable (e.g. options A/B side
    by side, C/D side by side below them) — its "columns" legitimately
    belong to one question, unlike the SSC bug's cross-column content,
    which sits at a materially different y and would never pass this row
    check. An empty pass-1 result never triggers pass 2 at all (no
    confirmed row to anchor a relaxed search against — stays conservative
    rather than searching the whole page blind).
    """
    restricted = [w for w in in_band if x_lo - 0.5 <= w[0] < x_hi + 0.5]
    found = _find_four_markers(restricted, top, marker_x, scheme)
    if len(found) == 4 or not found:
        return found

    # Anchor rows = every row WITHIN the column that carries any of this
    # scheme's marker tokens — not just the ones the sequential/break-on-
    # first-miss search above happened to record before giving up (that
    # search stops at the first unmatched option number, so it can miss
    # a later, still-genuinely-in-column row entirely — e.g. option 3
    # sitting two rows below a missing option 2 would never get recorded
    # by `found` alone, even though it's real, in-column content).
    scheme_set = set(scheme)
    anchor_ys = sorted({w[1] for w in restricted if w[4].lower() in scheme_set})
    relaxed_pool = [
        w for w in in_band
        if (x_lo - 0.5 <= w[0] < x_hi + 0.5) or any(abs(w[1] - ay) <= _ROW_EPS for ay in anchor_ys)
    ]
    relaxed = _find_four_markers(relaxed_pool, top, marker_x, scheme)
    return relaxed if len(relaxed) > len(found) else found


def _partition_grid(found: dict, x_lo: float, x_hi: float, page_x_hi: float, bottom: float) -> dict:
    """
    General 2D grid partition from exactly 4 resolved marker positions —
    replaces the old "either a single horizontal row OR a single vertical
    stack" special-casing, which broke on a genuine 2x2 grid (2 options
    per row, 2 rows): that case was falling into the vertical-stack branch
    and splitting by y ALONE at full column width, silently merging each
    row's two side-by-side options into one shared band. This handles all
    three shapes (row, stack, grid) uniformly by first clustering markers
    into rows (by y, within `_ROW_EPS`), then splitting each row by x
    using that row's own markers as column boundaries.

    Each row's final right boundary is the narrower detected-column x_hi
    UNLESS that row's own markers actually needed the two-pass search's
    relaxed pool (i.e. at least one marker in the row sits outside
    [x_lo, x_hi]) — only THEN does the row fall back to `page_x_hi` (the
    page's own right edge). This matters for two reasons: (1) using the
    narrow x_hi unconditionally would silently invert/degenerate a
    relaxed row's last band (right edge less than the marker's own x) —
    a real bug caught while building the synthetic grid-layout fixture
    for this fix; (2) using page_x_hi UNCONDITIONALLY would widen even an
    ordinary single-column row/stack's last band all the way to the page
    edge, reopening the original SSC cross-column bleed bug for exactly
    the row that never needed relaxing in the first place.
    """
    items = sorted(found.items(), key=lambda kv: (kv[1][0], kv[1][1]))  # (opt_num, (y,x)), reading order

    rows: list[list[tuple]] = []
    for opt_num, (y, x) in items:
        if rows and abs(y - rows[-1][0][1]) <= _ROW_EPS:
            rows[-1].append((opt_num, y, x))
        else:
            rows.append([(opt_num, y, x)])
    for row in rows:
        row.sort(key=lambda t: t[2])  # left-to-right within the row

    row_tops = [min(y for _, y, _ in row) for row in rows]
    row_bottoms = row_tops[1:] + [bottom]

    bands = {}
    for row, row_top, row_bottom in zip(rows, row_tops, row_bottoms):
        row_needed_relaxation = any(x < x_lo - 0.5 or x >= x_hi + 0.5 for _, _, x in row)
        row_right_edge = page_x_hi if row_needed_relaxation else x_hi
        x_bounds = [x for _, _, x in row] + [row_right_edge]
        for i, (opt_num, _y, _x) in enumerate(row):
            dest = _OPTION_DESTS[opt_num - 1]
            bands[dest] = (x_bounds[i], row_top, x_bounds[i + 1], row_bottom)
    return bands


def _sub_bands(
    words: list[tuple], top: float, bottom: float, x_lo: float, x_hi: float, marker_x: float,
    page_x_hi: float,
) -> tuple[dict, int]:
    """
    Splits the (x_lo, top, x_hi, bottom) region into a stem band and up to
    4 option bands — each a full (x0,y0,x1,y1) rect, not just a y-range,
    because real option layouts vary: vertical stacks, horizontal rows,
    and 2D grids (2 options per row, 2 rows) have all been confirmed in
    the real sample PDFs (the grid case in AFCAT itself).

    Tries each scheme in `_OPTION_MARKER_SCHEMES` in order; the first one
    that resolves all 4 markers is used. If none resolves all 4, the
    scheme that got the FARTHEST (most markers found) is reported, so the
    caller's 0-vs-partial-vs-4 signal stays meaningful regardless of which
    scheme a paper actually uses.

    Returns (bands, option_markers_found):
      - 4: a clean split — "question" (stem, everything above the option
        area) plus one rect per option.
      - 0: bands is just {"question": full region} — the whole region,
        options included, is presumed to be one image (an IMAGE_MCQ
        question has no extractable option markers in ANY recognized
        scheme — nothing to sub-divide against).
      - 1-3: a PARTIAL, inconsistent split under the best-matching scheme;
        the caller marks this ambiguous rather than guess which option
        the visual content belongs to.
    """
    in_band = [w for w in words if top - 0.5 <= w[1] < bottom]
    in_band.sort(key=lambda w: (w[1], w[0]))

    best_found: dict = {}
    for scheme in _OPTION_MARKER_SCHEMES:
        found = _find_four_markers_two_pass(in_band, top, marker_x, scheme, x_lo, x_hi)
        if len(found) > len(best_found):
            best_found = found
        if len(found) == 4:
            break  # first fully-resolved scheme wins — never mix schemes

    if len(best_found) < 4:
        return {"question": (x_lo, top, x_hi, bottom)}, len(best_found)

    option_bands = _partition_grid(best_found, x_lo, x_hi, page_x_hi, bottom)
    row_top = min(b[1] for b in option_bands.values())
    bands = {"question": (x_lo, top, x_hi, row_top), **option_bands}
    return bands, 4


def locate_image_regions(page, target_number: int, next_number_on_page: int | None) -> list[RegionCandidate]:
    """
    Main entry point. `page` is a fitz.Page already opened from the
    document containing this question. `target_number`/`next_number_on_page`
    are question numbers already known to appear on this page (from
    parser.py's `source_page` + the paper's `question_order`) — this
    function never invents a question boundary, only locates one it's
    already been told exists.

    Returns [] if the target question's own marker can't be confidently
    located — the caller must treat that as "automatic detection failed",
    never render a guessed full-page fallback silently.
    """
    words = _words_sorted(page)
    start = _find_marker(words, target_number, y_min=0)
    if start is None:
        return []

    top = start[1]
    if next_number_on_page is not None:
        end = _find_marker(words, next_number_on_page, y_min=top)
        bottom = end[1] if end is not None else min(page.rect.y1 - _BOTTOM_MARGIN, top + 500)
        band_confidence_ok = end is not None
    else:
        bottom = page.rect.y1 - _BOTTOM_MARGIN
        band_confidence_ok = True  # last question on page — bottom margin is a real boundary, not a guess

    if bottom <= top:
        return []

    x_lo, x_hi = _detect_column_bounds(page, words, start[0], top, bottom)

    # Marker-finding (via _sub_bands -> _find_four_markers_two_pass) sees
    # the FULL word list, not a column-pre-filtered one — a genuine
    # same-question multi-column option GRID needs markers from both
    # sides to be found (see _find_four_markers_two_pass's docstring).
    # Column restriction is applied below, per-band, only where it's
    # actually needed.
    bands, option_markers_found = _sub_bands(words, top, bottom, x_lo, x_hi, start[0], page.rect.x1)

    def _in_column(rect: tuple) -> bool:
        center = (rect[0] + rect[2]) / 2
        return x_lo - 1.0 <= center <= x_hi + 1.0

    raster_rects = []
    for im in page.get_images(full=True):
        try:
            bbox = page.get_image_bbox(im)
        except Exception:
            continue
        raster_rects.append((bbox.x0, bbox.y0, bbox.x1, bbox.y1, im[0]))

    drawing_rects = [d["rect"] for d in page.get_drawings() if "rect" in d]
    drawing_rects = [(r.x0, r.y0, r.x1, r.y1) for r in drawing_rects]

    candidates: list[RegionCandidate] = []
    clean_split = option_markers_found == 4

    for dest, band_rect in bands.items():
        # The "question" destination (the stem area, or the whole-region
        # fallback when <4 options were resolved) is NOT a tightly-scoped
        # grid cell the way an option band is — it's exactly the shape of
        # region the real SSC bug bled into from a neighboring page
        # column, so it alone still gets column-restricted candidates.
        # Option bands come from _partition_grid's own tight per-cell
        # geometry (already validated not to need this — an unrelated
        # rect from a genuinely different page column essentially never
        # overlaps >50% of a tight option cell).
        candidate_raster = [r for r in raster_rects if (dest != "question" or _in_column(r))]
        candidate_drawing = [r for r in drawing_rects if (dest != "question" or _in_column(r))]

        band_raster = [r for r in candidate_raster if _rect_overlap_fraction(r, band_rect) > 0.5]
        # NOT pre-filtered for "significance" per-rect — a thin-line
        # diagram (Q49/Q50) is made of individually-degenerate strokes;
        # only their UNION is checked for significance below.
        band_drawing = [r for r in candidate_drawing if _rect_overlap_fraction(r, band_rect) > 0.5]

        if band_raster:
            union = _union_bbox([r[:4] for r in band_raster] + band_drawing)
            confidence = "high" if (band_confidence_ok and clean_split) else "medium"
            candidates.append(RegionCandidate(
                destination=dest,
                bbox=union,
                confidence=confidence,
                source_type="embedded_raster",
                reason=f"{len(band_raster)} embedded image(s) intersecting this band",
                raster_xref=band_raster[0][4] if len(band_raster) == 1 else None,
            ))
        elif band_drawing:
            union = _union_bbox(band_drawing)
            if _is_significant_bbox(union):
                confidence = "high" if (band_confidence_ok and clean_split) else "medium"
                candidates.append(RegionCandidate(
                    destination=dest,
                    bbox=union,
                    confidence=confidence,
                    source_type="pdf_region_render",
                    reason=f"{len(band_drawing)} vector drawing element(s) (thin-stroke diagram union) intersecting this band",
                ))

    if not candidates:
        return []

    if 0 < option_markers_found < 4:
        # A PARTIAL split: some options had real text/markers and some
        # apparently didn't, so the visual content's true destination
        # (which specific option, or the question itself) can't be
        # trusted — surface as ambiguous rather than guess which one.
        c = candidates[0]
        candidates = [RegionCandidate(
            destination="ambiguous",
            bbox=c.bbox,
            confidence="needs_review",
            source_type=c.source_type,
            reason=f"only {option_markers_found}/4 option markers were found on this "
                   f"page, so the visual content's destination (question vs. a "
                   f"specific option) could not be determined automatically",
        )]
    elif not band_confidence_ok:
        # Bottom boundary itself was a guess (no next-question marker found
        # despite one being expected) — downgrade every candidate rather
        # than trust an unverified band edge.
        candidates = [
            RegionCandidate(
                destination=c.destination, bbox=c.bbox, confidence="needs_review",
                source_type=c.source_type,
                reason=c.reason + "; next question's marker was not found, so the "
                                   "bottom edge of this region is an estimate",
            )
            for c in candidates
        ]

    return candidates
