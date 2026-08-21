"""
math_reconstructor.py — Geometry-driven mathematical-content cleanup.

This module sits BETWEEN raw PDF text extraction and everything downstream
(parser.py's question/option/explanation extraction, then latex_processor.py):

    RAW PDF
      -> page.get_text() native extraction   (extractor.py, unchanged)
      -> reconstruct_math() (THIS MODULE)    <-- new stage
      -> clean text
      -> parser.py question/option/explanation extraction (unchanged)
      -> latex_processor.py LaTeX conversion (unchanged, extended for π/frac)
      -> Streamlit / excel_writer.py

WHY THIS EXISTS
----------------
Some PDFs (real example: an equation authored in Microsoft Equation Editor /
MathType and embedded as an OLE object, then exported to PDF) render a
fraction like 22/7 as TWO separately-positioned lines of text — a numerator
glyph run sitting above a denominator glyph run, connected only by a drawn
horizontal rule (the fraction bar), with NO textual "/" character anywhere.
PyMuPDF's plain page.get_text() has no notion of "these two lines are one
fraction" — it just emits the numerator's line, then the denominator's line,
so latex_processor.py (which only ever sees flat text) has no way to tell
that "22" followed by "7" on the next line is a fraction rather than, say,
"Question 22" followed by "Option 7".

Likewise, large multi-piece "stretchy" parentheses (used to bracket a whole
fraction) are typeset as 3 separate glyphs per side (top/middle/bottom) in a
dedicated math symbol font (observed here as "EuclidSymbol"), using Private
Use Area codepoints. Naively deleting them would violate the "never
fabricate, never blindly delete real math notation" rule; naively KEEPING
them renders as literal tofu-box artifacts in Streamlit/Excel.

This module uses the page's actual geometry (PyMuPDF's `dict` text spans AND
`get_drawings()` vector paths) to confidently reconstruct exactly these
patterns, and ONLY these patterns — verified against a fraction bar (an
actual drawn horizontal rule at the boundary) rather than mere textual
adjacency, per the "never fabricate a fraction from adjacent numbers"
requirement. Every reconstruction step is independently guarded and the
whole pipeline fails open: if anything about a page's geometry can't be
read, or a candidate doesn't pass its confidence check, the ORIGINAL text
for that page is returned completely unchanged.

Beyond isolated fractions/superscripts/brackets, section 5 below
(`reconstruct_equation_blocks`) goes further for a genuinely multi-part
equation-editor object (e.g. "Volume = (22/7) × (3.5)² × 8 = ..."): it
groups the object's spans by real (x, y) position into visual rows, sorts
each row left-to-right, and rebuilds ONE coherent linear expression —
something no amount of patching the already-scrambled flat text can do,
since PyMuPDF's own internal ordering for such an object doesn't follow
visual reading order at all. It only replaces a region when the rebuild
passes several independent confidence checks; a rejected region is left
byte-for-byte untouched, and the simpler patches below still get a chance
to fix whatever they can from it.

This module has no dependency on parser.py/latex_processor.py — it takes a
fitz.Page and its already-extracted plain text, and returns cleaned plain
text. It never emits LaTeX itself (with ONE deliberate, narrow exception:
a confirmed fraction is written inline as ``\\frac{a}{b}`` — a literal,
unambiguous plain-text marker meaning "this specific span is confidently a
fraction" that latex_processor.py already knows how to recognize as a
strong math token and wrap in ``$...$`` alongside the rest of its normal
detection logic). Everything else this module produces (merged decimal
points, plain "(" / ")" for collapsed stretchy-parenthesis glyphs, unicode
super/subscript characters for geometrically-confirmed exponents) is still
just plain text — latex_processor.py, unmodified in its core logic, is what
turns THOSE into LaTeX, exactly as it already does for hand-typed unicode
superscripts.
"""

from __future__ import annotations

import re
from collections import Counter

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Known font-artifact codepoints
# ---------------------------------------------------------------------------

# Microsoft Equation Editor / MathType's "Euclid Symbol" / "MT Extra" font
# uses these Private Use Area codepoints for the three vertically-stacked
# pieces of a large ("stretchy") parenthesis, used to bracket a fraction or
# a tall expression. Confirmed against the real PDF that exposed this bug
# (templates/AFCAT_Mock_1-5_SP_2026_Final_File.pdf, page 13, Q39's
# explanation) via span font name + stacked bbox positions. This is a
# standard, well-known convention for that font family — not something
# specific to one document — so recognizing exactly these codepoints (and
# nothing else in the PUA range) is a font-glyph mapping, not a guess.
_OPEN_PAREN_PIECES = ""   # top / middle / bottom of "("
_CLOSE_PAREN_PIECES = ""  # top / middle / bottom of ")"
_PAREN_PIECE_FONT_HINT_RE = re.compile(r"euclid|symbol|mtextra", re.I)

_OPEN_RUN_RE = re.compile(
    r"[" + _OPEN_PAREN_PIECES + r"](?:[ \t\n]*[" + _OPEN_PAREN_PIECES + r"])*"
)
_CLOSE_RUN_RE = re.compile(
    r"[" + _CLOSE_PAREN_PIECES + r"](?:[ \t\n]*[" + _CLOSE_PAREN_PIECES + r"])*"
)

# A numerator/denominator "atom" simple enough to be confidently a fraction
# component — a plain (possibly decimal) number, or a single-letter
# variable. Anything else (multi-letter words, punctuation, ...) is NOT
# treated as a fraction, per the "never fabricate" rule.
_ATOM_RE = re.compile(r"^-?\d+(?:\.\d+)?$|^[A-Za-z]$")

_SUP_DIGIT_MAP = {d: c for d, c in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")}
_SUB_DIGIT_MAP = {d: c for d, c in zip("0123456789", "₀₁₂₃₄₅₆₇₈₉")}
_SCRIPT_BASE_CHAR_RE = re.compile(r"[A-Za-z0-9)\]]")


def _iter_spans(dict_page: dict):
    for block in dict_page.get("blocks", ()):
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                yield span


# ---------------------------------------------------------------------------
# 1. Fraction reconstruction — driven by an actual drawn fraction bar, not
#    mere line adjacency.
# ---------------------------------------------------------------------------

def _find_isolated_thin_bars(page, max_width: float = 120.0, max_height: float = 1.5):
    """Thin, short horizontal drawn lines — candidate fraction bars.
    Excludes wide rules (table/column dividers) and, via the isolation
    filter below, repeating grid lines (a coding-decoding matrix or a
    multi-column answer table draws MANY same-width bars; a genuine
    fraction bar is a one-off for its exact x-span on the page).

    Decomposes each drawing into its individual line segments (not just
    the path's overall bounding `rect`) — two unrelated fraction bars on
    the same page can be recorded as ONE MuPDF drawing object with two
    disjoint 'l' (line) items, whose combined bounding rect would
    incorrectly span both bars at once and defeat the numerator/
    denominator matching below (confirmed against the real PDF: the
    Q39 explanation's two "22/7" fraction bars are exactly this case)."""
    candidates = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for d in drawings:
        for item in d.get("items", ()):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            x0, x1 = sorted((p1.x, p2.x))
            y0, y1 = sorted((p1.y, p2.y))
            w, h = x1 - x0, y1 - y0
            if h <= max_height and 2.0 < w <= max_width:
                candidates.append(fitz.Rect(x0, y0, x1, y1))

    counts: dict[tuple[int, int], int] = {}
    for r in candidates:
        key = (round(r.x0), round(r.x1))
        counts[key] = counts.get(key, 0) + 1
    isolated = [r for r in candidates if counts[(round(r.x0), round(r.x1))] <= 2]
    isolated.sort(key=lambda r: (round(r.y0, 1), round(r.x0, 1)))
    return isolated


def _match_fraction_for_bar(dict_page: dict, bar) -> tuple[str, str] | None:
    """Finds the span directly touching the bar from above (numerator) and
    directly touching it from below (denominator), requiring both to be
    simple atoms and to horizontally overlap the bar substantially. Returns
    None (never guesses) if either side is missing or not a simple atom."""
    y = (bar.y0 + bar.y1) / 2.0
    x0, x1 = bar.x0, bar.x1
    bar_width = x1 - x0
    if bar_width <= 0:
        return None

    above = below = None
    for span in _iter_spans(dict_page):
        sx0, sy0, sx1, sy1 = span["bbox"]
        overlap = min(sx1, x1) - max(sx0, x0)
        if overlap < 0.6 * min(sx1 - sx0, bar_width):
            continue
        if abs(sy1 - y) <= 2.5 and sy1 <= y + 0.5:
            above = span
        elif abs(sy0 - y) <= 2.5 and sy0 >= y - 0.5:
            below = span

    if above is None or below is None:
        return None
    num, den = above["text"].strip(), below["text"].strip()
    if not (_ATOM_RE.match(num) and _ATOM_RE.match(den)):
        return None
    return num, den


def _find_fractions(page, dict_page: dict) -> list[tuple[str, str]]:
    fractions = []
    for bar in _find_isolated_thin_bars(page):
        match = _match_fraction_for_bar(dict_page, bar)
        if match:
            fractions.append(match)
    return fractions


def _apply_fraction_substitutions(text: str, fractions: list[tuple[str, str]]) -> str:
    """Sequentially replaces each geometrically-confirmed "NUM<gap>DEN"
    occurrence with a literal ``\\frac{NUM}{DEN}`` marker, scanning strictly
    forward so repeated identical pairs elsewhere on the page (before the
    current search point) are never touched twice or out of order."""
    cursor = 0
    for num, den in fractions:
        pattern = re.compile(
            rf"(?<!\d){re.escape(num)}[ \t]*\n[ \t]*{re.escape(den)}(?!\d)"
        )
        m = pattern.search(text, cursor)
        if not m:
            continue
        replacement = f"\\frac{{{num}}}{{{den}}}"
        text = text[: m.start()] + replacement + text[m.end() :]
        cursor = m.start() + len(replacement)
    return text


# ---------------------------------------------------------------------------
# 2. Decimal-point reconstruction — "3 5" + a stray lone "." elsewhere ->
#    "3.5". PyMuPDF occasionally emits a decimal point as its own
#    disconnected span (confirmed: identical bbox y-range, x-centered in the
#    gap between the two digit runs) that then surfaces elsewhere in the
#    flattened text as a line containing only ".". Only merges when BOTH
#    the digit-space-digit shape AND a geometrically-matching lone period
#    are present — never merges a coincidental "3 5" that has no such period.
# ---------------------------------------------------------------------------

_SPLIT_DECIMAL_TEXT_RE = re.compile(r"^(\d+) (\d+)$")


def _find_decimal_merges(dict_page: dict) -> list[tuple[str, str]]:
    spans = list(_iter_spans(dict_page))
    periods = [s for s in spans if s["text"].strip() == "."]
    merges = []
    for s in spans:
        m = _SPLIT_DECIMAL_TEXT_RE.match(s["text"])
        if not m:
            continue
        sx0, sy0, sx1, sy1 = s["bbox"]
        for p in periods:
            px0, py0, px1, py1 = p["bbox"]
            pcx = (px0 + px1) / 2.0
            y_overlap = min(sy1, py1) - max(sy0, py0)
            if sx0 < pcx < sx1 and y_overlap > 0.5 * (sy1 - sy0):
                merges.append((m.group(1), m.group(2)))
                break
    return merges


_LONE_PERIOD_RE = re.compile(r"(?:^|\n)\.(?=\n|$)")


def _apply_decimal_merges(text: str, merges: list[tuple[str, str]]) -> str:
    cursor = 0
    for int_part, frac_part in merges:
        pattern = re.compile(rf"(?<!\d){re.escape(int_part)} {re.escape(frac_part)}(?!\d)")
        m = pattern.search(text, cursor)
        if not m:
            continue
        replacement = f"{int_part}.{frac_part}"
        text = text[: m.start()] + replacement + text[m.end() :]
        cursor = m.start() + len(replacement)

    # Remove exactly one stray standalone "." per successful merge above —
    # the orphaned decimal-point span that surfaced on its own line.
    removed = 0
    search_from = 0
    while removed < len(merges):
        m = _LONE_PERIOD_RE.search(text, search_from)
        if not m:
            break
        dot_pos = text.index(".", m.start())
        text = text[:dot_pos] + text[dot_pos + 1 :]
        search_from = dot_pos
        removed += 1
    return text


# ---------------------------------------------------------------------------
# 3. Stretchy-parenthesis glyph collapse — only when the font confirms it.
# ---------------------------------------------------------------------------

def _page_uses_paren_piece_font(dict_page: dict) -> bool:
    for span in _iter_spans(dict_page):
        text = span.get("text", "")
        if any(c in _OPEN_PAREN_PIECES or c in _CLOSE_PAREN_PIECES for c in text):
            if _PAREN_PIECE_FONT_HINT_RE.search(span.get("font", "")):
                return True
    return False


def _collapse_bracket_pieces(text: str) -> str:
    text = _OPEN_RUN_RE.sub("(", text)
    text = _CLOSE_RUN_RE.sub(")", text)
    return text


# ---------------------------------------------------------------------------
# 4. Superscript / subscript reconstruction from real span geometry.
# ---------------------------------------------------------------------------

def _find_script_pairs(dict_page: dict) -> list[tuple[str, str, str]]:
    """Detects a genuinely-raised-or-lowered, reduced-size digit run glued
    (no gap) to the immediately preceding span within the same line —
    PyMuPDF's actual encoding of an exponent/subscript in this document
    (confirmed via span size ratio + vertical origin offset, NOT merely
    "a digit follows a letter"). Returns (base_char, digits, "super"|"sub")
    triples in geometric document order."""
    pairs: list[tuple[str, str, str]] = []
    for block in dict_page.get("blocks", ()):
        for line in block.get("lines", ()):
            spans = line.get("spans", ())
            for i in range(len(spans) - 1):
                base, script = spans[i], spans[i + 1]
                base_text, script_text = base.get("text", ""), script.get("text", "")
                if not base_text or not re.fullmatch(r"\d{1,2}", script_text):
                    continue
                if not _SCRIPT_BASE_CHAR_RE.match(base_text[-1]):
                    continue
                base_size = base.get("size") or 0
                script_size = script.get("size") or 0
                if base_size <= 0 or script_size / base_size > 0.85:
                    continue
                gap = script["bbox"][0] - base["bbox"][2]
                if gap < -0.5 or gap > 1.0:
                    continue
                base_mid = (base["bbox"][1] + base["bbox"][3]) / 2.0
                script_mid = (script["bbox"][1] + script["bbox"][3]) / 2.0
                if script_mid < base_mid - 1:
                    direction = "super"
                elif script_mid > base_mid + 1:
                    direction = "sub"
                else:
                    continue
                pairs.append((base_text[-1], script_text, direction))
    return pairs


def _apply_script_substitutions(text: str, pairs: list[tuple[str, str, str]]) -> str:
    cursor = 0
    for base_char, digits, direction in pairs:
        table = _SUP_DIGIT_MAP if direction == "super" else _SUB_DIGIT_MAP
        try:
            converted = "".join(table[d] for d in digits)
        except KeyError:
            continue
        pattern = re.compile(rf"{re.escape(base_char)}{re.escape(digits)}(?!\d)")
        m = pattern.search(text, cursor)
        if not m:
            continue
        text = text[: m.start()] + base_char + converted + text[m.end() :]
        cursor = m.start() + len(base_char) + len(converted)
    return text


# ---------------------------------------------------------------------------
# 5. Full equation-block reconstruction.
#
# Sections 1-4 above patch the ALREADY-FLATTENED, scrambled plain text in
# place — safe and precise for an isolated fraction/superscript, but
# powerless against a genuinely multi-part equation-editor object (e.g.
# "Volume = (22/7) × (3.5)² × 8 = (22/7 × 12.25) × 8"), where PyMuPDF's own
# internal line ordering for the OLE object doesn't match the visual
# reading order at all — the surrounding "(", ")", "×", "=" tokens come out
# interleaved in an order no text-level patch can safely re-sequence.
#
# This section instead REBUILDS such a block from scratch, purely from
# geometry: every span's real (x, y) position, plus the same fraction-bar /
# stretchy-parenthesis-font / superscript-size-and-offset evidence used
# above, drives a fresh left-to-right, top-to-bottom reconstruction. The
# result is spliced back over the ORIGINAL scrambled text (never rebuilt
# for the rest of the page) only when it passes several independent
# confidence checks (see _confident_equation_text) — otherwise that one
# region is left byte-for-byte as extracted, exactly as if this section
# didn't run, and sections 1-4 above still get a chance to patch whatever
# they can from the untouched text (e.g. a lone fraction in a rejected
# region is still fixed by section 1).
# ---------------------------------------------------------------------------

class _EqAtom:
    """One positioned piece of an equation block being reconstructed —
    either straight from a PyMuPDF span, or a merged group (a fraction, a
    collapsed stretchy-parenthesis piece run). `kind` is "text" | "frac" |
    "paren" | "superscript"."""

    __slots__ = ("text", "x0", "x1", "y0", "y1", "size", "font", "kind")

    def __init__(self, text, x0, x1, y0, y1, size, font, kind="text"):
        self.text = text
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.size = size
        self.font = font
        self.kind = kind


_PUA_PIECE_RUN_RE = re.compile(
    r"^([" + _OPEN_PAREN_PIECES + _CLOSE_PAREN_PIECES + r"]+)(.*)$", re.S
)
_FORCE_SPACE_TOKENS = {"×", "÷", "≤", "≥", "≠", "±", "="}
_EQ_BODY_SIZE = 9.0  # dominant body-text size in the real PDF; superscripts
# are detected relative to whatever the base span's own size is (see
# _mark_eq_superscripts), this constant is only a last-resort default.


def _split_pua_atoms(spans) -> list[_EqAtom]:
    """One PyMuPDF span occasionally carries a PUA stretchy-bracket-piece
    run directly glued to trailing ordinary content in the SAME span (e.g.
    the real PDF's close-piece-run immediately followed by "×", or by a
    small ASCII "("). Splits those apart (proportionally, by character
    count, across the span's bbox width) so each piece can be matched and
    merged independently below — the trailing "×"/"(" must never be
    silently absorbed into a collapsed paren token."""
    atoms: list[_EqAtom] = []
    for s in spans:
        text = s.get("text", "")
        if not text:
            continue
        x0, y0, x1, y1 = s["bbox"]
        width = x1 - x0
        m = _PUA_PIECE_RUN_RE.match(text)
        if m and m.group(2):
            piece, rest = m.group(1), m.group(2)
            cut = x0 + width * (len(piece) / len(text))
            atoms.append(_EqAtom(piece, x0, cut, y0, y1, s["size"], s["font"]))
            atoms.append(_EqAtom(rest, cut, x1, y0, y1, s["size"], s["font"]))
        else:
            atoms.append(_EqAtom(text, x0, x1, y0, y1, s["size"], s["font"]))
    return atoms


def _evidence_block_indices(dict_page: dict, bars) -> set[int]:
    """Block indices containing actual equation-editor evidence: a
    font-confirmed stretchy-parenthesis-piece span, or a span directly
    touching a fraction bar. Ordinary prose blocks never match either
    condition, so they're never considered for reconstruction at all."""
    evidence: set[int] = set()
    for bi, block in enumerate(dict_page.get("blocks", ())):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                text = s.get("text", "")
                if (
                    text
                    and all(c in _OPEN_PAREN_PIECES + _CLOSE_PAREN_PIECES for c in text)
                    and _PAREN_PIECE_FONT_HINT_RE.search(s.get("font", ""))
                ):
                    evidence.add(bi)
    for bar in bars:
        x0, x1 = bar.x0, bar.x1
        y = (bar.y0 + bar.y1) / 2.0
        bar_width = x1 - x0
        if bar_width <= 0:
            continue
        for bi, block in enumerate(dict_page.get("blocks", ())):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for s in line["spans"]:
                    sx0, sy0, sx1, sy1 = s["bbox"]
                    overlap = min(sx1, x1) - max(sx0, x0)
                    if overlap < 0.6 * min(sx1 - sx0, bar_width):
                        continue
                    if abs(sy1 - y) <= 2.5 or abs(sy0 - y) <= 2.5:
                        evidence.add(bi)
    return evidence


def _cluster_consecutive(indices: set[int]) -> list[list[int]]:
    """Merges consecutive block INDICES into runs. PyMuPDF's own block
    order already matches the page's paragraph order, so a contiguous run
    of evidence-bearing block indices is exactly one equation-editor
    object's worth of blocks (confirmed against the real PDF: Q39's
    "Volume = ..." derivation is blocks 15-19)."""
    if not indices:
        return []
    ordered = sorted(indices)
    clusters = [[ordered[0]]]
    for i in ordered[1:]:
        if i == clusters[-1][-1] + 1:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _eq_merge_decimals(atoms: list[_EqAtom]) -> list[_EqAtom]:
    """Same evidence rule as _find_decimal_merges (a split "D D" span plus
    a geometrically-matching lone "." span), applied to atoms already
    gathered for one equation-block reconstruction."""
    periods = [a for a in atoms if a.text == "."]
    used = set()
    for a in atoms:
        m = _SPLIT_DECIMAL_TEXT_RE.match(a.text)
        if not m:
            continue
        for p in periods:
            if id(p) in used:
                continue
            pcx = (p.x0 + p.x1) / 2.0
            y_overlap = min(a.y1, p.y1) - max(a.y0, p.y0)
            if a.x0 < pcx < a.x1 and y_overlap > 0.5 * (a.y1 - a.y0):
                a.text = f"{m.group(1)}.{m.group(2)}"
                used.add(id(p))
                break
    return [a for a in atoms if id(a) not in used]


def _eq_merge_fractions(atoms: list[_EqAtom], bars) -> list[_EqAtom]:
    """Same fraction-bar evidence rule as _match_fraction_for_bar, but
    merges the matched numerator/denominator atoms into one `frac`-kind
    _EqAtom carrying their combined bbox, for row/x sorting below."""
    used = set()
    for bar in bars:
        x0, x1 = bar.x0, bar.x1
        y = (bar.y0 + bar.y1) / 2.0
        bar_width = x1 - x0
        above = below = None
        for a in atoms:
            if id(a) in used:
                continue
            overlap = min(a.x1, x1) - max(a.x0, x0)
            if overlap < 0.6 * min(a.x1 - a.x0, bar_width):
                continue
            if abs(a.y1 - y) <= 2.5 and a.y1 <= y + 0.5:
                above = a
            elif abs(a.y0 - y) <= 2.5 and a.y0 >= y - 0.5:
                below = a
        if above is None or below is None:
            continue
        num, den = above.text.strip(), below.text.strip()
        if not (_ATOM_RE.match(num) and _ATOM_RE.match(den)):
            continue
        merged = _EqAtom(
            f"\\frac{{{num}}}{{{den}}}",
            min(above.x0, below.x0), max(above.x1, below.x1),
            min(above.y0, below.y0), max(above.y1, below.y1),
            above.size, above.font, kind="frac",
        )
        used.add(id(above))
        used.add(id(below))
        atoms.append(merged)
    return [a for a in atoms if id(a) not in used]


def _eq_merge_paren_pieces(atoms: list[_EqAtom]) -> list[_EqAtom]:
    """Collapses each stretchy-parenthesis's 3 pieces (top/middle/bottom,
    sharing the same x-column — confirmed against the real PDF) into ONE
    plain "(" or ")" atom spanning their combined bbox."""
    pieces = [
        a for a in atoms
        if a.text and all(c in _OPEN_PAREN_PIECES + _CLOSE_PAREN_PIECES for c in a.text)
    ]
    groups: dict[int, list[_EqAtom]] = {}
    for a in pieces:
        groups.setdefault(round(a.x0), []).append(a)

    used = set()
    for group in groups.values():
        is_open = all(c in _OPEN_PAREN_PIECES for a in group for c in a.text)
        is_close = all(c in _CLOSE_PAREN_PIECES for a in group for c in a.text)
        if not (is_open or is_close):
            continue  # mixed/ambiguous column — never guess, leave as-is
        ch = "(" if is_open else ")"
        merged = _EqAtom(
            ch,
            min(a.x0 for a in group), max(a.x1 for a in group),
            min(a.y0 for a in group), max(a.y1 for a in group),
            group[0].size, group[0].font, kind="paren",
        )
        for a in group:
            used.add(id(a))
        atoms.append(merged)
    return [a for a in atoms if id(a) not in used]


def _eq_mark_superscripts(atoms: list[_EqAtom]) -> None:
    """Flags atoms that are themselves a short, reduced-size digit run as
    superscripts (in place) — the SAME size/shape evidence as
    _find_script_pairs, but not requiring the base to be the immediately
    adjacent dict span (equation-block spans are frequently NOT adjacent
    in PyMuPDF's own scrambled span order, even though they clearly are
    geometrically): a real exponent here is simply much smaller than the
    surrounding body text."""
    for a in atoms:
        if re.fullmatch(r"\d{1,2}", a.text) and a.size and a.size < 0.85 * _EQ_BODY_SIZE:
            a.kind = "superscript"


def _build_eq_rows(atoms: list[_EqAtom]) -> list[list[_EqAtom]]:
    """Clusters atoms into visual rows by y-center proximity (a fraction's
    combined bbox, and a superscript's raised bbox, both still cluster
    into their surrounding row this way since the row-height tolerance is
    wider than either offset), then sorts each row left-to-right by x —
    the actual visual reading order, independent of whatever order
    PyMuPDF's own span/line iteration happened to produce."""
    ordered = sorted(atoms, key=lambda a: (a.y0 + a.y1) / 2.0)
    rows: list[list[_EqAtom]] = []
    for a in ordered:
        cy = (a.y0 + a.y1) / 2.0
        for row in rows:
            row_cy = sum((x.y0 + x.y1) / 2.0 for x in row) / len(row)
            if abs(cy - row_cy) < 9:
                row.append(a)
                break
        else:
            rows.append([a])
    for row in rows:
        row.sort(key=lambda a: a.x0)
    return rows


def _rows_have_x_overlap(rows: list[list[_EqAtom]]) -> bool:
    """True if two non-superscript atoms placed in the same row genuinely
    overlap in x rather than merely sitting close together — a sign the
    y-proximity row clustering has conflated two distinct visual lines
    into one, which happens on some of this document's more elaborate
    equation objects (confirmed: a 4-option answer-choice row of
    independent fractions on the tank question's own stem page). This is
    the confidence check that stops such cases from being guessed at."""
    for row in rows:
        plain = [a for a in row if a.kind != "superscript"]
        for i in range(len(plain)):
            for j in range(i + 1, len(plain)):
                a, b = plain[i], plain[j]
                if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.0:
                    return True
    return False


def _assemble_eq_row(row: list[_EqAtom]) -> str:
    """Turns one visually-sorted row of atoms into a single text line:
    glues a superscript onto its immediately preceding atom as a real
    unicode superscript character (so the UNMODIFIED latex_processor.py
    superscript detector picks it up exactly as it already does for
    hand-typed superscripts), drops a stretchy-parenthesis pair that
    directly wraps nothing but a single \\frac{}{} (those parens are
    purely a font-rendering artifact sizing the bracket to the fraction's
    height — \\frac{}{} is already visually self-delimiting, so keeping
    them would be redundant decoration, not lost meaning), forces a space
    around genuine operators regardless of the original PDF's often-tight
    kerning, and starts a new line before a subsequent bare "=" only once
    a binary operator has already appeared on this line (a real multi-step
    derivation like "Volume = ... × 8 = ... × 8", vs. a simple one-line
    equality like "π = 22/7", which must stay on one line)."""
    merged: list[_EqAtom] = []
    for a in row:
        if a.kind == "superscript" and merged:
            digits = "".join(_SUP_DIGIT_MAP.get(d, d) for d in a.text.strip())
            merged[-1].text = merged[-1].text.rstrip() + digits
        else:
            merged.append(a)

    collapsed: list[_EqAtom] = []
    i = 0
    while i < len(merged):
        if (
            merged[i].kind == "paren" and merged[i].text == "("
            and i + 2 < len(merged)
            and merged[i + 1].kind == "frac"
            and merged[i + 2].kind == "paren" and merged[i + 2].text == ")"
        ):
            collapsed.append(merged[i + 1])
            i += 3
        else:
            collapsed.append(merged[i])
            i += 1

    pieces: list[str] = []
    prev: _EqAtom | None = None
    seen_operator_since_break = False
    for a in collapsed:
        orig = a.text
        text = orig.strip()
        if not text or text == "\t":
            continue
        if prev is None:
            pieces.append(text)
        elif text == "=" and pieces and seen_operator_since_break:
            pieces.append("\n" + text)
            seen_operator_since_break = False
        else:
            gap = a.x0 - prev.x1
            need_space = (
                text in _FORCE_SPACE_TOKENS
                or prev.text.strip() in _FORCE_SPACE_TOKENS
                or prev.text.endswith(" ")
                or orig.startswith(" ")
                or gap >= 1.5
            )
            pieces.append((" " if need_space else "") + text)
        if text in ("×", "÷"):
            seen_operator_since_break = True
        prev = a
    return "".join(pieces)


_EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
_BARE_FRAGMENT_RE = re.compile(r"^[\d.]{1,3}$")


def _confident_equation_text(dict_page: dict, bars, block_indices: list[int]) -> str | None:
    """Reconstructs one equation-block cluster and returns the clean text
    ONLY if it passes every confidence check below; otherwise returns None
    (caller leaves that region of the page completely untouched — per the
    "never guess when geometry is ambiguous" requirement, an imperfect
    reconstruction is worse than none)."""
    spans = []
    for bi in block_indices:
        block = dict_page["blocks"][bi]
        for line in block.get("lines", ()):
            spans.extend(line["spans"])
    if not spans:
        return None

    atoms = _split_pua_atoms(spans)
    original_alnum = sum(1 for a in atoms for c in a.text if c.isalnum())

    atoms = _eq_merge_decimals(atoms)
    atoms = _eq_merge_fractions(atoms, bars)
    atoms = _eq_merge_paren_pieces(atoms)
    _eq_mark_superscripts(atoms)
    rows = _build_eq_rows(atoms)

    if _rows_have_x_overlap(rows):
        return None

    assembled = [_assemble_eq_row(row) for row in rows]
    result = "\n".join(r for r in assembled if r)
    if not result:
        return None

    # -- confidence gate --
    if _EMPTY_PARENS_RE.search(result):
        return None  # a paren merge found no matching content to wrap
    if any(_BARE_FRAGMENT_RE.match(line.strip()) for line in result.split("\n")):
        return None  # an orphaned digit that never joined its row
    if any(not line.strip() or "=" not in line for line in result.split("\n")):
        return None  # a row that doesn't stand on its own as an equation
    if any(0xE000 <= ord(c) <= 0xF8FF for c in result):
        return None  # an unrecognized/ambiguous PUA run was left raw
    result_alnum = sum(1 for c in result if c.isalnum())
    if result_alnum < original_alnum:
        return None  # some digit/letter content was silently dropped

    return result


def _block_span_texts(block: dict) -> list[str]:
    return [s["text"] for line in block.get("lines", ()) for s in line["spans"] if s["text"]]


def _find_span_occurrences(text: str, span_texts: list[str], start: int, window: int) -> tuple[int, int] | None:
    """Locates every one of `span_texts` (COUNT-aware — a text value
    appearing twice in `span_texts` must be found twice) within
    text[start : start+window], and returns the (min_start, max_end)
    covering all of them. Returns None (never guesses a boundary) if any
    occurrence can't be found in that window."""
    counts = Counter(span_texts)
    bound = start + window
    min_start = None
    max_end = start
    for value, n in counts.items():
        if not value:
            continue
        pos = start
        for _ in range(n):
            idx = text.find(value, pos, bound)
            if idx == -1:
                return None
            if min_start is None or idx < min_start:
                min_start = idx
            end = idx + len(value)
            if end > max_end:
                max_end = end
            pos = idx + 1  # allow the next occurrence to start right after
    if min_start is None:
        return None
    return min_start, max_end


def _compute_block_text_ranges(text: str, dict_page: dict) -> dict[int, tuple[int, int]]:
    """Maps each block index to its [start, end) span in `text`, by
    walking blocks in PyMuPDF's own index order (which page.get_text()
    itself follows) and advancing a cursor strictly forward — safe even
    though a block's OWN internal line order may be scrambled, because
    blocks themselves are still emitted in order. A block whose span texts
    can't all be relocated (should not happen for well-formed pages) is
    simply omitted, and every block after it in the SAME failed search
    still gets its own independent attempt from the same cursor."""
    ranges: dict[int, tuple[int, int]] = {}
    cursor = 0
    for bi, block in enumerate(dict_page.get("blocks", ())):
        if "lines" not in block:
            continue
        span_texts = _block_span_texts(block)
        if not span_texts:
            continue
        window = max(600, 3 * sum(len(t) for t in span_texts))
        found = _find_span_occurrences(text, span_texts, cursor, window)
        if found is None:
            continue
        ranges[bi] = found
        cursor = found[1]
    return ranges


def reconstruct_equation_blocks(page, dict_page: dict, text: str) -> str:
    """Finds equation-editor blocks (contiguous runs of blocks carrying
    fraction-bar/stretchy-parenthesis evidence), reconstructs each one
    from pure geometry into a single coherent linear expression, and
    splices the confident ones back over the page's original scrambled
    text. Rejected (low-confidence) clusters are left completely
    untouched, so sections 1-4 of this module still get to patch whatever
    they can from them afterwards."""
    bars = _find_isolated_thin_bars(page)
    evidence = _evidence_block_indices(dict_page, bars)
    clusters = _cluster_consecutive(evidence)
    if not clusters:
        return text

    block_ranges = _compute_block_text_ranges(text, dict_page)

    replacements = []  # (start, end, new_text), applied last-to-first
    for cluster in clusters:
        if not all(bi in block_ranges for bi in cluster):
            continue
        result = _confident_equation_text(dict_page, bars, cluster)
        if result is None:
            continue
        start = block_ranges[cluster[0]][0]
        end = block_ranges[cluster[-1]][1]
        if start >= end:
            continue
        replacements.append((start, end, result))

    for start, end, new_text in sorted(replacements, key=lambda r: r[0], reverse=True):
        text = text[:start] + new_text + text[end:]
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct_math(page, text: str) -> str:
    """Cleans up mathematical-content corruption in `text` (the plain text
    already extracted from `page`) using the page's real geometry. Fails
    open: any error, or the absence of confirming geometric evidence,
    leaves `text` completely unchanged — this function never fabricates
    structure it can't verify."""
    if not text:
        return text
    original = text
    try:
        dict_page = page.get_text("dict")

        # Full equation-block reconstruction runs FIRST, on the page's
        # original text — it replaces whole confidently-reconstructed
        # regions wholesale (see section 5 below). Whatever it doesn't
        # confidently resolve is left untouched for the simpler,
        # string-patching passes below (sections 1-4) to still improve
        # incrementally — e.g. an isolated fraction inside a rejected
        # equation-block cluster is still fixed by _find_fractions below.
        text = reconstruct_equation_blocks(page, dict_page, text)

        fractions = _find_fractions(page, dict_page)
        text = _apply_fraction_substitutions(text, fractions)

        merges = _find_decimal_merges(dict_page)
        text = _apply_decimal_merges(text, merges)

        if _page_uses_paren_piece_font(dict_page):
            text = _collapse_bracket_pieces(text)

        pairs = _find_script_pairs(dict_page)
        text = _apply_script_substitutions(text, pairs)

        return text
    except Exception:
        return original
