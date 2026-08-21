"""
latex_processor.py — Isolated LaTeX-detection/conversion layer.

Scans extracted question text (stem, options A-D, explanation) for
CLEARLY mathematical substrings — unicode superscripts/subscripts, square
roots, ×÷≤≥≠±, or a bare "letter = number" equation — and rewrites just
that substring as LaTeX, wrapped in $...$. Everything else in the text is
left byte-for-byte untouched.

Design priorities (per spec), in order:
    1. Never fabricate or guess math — when uncertain, keep the original
       text exactly as extracted.
    2. Produce valid, consistent LaTeX ($...$ inline only — no $$...$$).
    3. Only then, be as complete as possible about real formulae.

This module has NO dependency on parser.py/app.py/excel_writer.py and no
side effects — it only transforms strings. See parser.py's parse_paper()
for the one place it's wired into the pipeline (once, at parse time, so
it can never double-run on a Streamlit rerun or re-process a field the
user has since hand-edited).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Unicode superscript / subscript digit maps
# ---------------------------------------------------------------------------

_SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
}
_SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-",
}
_SUP_CHARS = "".join(_SUPERSCRIPT_MAP)
_SUB_CHARS = "".join(_SUBSCRIPT_MAP)

_SYMBOL_MAP = {
    "×": r"\times",
    "÷": r"\div",
    "≤": r"\le",
    "≥": r"\ge",
    "≠": r"\neq",
    "±": r"\pm",
}
_SYMBOL_CHARS = "".join(_SYMBOL_MAP)

# ---------------------------------------------------------------------------
# Existing-LaTeX guard — never re-wrap/double-convert a $...$ span already
# present in the text (idempotency requirement).
# ---------------------------------------------------------------------------

_EXISTING_LATEX_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.S)


def _existing_latex_ranges(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _EXISTING_LATEX_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Tokenizer — every character of the input is classified as exactly one of:
#   strong    — on its own, proves this position is genuinely mathematical
#               (superscript/subscript attachment, √, ×÷≤≥≠±, or a bare
#               "letter = number" / "number = letter" equation)
#   weak      — a single letter or a plain number; math-adjacent but never
#               a trigger on its own (this is what keeps "The value of x
#               is 5." from becoming math — no strong signal anywhere)
#   connector — a +, -, or = sign (with optional surrounding whitespace)
#               that may glue two math atoms together
#   other     — anything else (whitespace, punctuation, words) — always a
#               hard boundary; a math span never crosses one
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"(?P<strong>"
    r"[A-Za-z][ \t]*=[ \t]*-?\d+(?:\.\d+)?"          # x = 5
    r"|\d+(?:\.\d+)?[ \t]*=[ \t]*-?[A-Za-z]\b"        # 5 = x
    r"|√\([^()]{1,80}\)"                              # √(a^2 + b^2)
    r"|√[A-Za-z0-9]+(?:\.\d+)?"                       # √16, √x
    r"|[A-Za-z0-9)\]][" + _SUP_CHARS + r"]+"          # x², 25³, )²
    r"|[A-Za-z0-9)\]][" + _SUB_CHARS + r"]+"          # a₁
    r"|[ \t]*[" + _SYMBOL_CHARS + r"][ \t]*"          # × ÷ ≤ ≥ ≠ ± (with
    r")"                                               # any adjacent spacing,
    # so "x ≤ 5" glues to its neighbors the same way "x + 5" does via the
    # connector token — a bare space alone is never a connector (that
    # would let ordinary prose join across a strong token), but a space
    # directly touching one of these symbols is unambiguously part of the
    # same mathematical expression.
    r"|(?P<connector>[ \t]*[+\-=][ \t]*)"
    r"|(?P<weak>[A-Za-z]+|\d+(?:\.\d+)?)"
    r"|(?P<other>.)",
    re.S,
)

_JOINABLE = ("weak", "connector", "strong")

_MAX_SPAN_TOKENS = 60  # safety valve against pathological expansion

# A slash immediately glued to another atom, with no space — the shape of
# an unrecognized inline fraction like "35/2". Used to stop a math span
# from ending mid-fraction (see _retract_trailing_slash_fraction).
_SLASH_TAIL_RE = re.compile(r"/[A-Za-z0-9]")


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Splits `text` into (kind, start, end) tokens covering every
    character. A "weak" match that's a real multi-letter word (e.g.
    "number") is reclassified as "other" — only a genuine SINGLE-letter
    token (a plausible variable like x/y/n) is ever joinable into a math
    span. Without this, an ordinary English word sitting right next to an
    operator (e.g. "number × (...)") would get partially swallowed into
    the span merely because it touches the operator's own consumed
    whitespace — a real failure mode confirmed against the actual
    templates/sample_input.pdf mock-test text (see tests/test_latex_real_pdf.py)."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        word = m.group()
        if kind == "weak" and len(word) > 1 and word.isalpha():
            kind = "other"
        tokens.append((kind, m.start(), m.end()))
    return tokens


def detect_math_segments(text: str) -> list[tuple[int, int]]:
    """Returns a sorted, non-overlapping list of (start, end) character
    offsets identifying substrings of `text` that are confidently
    mathematical. Never returns a span for plain prose — see module
    docstring for the exact signals required."""
    if not text:
        return []

    existing = _existing_latex_ranges(text)

    def _in_existing(pos: int) -> bool:
        return any(a <= pos < b for a, b in existing)

    tokens = _tokenize(text)
    n = len(tokens)
    visited = [False] * n
    spans: list[tuple[int, int]] = []

    for i in range(n):
        if visited[i] or tokens[i][0] != "strong" or _in_existing(tokens[i][1]):
            continue

        lo = i
        while (
            lo - 1 >= 0
            and tokens[lo - 1][0] in _JOINABLE
            and not _in_existing(tokens[lo - 1][1])
            and (i - lo) < _MAX_SPAN_TOKENS
        ):
            lo -= 1
        hi = i
        while (
            hi + 1 < n
            and tokens[hi + 1][0] in _JOINABLE
            and not _in_existing(tokens[hi + 1][1])
            and (hi - i) < _MAX_SPAN_TOKENS
        ):
            hi += 1

        # never start/end a span on a dangling connector (e.g. trailing " + ")
        while hi > lo and tokens[hi][0] == "connector":
            hi -= 1
        while lo < hi and tokens[lo][0] == "connector":
            lo += 1

        # never end a span on a bare weak token that's immediately glued
        # (no space) to an unrecognized "/denominator" — e.g. "...≠ 35/2"
        # must not become "...≠ 35$/2" (a half-converted fraction is
        # worse than an unconverted one). Retract past the whole trailing
        # weak run instead; the slash-fraction itself is left as plain
        # text, same as any other embedded (non-whole-field) fraction.
        while hi > lo and tokens[hi][0] == "weak" and _SLASH_TAIL_RE.match(text, tokens[hi][2]):
            hi -= 1
        while hi > lo and tokens[hi][0] == "connector":
            hi -= 1

        for k in range(lo, hi + 1):
            visited[k] = True

        spans.append((tokens[lo][1], tokens[hi][2]))

    spans.sort()
    return spans


# ---------------------------------------------------------------------------
# Conversion of one matched raw segment into LaTeX (without $ delimiters)
# ---------------------------------------------------------------------------

def _convert_superscripts(s: str) -> str:
    pattern = re.compile(r"([A-Za-z0-9)\]])([" + _SUP_CHARS + r"]+)")

    def repl(m: re.Match) -> str:
        base = m.group(1)
        digits = "".join(_SUPERSCRIPT_MAP[c] for c in m.group(2))
        return f"{base}^{digits}" if len(digits) == 1 else f"{base}^{{{digits}}}"

    return pattern.sub(repl, s)


def _convert_subscripts(s: str) -> str:
    pattern = re.compile(r"([A-Za-z0-9)\]])([" + _SUB_CHARS + r"]+)")

    def repl(m: re.Match) -> str:
        base = m.group(1)
        digits = "".join(_SUBSCRIPT_MAP[c] for c in m.group(2))
        return f"{base}_{digits}" if len(digits) == 1 else f"{base}_{{{digits}}}"

    return pattern.sub(repl, s)


def _convert_symbols(s: str) -> str:
    for sym, latex in _SYMBOL_MAP.items():
        s = s.replace(sym, latex + " ")
    return s


def _convert_sqrt(s: str) -> str:
    def repl_paren(m: re.Match) -> str:
        inner = convert_math_to_latex(m.group(1))
        return r"\sqrt{" + inner + "}"

    s = re.sub(r"√\(([^()]{1,80})\)", repl_paren, s)

    def repl_bare(m: re.Match) -> str:
        return r"\sqrt{" + m.group(1) + "}"

    s = re.sub(r"√([A-Za-z0-9]+(?:\.\d+)?)", repl_bare, s)
    return s


def convert_math_to_latex(segment: str) -> str:
    """Converts one already-identified mathematical raw substring into
    valid LaTeX source (no surrounding $ — callers add that). Order
    matters: √ is expanded first (recursing into its own contents), THEN
    superscripts/subscripts/symbols are converted in what's left, so a
    unicode superscript that was already inside a √(...) group is never
    scanned/converted twice."""
    s = _convert_sqrt(segment)
    s = _convert_superscripts(s)
    s = _convert_subscripts(s)
    s = _convert_symbols(s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Whole-field fraction special case — ONLY when the entire trimmed field is
# exactly "<atom>/<atom>" (never a slash embedded in a sentence, a date, or
# an abbreviation like "AC/DC" — those never match this pattern because
# something other than a single letter/number sits on one side, or there's
# more text around it).
# ---------------------------------------------------------------------------

_WHOLE_FIELD_FRACTION_RE = re.compile(
    r"^([A-Za-z]|\d+(?:\.\d+)?)[ \t]*/[ \t]*([A-Za-z]|\d+(?:\.\d+)?)$"
)


def _try_whole_field_fraction(text: str) -> str | None:
    stripped = text.strip()
    if not stripped or "\n" in stripped:
        return None
    m = _WHOLE_FIELD_FRACTION_RE.match(stripped)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    latex = f"\\frac{{{a}}}{{{b}}}"
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    return f"{lead}${latex}${trail}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_text(text: str) -> str:
    """Detects and converts clearly-mathematical substrings of `text` into
    LaTeX (wrapped in $...$), leaving every other character exactly as
    extracted. Idempotent: re-running this on its own output is a no-op
    (already-$-wrapped spans are recognized and skipped, never re-wrapped
    or nested into $$...$$)."""
    if not text or not isinstance(text, str):
        return text

    stripped = text.strip()
    if len(stripped) > 1 and stripped.startswith("$") and stripped.endswith("$"):
        return text  # already fully LaTeX end-to-end — never touch

    frac = _try_whole_field_fraction(text)
    if frac is not None:
        return frac

    spans = detect_math_segments(text)
    if not spans:
        return text

    out = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue  # defensive: never emit overlapping spans
        raw = text[start:end]
        # A span's own boundary tokens (e.g. the symbol pattern's optional
        # surrounding whitespace) can include leading/trailing spaces that
        # are really just separating the math from unrelated neighboring
        # text (e.g. "number × (" — the space before "×" is not part of
        # the expression). Keep that whitespace OUTSIDE the $...$ wrapper
        # rather than silently dropping it.
        core = raw.strip(" \t")
        leading_ws = raw[: len(raw) - len(raw.lstrip(" \t"))]
        trailing_ws = raw[len(raw.rstrip(" \t")):]
        out.append(text[cursor:start])
        out.append(leading_ws)
        out.append(f"${convert_math_to_latex(core)}$")
        out.append(trailing_ws)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# Fields a parsed question dict carries that can legitimately contain
# mathematical text. Deliberately does NOT include "passage", "section",
# image data, or any numeric/flag field — those are never touched.
QUESTION_MATH_FIELDS = ("stem",)
OPTION_LETTERS = ("A", "B", "C", "D")


def process_question_fields(q: dict) -> dict:
    """Applies process_text() to one parsed question dict's math-bearing
    fields — stem, options A-D, explanation — IN PLACE, and returns it.
    Every other key (is_image, images, passage, section, correct_answer,
    anomaly_notes, source_page, ...) is left completely untouched."""
    for field in QUESTION_MATH_FIELDS:
        value = q.get(field)
        if value:
            q[field] = process_text(value)

    options = q.get("options")
    if isinstance(options, dict):
        for letter in OPTION_LETTERS:
            value = options.get(letter)
            if value:
                options[letter] = process_text(value)

    explanation = q.get("explanation")
    if explanation:
        q["explanation"] = process_text(explanation)

    return q
