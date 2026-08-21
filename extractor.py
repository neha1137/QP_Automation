"""
extractor.py — Page-level PDF text extraction with hybrid native/OCR fallback.

Strategy (per page, not per document):
    1. Try native text extraction (PyMuPDF).
    2. Score the extracted text for quality (length, printable/alnum ratio,
       presence of question/option numbering patterns, word coherence).
    3. If quality is GOOD -> use native text.
       If quality is BAD  -> render the page to an image and run Tesseract OCR.

The rest of the pipeline (parser.py) only ever sees plain text + metadata
about how it was obtained. It never needs to know native vs OCR.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from math_reconstructor import reconstruct_math

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    OCR_AVAILABLE = False

TESSERACT_INSTALL_HELP = (
    "OCR fallback requires the Tesseract OCR engine — a system binary, "
    "separate from the 'pytesseract' Python package — to be installed and "
    "on PATH.\n\n"
    "Install it with:\n"
    "  macOS (Homebrew):   brew install tesseract\n"
    "  Ubuntu/Debian:      sudo apt-get install tesseract-ocr\n"
    "  Windows:            https://github.com/UB-Mannheim/tesseract/wiki\n\n"
    "Then restart the app. Native-text PDFs (the common case) work fine "
    "without Tesseract — this is only needed for scanned/image-only pages."
)


class TesseractNotAvailableError(RuntimeError):
    """Raised only when a page actually needs OCR (native extraction failed
    the quality check) but the Tesseract binary can't be found. Documents
    that are 100% native text never hit this, by design."""


def check_tesseract_available() -> tuple[bool, str]:
    """
    Cheap (no subprocess spawn) check for whether OCR can actually run.
    Two independent things are required: the pytesseract/Pillow *Python
    packages*, and the *Tesseract binary* itself (a separate system-level
    install) resolvable on PATH — pytesseract just shells out to it.
    """
    if not OCR_AVAILABLE:
        return False, (
            "pytesseract/Pillow Python packages are not installed.\n"
            "Run: pip install -r requirements.txt"
        )
    if shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
        return False, TESSERACT_INSTALL_HELP
    return True, ""

# Boilerplate header/footer lines found in the sample paper. Kept generic
# (regex, not literal strings only) so it generalizes to similar mock papers.
# The bare-page-number pattern is handled SEPARATELY (see PAGE_NUMBER_RE
# below) because, unlike the others, it's ambiguous: a lone "0"/"1"/"2" can
# legitimately be a matrix/table row-or-column label inside a question's
# explanation (confirmed in this document — Q49's coding-decoding matrix).
# A literal footer string like "MOCK TEST PAPER - 2" is specific enough to
# never collide with real content anywhere on the page, so those are safe
# to strip unconditionally.
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*MOCK TEST PAPER.*$", re.I),
    re.compile(r"^\s*Oswaal .*$", re.I),
    re.compile(r"^\s*STAFF SELECTION COMMISSION\s*$", re.I),
    re.compile(r"^\s*STENOGRAPHER GRADE.*EXAMINATION\s*$", re.I),
    re.compile(r"^\s*Time Allotted[- ].*$", re.I),
    re.compile(r"^\s*Max marks[- ].*$", re.I),
    re.compile(r"^\s*Answers with Explanations\s*$", re.I),
    # AFCAT-shaped page furniture: "8  |   \x08" (page no. + pipe + stray
    # control char), "|  73" (running-header page no.), and a bare
    # "Answers" running header on solution pages. Generic (control-char
    # class + digit/pipe shape, not literal per-page strings) so it
    # generalizes to similar papers rather than this one PDF specifically.
    re.compile(r"^\s*\d{1,3}\s*\|\s*[\x00-\x1f]*\s*$"),
    re.compile(r"^\s*\|\s*\d{1,3}\s*$"),
    re.compile(r"^\s*Answers\s*[\x00-\x1f]*\s*$", re.I),
]
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
# only treat a bare number as a page number if it's near the very top or
# very bottom of the page's extracted text — real page numbers live in the
# header/footer, never buried in the middle of the content.
PAGE_NUMBER_EDGE_LINES = 3

NATIVE_QUALITY_THRESHOLD = 0.55   # below this -> OCR fallback
OCR_RENDER_DPI = 280


@dataclass
class PageExtractionResult:
    page_number: int
    text: str
    extraction_method: str          # "native" | "ocr" | "empty"
    confidence: float                # 0..1
    native_char_count: int
    ocr_used: bool
    quality_score: float = 0.0
    ocr_word_confidence: float | None = None


def strip_boilerplate(text: str) -> str:
    """Remove known header/footer noise lines. Safe to run on any text."""
    lines = text.split("\n")
    non_empty_idx = [i for i, l in enumerate(lines) if l.strip()]
    edge_idx = set(non_empty_idx[:PAGE_NUMBER_EDGE_LINES] + non_empty_idx[-PAGE_NUMBER_EDGE_LINES:])

    out_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        if any(p.match(stripped) for p in BOILERPLATE_PATTERNS):
            continue
        if i in edge_idx and PAGE_NUMBER_RE.match(stripped):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def compute_quality_score(text: str) -> float:
    """
    Heuristic 0..1 score of whether `text` looks like a clean, usable
    extraction of a real text page (vs. empty / garbled / near-empty).

    Not a simple len(text) > 0 check — combines several signals so a page
    with some text but garbage content still scores low.
    """
    if not text or not text.strip():
        return 0.0

    length = len(text)
    printable_ratio = sum(1 for c in text if c.isprintable() or c in "\n\t") / length
    alnum_ratio = sum(1 for c in text if c.isalnum()) / length

    # word coherence: how many alphabetic tokens of length >= 2 are present
    tokens = re.findall(r"[A-Za-z]{2,}", text)
    coherence = min(len(tokens) / 60.0, 1.0)  # a normal page has 60+ real words

    has_question_marker = bool(re.search(r"(?m)^\s*\d{1,3}\.\s", text))
    has_option_marker = bool(re.search(r"(?m)^\s*[1-4]\.\s*\S", text))

    length_factor = min(length / 400.0, 1.0)  # very short pages are suspicious

    score = (
        0.20 * printable_ratio
        + 0.20 * alnum_ratio
        + 0.25 * coherence
        + 0.15 * length_factor
        + 0.10 * float(has_question_marker)
        + 0.10 * float(has_option_marker)
    )
    return round(score, 3)


def normalize_ocr_text(text: str) -> str:
    """
    Light, non-destructive cleanup of raw OCR output.
    Deliberately does NOT attempt spelling/word autocorrection — that could
    silently change question meaning.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of spaces/tabs (but keep newlines intact)
    text = re.sub(r"[ \t]+", " ", text)
    # collapse 3+ blank lines down to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip("\n")


def _ocr_page(page: "fitz.Page", page_number: int) -> PageExtractionResult:
    available, help_msg = check_tesseract_available()
    if not available:
        # Fail loudly and specifically rather than silently returning empty
        # text for this page — a silent empty page would show up downstream
        # only as "questions missing", with no clue why.
        raise TesseractNotAvailableError(
            f"Page {page_number} has no usable native text and needs OCR, "
            f"but OCR is unavailable.\n\n{help_msg}"
        )

    pix = page.get_pixmap(dpi=OCR_RENDER_DPI)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # psm 6 = "assume a single uniform block of text" — works much better than
    # the default automatic segmentation (psm 3) for dense question-paper
    # layouts, where short left-aligned option lines otherwise get reordered.
    ocr_config = "--psm 6"
    raw_text = pytesseract.image_to_string(img, config=ocr_config)

    # word-level confidence from Tesseract
    try:
        data = pytesseract.image_to_data(img, config=ocr_config, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if c not in ("-1", -1)]
        avg_conf = (sum(confs) / len(confs)) / 100.0 if confs else 0.0
    except Exception:
        avg_conf = 0.0

    cleaned = normalize_ocr_text(raw_text)
    return PageExtractionResult(
        page_number=page_number,
        text=cleaned,
        extraction_method="ocr",
        confidence=avg_conf,
        native_char_count=0,
        ocr_used=True,
        quality_score=compute_quality_score(cleaned),
        ocr_word_confidence=avg_conf,
    )


def extract_page_text(page: "fitz.Page", page_number: int) -> PageExtractionResult:
    """
    Core hybrid extraction for a single page.
    Native first; OCR only if native quality is below threshold.
    """
    native_text = page.get_text()
    # Mathematical-content cleanup/reconstruction — a dedicated stage that
    # runs BEFORE boilerplate stripping/quality scoring and BEFORE parser.py
    # ever sees the text. Uses the page's real geometry (drawn fraction
    # bars, span fonts/sizes/positions) to repair PDF-extraction-level
    # corruption (split fractions, decorative stretchy-parenthesis glyphs,
    # scrambled decimal points, true superscripts/subscripts) that
    # latex_processor.py has no way to recover from once flattened to
    # plain text. See math_reconstructor.py's module docstring for why this
    # exists and its confidence guarantees; it fails open (returns the
    # input text unchanged) whenever it lacks confirming geometric
    # evidence, so ordinary pages are unaffected.
    native_text = reconstruct_math(page, native_text)
    native_clean = strip_boilerplate(native_text)
    score = compute_quality_score(native_clean)

    if score >= NATIVE_QUALITY_THRESHOLD:
        return PageExtractionResult(
            page_number=page_number,
            text=native_clean,
            extraction_method="native",
            confidence=min(1.0, 0.6 + score * 0.4),
            native_char_count=len(native_text),
            ocr_used=False,
            quality_score=score,
        )

    # Native extraction is weak/empty -> fall back to OCR for this page only.
    ocr_result = _ocr_page(page, page_number)
    ocr_result.native_char_count = len(native_text)
    ocr_result.text = strip_boilerplate(ocr_result.text)
    return ocr_result


@dataclass
class DocumentExtractionResult:
    pages: list[PageExtractionResult] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Unified text with page boundary markers preserved for debugging."""
        parts = []
        for p in self.pages:
            parts.append(f"\n\x0c[[PAGE {p.page_number} | {p.extraction_method}]]\x0c\n")
            parts.append(p.text)
        return "".join(parts)

    @property
    def native_page_count(self) -> int:
        return sum(1 for p in self.pages if p.extraction_method == "native")

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.extraction_method == "ocr")

    def summary(self) -> dict:
        return {
            "total_pages": len(self.pages),
            "native_pages": self.native_page_count,
            "ocr_pages": self.ocr_page_count,
            "per_page": [
                {
                    "page": p.page_number,
                    "method": p.extraction_method,
                    "confidence": round(p.confidence, 2),
                    "quality_score": p.quality_score,
                }
                for p in self.pages
            ],
        }


def extract_document(pdf_path: str) -> DocumentExtractionResult:
    doc = fitz.open(pdf_path)
    result = DocumentExtractionResult()
    for i, page in enumerate(doc):
        page_result = extract_page_text(page, i + 1)
        result.pages.append(page_result)
    doc.close()
    return result


MAX_MARKS_RE = re.compile(r"Max\s*marks[\s:-]*\s*(\d+)", re.I)
# AFCAT-style phrasing ("Maximum Marks: 300") — additive alternate, tried
# alongside MAX_MARKS_RE rather than replacing it.
MAX_MARKS_RE_ALT = re.compile(r"Maximum\s*Marks\s*[:\-]?\s*(\d+)", re.I)

TOTAL_QUESTIONS_RE = re.compile(r"Total\s*Questions?\s*[:\-]?\s*(\d+)", re.I)

NEGATIVE_MARKS_RE = re.compile(
    r"Negative\s*Mark(?:ing|s)?\s*[:\-]?\s*(-?\d+(?:\.\d+)?)", re.I
)

# Small mock-test papers often state small mark values in words rather
# than digits (e.g. "One mark is deducted", "carries ... three marks").
# Only resolves words actually in this map — an unrecognized word (e.g. a
# typo, or a value outside 1-10) is never guessed at, it just fails to
# match, per the "never fabricate" principle.
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_WORD_NUM_ALT = "|".join(WORD_NUMBERS)

# "Each Question carries ... three marks" / "... carries 3 marks" — the
# real doc has a typo ("carries of three marks"), so the pattern is
# deliberately loose between "carries" and the number/word.
MARKS_PER_Q_WORDS_RE = re.compile(
    rf"carries[^.]*?\b(\d+|{_WORD_NUM_ALT})\b\s+marks?\b", re.I
)
# "One mark is deducted for every wrong answer" / "1 mark is deducted..."
NEGATIVE_MARKS_WORDS_RE = re.compile(
    rf"\b(\d+|{_WORD_NUM_ALT})\b\s+marks?\s+(?:is|are|will\s+be)\s+deducted", re.I
)


def _word_or_digit_to_number(token: str) -> float | None:
    if token.isdigit():
        return float(token)
    return WORD_NUMBERS.get(token.lower())


def _distinct_numbers(matches: list[str], resolver=float) -> set:
    """Resolve each matched token and collect the distinct numeric values
    seen — used to detect conflicting statements of the same fact rather
    than just taking the first match."""
    values = set()
    for tok in matches:
        v = resolver(tok)
        if v is not None:
            values.add(v)
    return values


def find_max_marks_in_text(text: str) -> tuple[float | None, str]:
    """Returns (value, status) where status is 'pdf' | 'not_specified' |
    'ambiguous'. Tries both known phrasings; if they disagree, that's a
    conflict to flag, not a value to pick."""
    matches = [m.group(1) for m in MAX_MARKS_RE.finditer(text)]
    matches += [m.group(1) for m in MAX_MARKS_RE_ALT.finditer(text)]
    values = _distinct_numbers(matches, resolver=lambda t: float(t))
    if not values:
        return None, "not_specified"
    if len(values) > 1:
        return None, "ambiguous"
    return next(iter(values)), "pdf"


def find_total_questions_in_text(text: str) -> tuple[int | None, str]:
    matches = [m.group(1) for m in TOTAL_QUESTIONS_RE.finditer(text)]
    values = _distinct_numbers(matches, resolver=lambda t: int(t))
    if not values:
        return None, "not_specified"
    if len(values) > 1:
        return None, "ambiguous"
    return int(next(iter(values))), "pdf"


def find_negative_marks_in_text(text: str) -> tuple[float | None, str]:
    """Digit-explicit ("Negative Marking: 0.25") and worded ("One mark is
    deducted...") phrasings both contribute candidate values; conflicting
    distinct values -> ambiguous, never a silent pick."""
    matches = [abs(float(m.group(1))) for m in NEGATIVE_MARKS_RE.finditer(text)]
    for m in NEGATIVE_MARKS_WORDS_RE.finditer(text):
        v = _word_or_digit_to_number(m.group(1))
        if v is not None:
            matches.append(abs(v))
    values = set(matches)
    if not values:
        return None, "not_specified"
    if len(values) > 1:
        return None, "ambiguous"
    return next(iter(values)), "pdf"


def find_marks_per_question_in_text(text: str) -> tuple[float | None, str]:
    """Explicit "carries N/word marks" statement only — the derived
    (max_marks / total_questions) fallback is computed by the caller,
    which can see both this doc-stated value and the derived one and
    flag a conflict between them rather than silently preferring one."""
    matches = []
    for m in MARKS_PER_Q_WORDS_RE.finditer(text):
        v = _word_or_digit_to_number(m.group(1))
        if v is not None:
            matches.append(v)
    values = set(matches)
    if not values:
        return None, "not_specified"
    if len(values) > 1:
        return None, "ambiguous"
    return next(iter(values)), "pdf"


def detect_max_marks(pdf_path: str) -> int | None:
    """
    Looks for an explicit "Max marks- N" (or similar) statement in the raw
    PDF text — this is stripped out as boilerplate before parsing, so it's
    checked separately here rather than in the main extraction pipeline.
    Returns None if the PDF doesn't state a maximum marks value (or states
    conflicting ones); never guesses or infers one. Signature/behavior
    unchanged from before — delegates to the pure-text helper above.
    """
    doc = fitz.open(pdf_path)
    try:
        full_text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    value, _status = find_max_marks_in_text(full_text)
    return int(value) if value is not None else None


def detect_negative_marks(pdf_path: str) -> float | None:
    """
    Looks for an explicit negative-marking statement (e.g. "Negative
    Marking: 0.25") in the raw PDF text. Returns None — never a guessed
    convention like -0.25 — if the PDF doesn't state one explicitly (or
    states conflicting ones). This sample paper has no such statement, so
    it correctly returns None. Signature/behavior unchanged from before —
    delegates to the pure-text helper above.
    """
    doc = fitz.open(pdf_path)
    try:
        full_text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    value, _status = find_negative_marks_in_text(full_text)
    return value


def get_raw_page_text_range(pdf_path: str, start_page: int, end_page: int) -> str:
    """Raw (NOT boilerplate-stripped), 1-indexed inclusive page range —
    used for per-paper marking-scheme detection, which must see the same
    raw text `detect_max_marks`/`detect_negative_marks` see (marking
    statements are deliberately stripped as boilerplate before parser.py
    runs, so re-reading the PDF is required, not a shortcut)."""
    doc = fitz.open(pdf_path)
    try:
        parts = [
            doc[i].get_text()
            for i in range(start_page - 1, min(end_page, len(doc)))
        ]
    finally:
        doc.close()
    return "\n".join(parts)


def detect_marking_scheme_for_range(pdf_path: str, start_page: int, end_page: int) -> dict:
    """
    Per-paper marking scheme, scanning ONLY that paper's raw page range —
    never the whole document (a later paper's differing marks would
    otherwise leak into an earlier one). Every field independently reports
    its own provenance: "pdf" (explicit statement), "derived" (computed
    from other PDF-stated numbers), "not_specified" (absent), or
    "ambiguous" (conflicting statements found — never silently resolved).
    """
    text = get_raw_page_text_range(pdf_path, start_page, end_page)

    total_questions, tq_status = find_total_questions_in_text(text)
    max_marks, mm_status = find_max_marks_in_text(text)
    if max_marks is not None and max_marks == int(max_marks):
        max_marks = int(max_marks)
    negative_marks, nm_status = find_negative_marks_in_text(text)
    if negative_marks is not None and negative_marks == int(negative_marks):
        negative_marks = int(negative_marks)
    stated_mpq, mpq_status = find_marks_per_question_in_text(text)

    derived_mpq = None
    if max_marks is not None and total_questions:
        derived_mpq = max_marks / total_questions
        if derived_mpq == int(derived_mpq):
            derived_mpq = int(derived_mpq)

    if stated_mpq is not None and derived_mpq is not None and stated_mpq != derived_mpq:
        # Doc explicitly says one thing, its own numbers imply another —
        # a real conflict, not a rounding nuance to paper over.
        marks_per_question, mpq_final_status = None, "ambiguous"
    elif stated_mpq is not None:
        marks_per_question, mpq_final_status = stated_mpq, "pdf"
    elif derived_mpq is not None:
        marks_per_question, mpq_final_status = derived_mpq, "derived"
    else:
        marks_per_question, mpq_final_status = None, mpq_status

    return {
        "total_questions": total_questions,
        "max_marks": max_marks,
        "marks_per_question": marks_per_question,
        "negative_marks": negative_marks,
        "source": {
            "total_questions": tq_status,
            "max_marks": mm_status,
            "marks_per_question": mpq_final_status,
            "negative_marks": nm_status,
        },
    }
