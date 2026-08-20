"""
parser.py — Turns unified extracted text (native + OCR, page-tagged) into
structured questions, sections, passages, and answer/explanation records.

Why a state machine and not "regex for question numbers":
Options in this paper are numbered 1-4(-5), the SAME digit scheme as
question numbers. A line like "2.\tBoast" can be either option B of a
question, or (coincidentally) the literal digits of question 2's marker.
Numeric value alone cannot disambiguate for small numbers. Instead we
track *structural state* (are we mid-option-sequence or not) and only
treat a marker as "end of this question" when it breaks the option
sequence (1,2,3,4[,5], optionally restarting at 1 for the two-list
"arrange in dictionary order" question format) — see extractor findings
in the design report for the concrete examples this handles.

This digit-option state machine (everything up to `parse_document`) is
untouched by, and handles exactly the same papers it always has, even
after the additions below: a second, LETTER-option ("(a)"/"(b)"/"(c)"/
"(d)") parsing path further down this file handles papers like AFCAT's,
where options are lettered instead of digit-numbered. `parse_paper()`
picks whichever path applies to a given paper's own text — never by
filename or exam name — and `parse_document()` is a one-line wrapper
around `parse_paper()` for the (still fully supported) single-paper case.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from extractor import DocumentExtractionResult
from paper_segmenter import PaperSpan

OPTION_LETTERS = ["A", "B", "C", "D"]

MARKER_RE = re.compile(r"^[ \t]*(\d{1,3})\.[ \t]*(.*)$")

PASSAGE_TRIGGER_RE = re.compile(
    r"Read the given passage carefully.*?alternatives\.",
    re.I | re.S,
)
CLOZE_TRIGGER_RE = re.compile(
    r"In the following passage some of the words have been left out.*?alternatives\.",
    re.I | re.S,
)

ANSWER_KEY_RE = re.compile(
    r"(?m)^[ \t]*(\d{1,3})\.[ \t]*Option[ \t]*\((\d)\)[ \t]*is[ \t]*correct\.[ \t]*"
)

OPTION_NUM_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}


def _looks_like_section_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60:
        return False
    if s[-1] in "?.:,;":
        return False
    if re.search(r"\d", s):
        return False
    words = s.split()
    if not (2 <= len(words) <= 8):
        return False
    cap_words = sum(1 for w in words if w[:1].isupper())
    return cap_words >= max(1, len(words) - 2)


def _extract_trailing_annotation(text: str) -> tuple[str, str | None, str | None, str | None]:
    """
    A "last option" chunk's content sometimes has a section header and/or a
    passage (trigger phrase + body) appended after the real option value,
    because in the source PDF those lines sit between one question's last
    option and the next question's marker with no other delimiter.

    Returns (clean_option_text, section_header_or_None, passage_text_or_None,
    passage_type_or_None ["narrative"|"cloze"]).
    """
    m = PASSAGE_TRIGGER_RE.search(text)
    if m:
        return text[: m.start()].strip(), None, text[m.end():].strip(), "narrative"
    m = CLOZE_TRIGGER_RE.search(text)
    if m:
        return text[: m.start()].strip(), None, text[m.end():].strip(), "cloze"
    # A section header can only be a line AFTER the option's own text — the
    # option's real value always starts at line 0, so never test line 0
    # itself (a short, plain, capitalized-first-word option like "To take
    # rest" or "Nanda Dynasty" would otherwise false-positive as a header).
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if _looks_like_section_header(line):
            return "\n".join(lines[:i]).strip(), line.strip(), None, None
    return text.strip(), None, None, None


@dataclass
class RawQuestion:
    printed_number: int
    sequence_number: int
    section: str | None
    stem_text: str = ""
    options: dict[int, str] = field(default_factory=dict)  # opt_num -> accumulated text
    passage: str | None = None
    passage_type: str | None = None  # "narrative" | "cloze"
    char_start: int = 0
    char_end: int = 0
    anomaly_notes: list[str] = field(default_factory=list)


def _is_short_option_like(content: str) -> bool:
    """Distinguishes a genuine 5th pre-list option value (e.g. "Banana") from
    the stem of a brand-new question that happens to be numbered 5 — the one
    case in this document where option-sequence continuation (opt 4 -> 5)
    and a real next-question boundary are both plausible for the same digit.
    A real option value here is short and not sentence-like."""
    s = content.strip()
    return bool(s) and len(s) <= 25 and "?" not in s and "\n\n" not in s


def build_unified_text(doc_result: DocumentExtractionResult) -> tuple[str, list[tuple[int, int, str, float]]]:
    """Concatenate page texts; return (text, boundaries) where boundaries is
    a sorted list of (start_offset, page_number, method, confidence) for
    per-question confidence lookup."""
    parts = []
    boundaries: list[tuple[int, int, str, float]] = []
    offset = 0
    for p in doc_result.pages:
        boundaries.append((offset, p.page_number, p.extraction_method, p.confidence))
        chunk = p.text + "\n"
        parts.append(chunk)
        offset += len(chunk)
    return "".join(parts), boundaries


def page_info_at(offset: int, boundaries: list[tuple[int, int, str, float]]) -> tuple[int, str, float]:
    starts = [b[0] for b in boundaries]
    idx = bisect_right(starts, offset) - 1
    idx = max(0, min(idx, len(boundaries) - 1))
    return boundaries[idx][1], boundaries[idx][2], boundaries[idx][3]


def _finalize_option_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _finalize_stem_text(text: str) -> str:
    # keep paragraph structure but collapse excessive blank lines/whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = "\n".join(l.strip() for l in text.split("\n"))
    return text.strip()


def _chunk_text(full_text: str) -> tuple[str, list[tuple[int, str, int]]]:
    """
    Split full_text on line-start "N." markers into (leading_text, chunks)
    where each chunk is (number, content, char_start_of_marker). `content`
    is everything from just after the marker up to the next marker (or end
    of text) — i.e. the marker "owns" all text until something else claims
    a new marker line.
    """
    # Some option pairs are crammed two-per-line to save space, e.g.
    # "1.\t102 : 10408\t 2.\t99 : 9801" — the second marker sits after a
    # tab, not at line start. Match markers at line start OR right after
    # a tab so both are caught.
    marker_starts = list(re.finditer(r"(?:^|(?<=\t))[ \t]*(\d{1,3})\.[ \t]*", full_text, re.M))
    if not marker_starts:
        return full_text, []
    leading_text = full_text[: marker_starts[0].start()]
    chunks = []
    for i, m in enumerate(marker_starts):
        num = int(m.group(1))
        content_start = m.end()
        content_end = marker_starts[i + 1].start() if i + 1 < len(marker_starts) else len(full_text)
        chunks.append((num, full_text[content_start:content_end], m.start()))
    return leading_text, chunks


def parse_questions(full_text: str) -> list[RawQuestion]:
    """
    Structural state machine over the question-paper region of the text,
    operating on marker-delimited chunks (see _chunk_text) rather than raw
    lines. See module docstring for why numeric value alone can't decide
    question vs. option boundaries.
    """
    leading_text, chunks = _chunk_text(full_text)

    questions: list[RawQuestion] = []
    active_section: str | None = None
    active_passage: str | None = None
    active_passage_type: str | None = None

    # A section header sometimes precedes the very first question of the doc.
    for line in leading_text.split("\n"):
        if _looks_like_section_header(line):
            active_section = line.strip()

    if not chunks:
        return []

    cur: RawQuestion | None = None
    state = "AWAITING_OPTIONS"  # right after a question's marker+stem chunk
    opt_num = 0
    seq = 0

    def start_question(num: int, content: str, char_start: int):
        nonlocal cur, state, opt_num, seq
        seq += 1
        cur = RawQuestion(
            printed_number=num,
            sequence_number=seq,
            section=active_section,
            passage=active_passage,
            passage_type=active_passage_type,
            char_start=char_start,
            stem_text=content,
        )
        state = "AWAITING_OPTIONS"
        opt_num = 0

    def close_current(char_end: int):
        nonlocal cur, active_section, active_passage, active_passage_type
        if cur is None:
            return
        last_opt = max(cur.options) if cur.options else None
        if last_opt is not None:
            clean, header, passage_body, passage_type = _extract_trailing_annotation(
                cur.options[last_opt]
            )
            cur.options[last_opt] = clean
            if header:
                active_section = header
                active_passage = None
                active_passage_type = None
            if passage_body:
                active_passage = passage_body
                active_passage_type = passage_type
        cur.char_end = char_end
        questions.append(cur)
        cur = None

    # The first chunk is always question 1's own marker+stem.
    first_num, first_content, first_start = chunks[0]
    start_question(first_num, first_content, first_start)

    for idx, (num, content, char_start) in enumerate(chunks[1:], start=1):
        if state == "AWAITING_OPTIONS":
            if num == 1:
                state = "OPTIONS"
                opt_num = 1
                cur.options[1] = content
            else:
                # No options seen yet but marker isn't "1." — keep folding
                # into the stem (malformed/unexpected structure); flagged
                # via anomaly if it happens more than once.
                cur.stem_text += "\n" + f"{num}." + content
                cur.anomaly_notes.append(
                    f"expected option 1 but saw marker {num} before any options"
                )
            continue

        # state == "OPTIONS"
        if num == 1:
            if opt_num >= 1:
                cur.anomaly_notes.append(
                    f"option list restarted after reaching option {opt_num} "
                    f"(kept the later block)"
                )
            cur.options = {1: content}
            opt_num = 1
        elif num == opt_num + 1 and num <= 4:
            opt_num = num
            cur.options[opt_num] = content
        elif num == 5 and opt_num == 4 and _is_short_option_like(content):
            # Genuine 5th pre-list item ("arrange in dictionary order"
            # format) — NOT a new question, even though 5 == a plausible
            # next question number this early in the document.
            opt_num = 5
            cur.options[5] = content
        else:
            # Option sequence broken -> this marker starts a new question,
            # whatever its printed number.
            close_current(char_start)
            start_question(num, content, char_start)

    close_current(len(full_text))
    return questions


def finalize_questions(raw_questions: list[RawQuestion]) -> list[dict]:
    out = []
    for rq in raw_questions:
        options = {}
        for letter, num in zip(OPTION_LETTERS, [1, 2, 3, 4]):
            options[letter] = _finalize_option_text(rq.options.get(num, ""))
        stem = _finalize_stem_text(rq.stem_text)

        is_image = bool(stem) and all(v == "" for v in options.values())

        if rq.passage:
            q_type = "PASSAGE_FILL_BLANK" if rq.passage_type == "cloze" else "PASSAGE_MCQ"
        elif "______" in stem or "_____" in stem:
            q_type = "FILL_BLANK"
        else:
            q_type = "MCQ"
        if is_image:
            q_type = "IMAGE_MCQ"

        out.append({
            "number": rq.printed_number,
            "sequence_number": rq.sequence_number,
            "section": rq.section,
            "q_type": q_type,
            "stem": stem,
            "passage": rq.passage,
            "options": options,
            "is_image": is_image,
            "anomaly_notes": rq.anomaly_notes,
            "char_start": rq.char_start,
            "char_end": rq.char_end,
        })
    return out


@dataclass
class ParsedAnswer:
    number: int
    correct_option_num: int
    correct_letter: str
    explanation: str
    char_start: int
    char_end: int
    # True only for the letter-mode path, when the compact Answer Key page
    # and the Solutions explanations disagree on this question's correct
    # letter — never silently resolved (see parse_answers_letter_mode).
    conflict: bool = False


def parse_answers(full_text: str) -> dict[int, ParsedAnswer]:
    matches = list(ANSWER_KEY_RE.finditer(full_text))
    answers: dict[int, ParsedAnswer] = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        opt_num = int(m.group(2))
        letter = OPTION_NUM_TO_LETTER.get(opt_num, "")
        exp_start = m.end()
        exp_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        explanation = full_text[exp_start:exp_end].strip()
        explanation = re.sub(r"[ \t]+", " ", explanation)
        explanation = re.sub(r"\n{3,}", "\n\n", explanation)
        answers[num] = ParsedAnswer(
            number=num,
            correct_option_num=opt_num,
            correct_letter=letter,
            explanation=explanation,
            char_start=m.start(),
            char_end=exp_end,
        )
    return answers


def split_question_and_answer_regions(full_text: str) -> tuple[str, str]:
    m = ANSWER_KEY_RE.search(full_text)
    if not m:
        return full_text, ""
    return full_text[: m.start()], full_text[m.start():]


# ---------------------------------------------------------------------------
# Letter-option ("(a)"/"(b)"/"(c)"/"(d)") parsing path — additive. Used only
# for a paper whose own text is letter-option-shaped (per
# detect_option_style below); the digit-option state machine above is
# completely untouched and keeps handling papers shaped like it always has.
#
# Because options here are LETTERED and questions are DIGIT-numbered, the
# two marker schemes never collide with each other the way SSC's digit/
# digit scheme did. The one remaining ambiguity is that a question's own
# stem can contain an embedded numbered sub-list (e.g. "Rearrange these
# sentences: 1. ... 2. ...", or "consider the following statements: 1. ...
# 2. ..."), which must not be mistaken for a fresh question boundary.
# ---------------------------------------------------------------------------

LETTER_MARKER_RE = re.compile(r"(?:^|(?<=\t))[ \t]*\(([a-eA-E])\)[ \t]*", re.M)
QUESTION_NUM_LINESTART_RE = re.compile(r"^[ \t]*(\d{1,3})\.[ \t]*", re.M)

# Strict, structural, bare-line section heading — e.g. "SECTION-A | VERBAL
# ABILITY" on a line by itself. Deliberately does NOT match the doc's own
# bulleted instructions-preamble mentions of the same sections (e.g.
# "l SECTION-A | VERBAL ABILITY — 30 Questions"): those have a leading
# bullet glyph and a trailing "— NN Questions" suffix, both of which fail
# this pattern by construction. This is the fix for the old free-text
# `_looks_like_section_header` heuristic misfiring on this document shape
# (it produced garbage like "(a)\t Facilitated" / "There are") — this path
# never uses that heuristic at all.
SECTION_HEADER_RE = re.compile(
    r"^[ \t]*SECTION[-\s]*([A-Z])[ \t]*\|[ \t]*([A-Z][A-Z &]*)[ \t]*$", re.M
)

# "Directions (Q. Nos. 2–6):" / "Direction (Q. Nos. 25-27):" — an explicit,
# reliable question range for a shared passage, more robust than phrase-
# matching since the wording around it varies but this annotation doesn't.
DIRECTIONS_RANGE_RE = re.compile(
    r"Directions?[ \t]*\(Q\.?[ \t]*Nos?\.?[ \t]*(\d{1,3})[ \t]*[–\-][ \t]*(\d{1,3})\)[ \t]*:",
    re.I,
)
_CLOZE_KEYWORDS_RE = re.compile(r"left out|numbered blank", re.I)

# Same explanation shape as the legacy ANSWER_KEY_RE ("N. Option (x) is
# correct.") but the option is a LETTER in parens, not a digit — kept as a
# separate constant rather than editing ANSWER_KEY_RE, so the legacy
# digit-mode regex object used by parse_answers()/split_question_and_
# answer_regions() is untouched.
ANSWER_KEY_LETTER_RE = re.compile(
    r"(?m)^[ \t]*(\d{1,3})\.[ \t]*Option[ \t]*\(([a-dA-D])\)[ \t]*is[ \t]*correct\.[ \t]*"
)
ANSWER_KEY_HEADING_RE = re.compile(r"(?m)^[ \t]*Answer[ \t]*Key[ \t]*$", re.I)
# Compact "Answer Key" page entries: "N.<tab>(x)", one per source line but
# some lines carry a leading tab from the page's 2-column layout — the
# line-start anchor already tolerates that; the after-tab alternative is
# kept for robustness against a genuinely same-line-crammed variant.
COMPACT_ANSWER_RE = re.compile(
    r"(?:^|(?<=\t))[ \t]*(\d{1,3})\.[ \t]*\(([a-dA-D])\)[ \t]*", re.M
)


def detect_option_style(text: str) -> str:
    """'letter' if (a)-(d)-style option markers clearly dominate this
    text, else 'digit' (the legacy scheme). A cheap structural check on
    the paper's own text — never based on filename or exam name."""
    letter_hits = len(LETTER_MARKER_RE.findall(text))
    return "letter" if letter_hits >= 8 else "digit"


def _find_letter_option_blocks(text: str) -> list[dict]:
    """Maximal runs of sequential (a)->(b)->(c)->(d)[->(e)] markers.
    Accepts PARTIAL runs (e.g. just a,b) rather than requiring exactly
    4 — mirrors the legacy parser's tolerance for "sequence broken", and
    is what lets a genuine content typo (a real duplicate-letter case
    exists in the AFCAT doc: "(a)...(b)...(b)...(d)...", where (c) should
    have been) degrade to a flagged incomplete question instead of a
    crash or a silently wrong pairing.

    An option's content ends at whichever comes first: the next letter
    marker, the next question-number marker, or the next SECTION-X|Name
    header — NOT just the next letter marker alone. Without the extra
    bounds, a question's last option (e.g. Q1's "(d)") would swallow
    everything up to the *next question that has any option letters at
    all*, including an intervening section heading (e.g. "SECTION-B |
    NUMERAL ABILITY", confirmed to sit between one section's last
    question and the next section's Q1 with no marker of its own before
    it) and/or the next question's own stem and number, since nothing
    else delimits it in letter-marker space alone.
    """
    matches = list(LETTER_MARKER_RE.finditer(text))
    qnum_starts = [m.start() for m in QUESTION_NUM_LINESTART_RE.finditer(text)]
    section_starts = [m.start() for m in SECTION_HEADER_RE.finditer(text)]
    letters_seq = "ABCDE"
    blocks = []
    i, n = 0, len(matches)
    while i < n:
        if matches[i].group(1).upper() != "A":
            i += 1
            continue
        opts: dict[int, str] = {}
        expected_idx = 0
        j = i
        last_content_end = matches[i].start()
        while j < n and expected_idx < 5:
            letter = matches[j].group(1).upper()
            if letter != letters_seq[expected_idx]:
                break
            content_start = matches[j].end()
            next_letter_start = matches[j + 1].start() if j + 1 < n else len(text)
            next_qnum_start = next(
                (s for s in qnum_starts if content_start <= s < next_letter_start),
                next_letter_start,
            )
            next_section_start = next(
                (s for s in section_starts if content_start <= s < next_letter_start),
                next_letter_start,
            )
            content_end = min(next_letter_start, next_qnum_start, next_section_start)
            opts[expected_idx + 1] = text[content_start:content_end]
            last_content_end = content_end
            expected_idx += 1
            j += 1
        blocks.append({
            "start": matches[i].start(),
            "end": last_content_end,
            "options": opts,
            "complete": expected_idx >= 4,
        })
        i = j if j > i else i + 1
    return blocks


def _find_preamble_trim_point(text: str, first_block_start: int) -> int:
    """Real question content starts right after the LAST bare SECTION-X|
    Name heading before the first option block — this is always the
    genuine in-stream heading directly preceding Q1 (the earlier bulleted
    preamble mentions of the same sections never match SECTION_HEADER_RE's
    strict bare-line shape at all, so no further disambiguation is
    needed). If a paper has no such heading, don't trim anything."""
    last = None
    for m in SECTION_HEADER_RE.finditer(text, 0, first_block_start):
        last = m
    return last.start() if last else 0


def parse_questions_letter_mode(
    q_region: str, expected_total: int | None = None
) -> tuple[list[RawQuestion], list[str]]:
    """
    Letter-option analog of parse_questions(). Block-anchored: each
    option block's owning question number is the FIRST line-start
    digit-dot marker found in the gap before it — safe against a stem's
    own embedded numbered sub-lists, which (in every case observed in the
    real document) only ever appear AFTER that question's own opening
    marker within the same gap.

    Returns (questions, paper_level_anomalies). The anomalies list is for
    cases this heuristic can't confidently resolve (missing markers, or a
    question whose option markers were never extractable at all) — those
    questions are still surfaced (never silently dropped), just flagged.
    """
    paper_anomalies: list[str] = []
    blocks = _find_letter_option_blocks(q_region)

    trim_point = _find_preamble_trim_point(q_region, blocks[0]["start"]) if blocks else 0

    active_section: str | None = None
    for m in SECTION_HEADER_RE.finditer(q_region, 0, trim_point):
        active_section = f"SECTION-{m.group(1)} | {m.group(2).strip()}"

    raw_questions: list[RawQuestion] = []
    prev_end = trim_point
    seq = 0
    for block in blocks:
        if block["start"] < trim_point:
            continue  # part of the preamble, not a real question

        gap = q_region[prev_end:block["start"]]
        for m in SECTION_HEADER_RE.finditer(gap):
            active_section = f"SECTION-{m.group(1)} | {m.group(2).strip()}"

        num_match = QUESTION_NUM_LINESTART_RE.search(gap)
        if num_match is None:
            paper_anomalies.append(
                f"an option block near text offset {block['start']} has no "
                f"preceding question-number marker — skipped, needs review"
            )
            prev_end = block["end"]
            continue

        printed_number = int(num_match.group(1))
        seq += 1
        rq = RawQuestion(
            printed_number=printed_number,
            sequence_number=seq,
            section=active_section,
            stem_text=gap[num_match.end():],
            options=dict(block["options"]),
            char_start=block["start"],
            char_end=block["end"],
        )
        if not block["complete"]:
            rq.anomaly_notes.append(
                f"incomplete option list (found {len(block['options'])} of 4)"
            )
        raw_questions.append(rq)
        prev_end = block["end"]

    # Safety net: a question must never silently vanish just because its
    # option markers weren't extractable (e.g. an image-only diagram
    # question whose (a)-(d) labels are themselves rasterized). Cross-
    # check every digit-dot marker actually present in the region against
    # the recognized question numbers; anything expected-but-missing gets
    # a best-effort stub (real text where findable, honestly empty where
    # not) and an explicit anomaly — never a fabricated one.
    recognized_numbers = {rq.printed_number for rq in raw_questions}
    expected_max = max(recognized_numbers) if recognized_numbers else 0
    if expected_total:
        expected_max = max(expected_max, expected_total)

    if expected_max:
        all_markers = list(QUESTION_NUM_LINESTART_RE.finditer(q_region, trim_point))
        marker_by_number = {int(m.group(1)): m for m in all_markers}
        for missing_num in sorted(set(range(1, expected_max + 1)) - recognized_numbers):
            m = marker_by_number.get(missing_num)
            if m is None:
                stub_stem = ""
                note = (
                    f"question {missing_num} was expected but no marker for "
                    f"it was found anywhere in the text — needs review"
                )
            else:
                next_marker = next((mm for mm in all_markers if mm.start() > m.start()), None)
                end = next_marker.start() if next_marker else len(q_region)
                stub_stem = q_region[m.end():end]
                note = (
                    f"question {missing_num} has no distinct option block "
                    f"(likely image-only options) — content extracted "
                    f"best-effort, needs review"
                )
            paper_anomalies.append(note)
            seq += 1
            raw_questions.append(RawQuestion(
                printed_number=missing_num,
                sequence_number=seq,
                section=None,  # never guess a section for a stub
                stem_text=stub_stem,
                options={},
                char_start=m.start() if m else -1,
                char_end=m.end() if m else -1,
                anomaly_notes=[note],
            ))

    raw_questions.sort(key=lambda rq: rq.printed_number)
    for i, rq in enumerate(raw_questions, start=1):
        rq.sequence_number = i

    return raw_questions, paper_anomalies


def detect_passage_ranges_letter_mode(q_region: str) -> tuple[dict[int, tuple[str, str]], list[str]]:
    """
    Detects `Directions (Q. Nos. X-Y):` headers and attaches the passage
    body verbatim (including its own short lead-in sentence — no attempt
    is made to trim that further, to avoid a second ambiguous boundary
    heuristic) to every question number in X..Y. A range whose end
    boundary (question X's own marker) can't be cleanly located
    contributes no passage entries and an anomaly instead of a truncated
    or misaligned one.
    """
    result: dict[int, tuple[str, str]] = {}
    anomalies: list[str] = []
    directive_matches = list(DIRECTIONS_RANGE_RE.finditer(q_region))
    all_markers = list(QUESTION_NUM_LINESTART_RE.finditer(q_region))

    for idx, dm in enumerate(directive_matches):
        x, y = int(dm.group(1)), int(dm.group(2))
        search_end = (
            directive_matches[idx + 1].start()
            if idx + 1 < len(directive_matches)
            else len(q_region)
        )
        marker_for_x = next(
            (m for m in all_markers if int(m.group(1)) == x and dm.end() <= m.start() < search_end),
            None,
        )
        passage_text = q_region[dm.end():marker_for_x.start()].strip() if marker_for_x else ""
        if marker_for_x is None or not passage_text:
            anomalies.append(f"passage boundary ambiguous for Q.Nos {x}-{y}")
            continue
        lead_in = q_region[dm.end():dm.end() + 200]
        passage_type = "cloze" if _CLOZE_KEYWORDS_RE.search(lead_in) else "narrative"
        for qn in range(x, y + 1):
            result[qn] = (passage_text, passage_type)
    return result, anomalies


def parse_answers_letter_mode(a_region: str) -> tuple[dict[int, ParsedAnswer], list[str]]:
    """
    Merges the compact Answer Key page (letter only, no explanation) with
    the Solutions section (letter + explanation). When only one source
    has an entry for a question, it's used as-is. When both exist and
    agree, the explanation-bearing one is used. When both exist and
    DISAGREE, that's a genuine conflict, not a preference order: flagged,
    confidence-downgraded, and `correct_answer` is left blank rather than
    silently picking either value (see ParsedAnswer.conflict).
    """
    anomalies: list[str] = []
    compact: dict[int, str] = {
        int(m.group(1)): m.group(2).upper() for m in COMPACT_ANSWER_RE.finditer(a_region)
    }

    explanation_matches = list(ANSWER_KEY_LETTER_RE.finditer(a_region))
    explained: dict[int, ParsedAnswer] = {}
    for i, m in enumerate(explanation_matches):
        num = int(m.group(1))
        letter = m.group(2).upper()
        exp_start = m.end()
        exp_end = explanation_matches[i + 1].start() if i + 1 < len(explanation_matches) else len(a_region)
        explanation = a_region[exp_start:exp_end].strip()
        explanation = re.sub(r"[ \t]+", " ", explanation)
        explanation = re.sub(r"\n{3,}", "\n\n", explanation)
        explained[num] = ParsedAnswer(
            number=num,
            correct_option_num=OPTION_NUM_TO_LETTER_REVERSE.get(letter, 0),
            correct_letter=letter,
            explanation=explanation,
            char_start=m.start(),
            char_end=exp_end,
        )

    answers: dict[int, ParsedAnswer] = {}
    for num in sorted(set(compact) | set(explained)):
        c, e = compact.get(num), explained.get(num)
        if e is not None and c is not None and e.correct_letter != c:
            anomalies.append(f"answer key conflict for question {num}: compact={c}, solutions={e.correct_letter}")
            answers[num] = ParsedAnswer(
                number=num, correct_option_num=0, correct_letter="",
                explanation=e.explanation, char_start=e.char_start, char_end=e.char_end,
                conflict=True,
            )
        elif e is not None:
            answers[num] = e
        elif c is not None:
            answers[num] = ParsedAnswer(
                number=num,
                correct_option_num=OPTION_NUM_TO_LETTER_REVERSE.get(c, 0),
                correct_letter=c, explanation="", char_start=-1, char_end=-1,
            )
    return answers, anomalies


def split_question_and_answer_regions_letter_mode(full_text: str) -> tuple[str, str]:
    """Letter-mode analog of split_question_and_answer_regions() — ends
    the question region at whichever comes first: the compact "Answer
    Key" page's bare heading, or the first Solutions-style explanation.
    Using the heading (when present) keeps the compact page itself out of
    the question region, so it's never re-chunked as ~100 phantom
    questions."""
    heading = ANSWER_KEY_HEADING_RE.search(full_text)
    explanation = ANSWER_KEY_LETTER_RE.search(full_text)
    candidates = [m.start() for m in (heading, explanation) if m]
    if not candidates:
        return full_text, ""
    split_at = min(candidates)
    return full_text[:split_at], full_text[split_at:]


OPTION_NUM_TO_LETTER_REVERSE = {v: k for k, v in OPTION_NUM_TO_LETTER.items()}


def parse_paper(
    doc_result: DocumentExtractionResult, span: PaperSpan, mode_hint: str | None = None
) -> dict:
    """
    Paper-scoped orchestration — the single implementation used both by
    the whole-document entrypoint (parse_document, span = the entire doc
    as one implicit paper) and by the multi-paper pipeline (one call per
    detected PaperSpan). Chooses digit-mode (legacy, byte-for-byte
    unchanged code path) or letter-mode per this paper's OWN text, never
    by filename or exam name.
    """
    pages = [p for p in doc_result.pages if span.start_page <= p.page_number <= span.end_page]
    sliced = DocumentExtractionResult(pages=pages)
    full_text, boundaries = build_unified_text(sliced)

    mode = mode_hint or detect_option_style(full_text)
    paper_anomalies: list[str] = []

    if mode == "digit":
        q_region, a_region = split_question_and_answer_regions(full_text)
        raw_questions = parse_questions(q_region)
        questions = finalize_questions(raw_questions)
        a_offset = len(q_region)
        answers_local = parse_answers(a_region)
        answers = {
            num: ParsedAnswer(
                number=a.number,
                correct_option_num=a.correct_option_num,
                correct_letter=a.correct_letter,
                explanation=a.explanation,
                char_start=a.char_start + a_offset,
                char_end=a.char_end + a_offset,
            )
            for num, a in answers_local.items()
        }
    else:
        q_region, a_region = split_question_and_answer_regions_letter_mode(full_text)
        raw_questions, q_anomalies = parse_questions_letter_mode(q_region)
        passage_map, passage_anomalies = detect_passage_ranges_letter_mode(q_region)
        for rq in raw_questions:
            if rq.printed_number in passage_map:
                rq.passage, rq.passage_type = passage_map[rq.printed_number]
        questions = finalize_questions(raw_questions)
        a_offset = len(q_region)
        answers_local, a_anomalies = parse_answers_letter_mode(a_region)
        answers = {
            num: ParsedAnswer(
                number=a.number,
                correct_option_num=a.correct_option_num,
                correct_letter=a.correct_letter,
                explanation=a.explanation,
                char_start=a.char_start + a_offset if a.char_start >= 0 else a.char_start,
                char_end=a.char_end + a_offset if a.char_end >= 0 else a.char_end,
                conflict=a.conflict,
            )
            for num, a in answers_local.items()
        }
        paper_anomalies = q_anomalies + passage_anomalies + a_anomalies

    for q in questions:
        page_no, method, confidence = page_info_at(q["char_start"], boundaries)
        q["source_page"] = page_no
        q["source_method"] = method
        q["source_confidence"] = confidence

        ans = answers.get(q["number"])
        if ans:
            q["correct_answer"] = ans.correct_letter
            q["explanation"] = ans.explanation
            if ans.conflict:
                q["anomaly_notes"].append(
                    "answer key conflict: compact answer key and solutions "
                    "disagree on the correct option"
                )
        else:
            q["correct_answer"] = ""
            q["explanation"] = ""

    return {
        "paper_id": span.paper_id,
        "paper_name": span.paper_name,
        "status": span.status,
        "ambiguity_reason": span.ambiguity_reason,
        "mode": mode,
        "questions": questions,
        "answers": answers,
        "boundaries": boundaries,
        "paper_anomalies": paper_anomalies,
    }


def parse_document(doc_result: DocumentExtractionResult) -> dict:
    """Whole-document entrypoint, unchanged in behavior — a thin wrapper
    around parse_paper() with the entire document as one implicit paper.
    Existing callers that only ever process one paper keep working
    exactly as before."""
    span = PaperSpan(paper_id=1, paper_name=None, start_page=1, end_page=len(doc_result.pages))
    return parse_paper(doc_result, span)
