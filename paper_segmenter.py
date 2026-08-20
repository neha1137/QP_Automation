"""
paper_segmenter.py — Detects paper boundaries in a PDF that may contain
one or several independent question papers concatenated together (e.g.
"Sample Question Paper-1" through "-5" back to back), each with its own
1..N question numbering.

Generic by design: matches a structural pattern ("Sample Question
Paper-N" / "Question Paper N" / "Mock Test N" / "Test Paper N"), not a
literal exam name — a PDF with no such header at all is treated as a
single synthetic paper spanning the whole document, which is exactly the
path a document like the SSC Stenographer paper takes today.

Never guesses a boundary it isn't confident about (see module docstring
for `detect_paper_spans`): a page sequence that doesn't cleanly increase
with no gaps/repeats is flagged AMBIGUOUS with the reason and page range
recorded, rather than silently picked one way or another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from extractor import DocumentExtractionResult

# Matched against each page's already boilerplate-stripped text (so it can
# never re-match a stripped footer like SSC's "MOCK TEST PAPER - 2" —
# extractor.strip_boilerplate() has already removed that whole line by
# the time this ever sees the text). Anchored at line start; the caller
# only looks at the first few non-blank lines of a page and additionally
# checks the line is short and doesn't end in sentence punctuation, so a
# stray mid-prose mention of "test paper 2" can't masquerade as a header.
PAPER_HEADER_RE = re.compile(
    r"^\s*(?:Sample\s+)?(?:Question\s+Paper|Mock\s+Test|Test\s+Paper)\s*[-–—]?\s*(\d{1,3})\b",
    re.I,
)

MAX_HEADER_LINE_LEN = 40
HEADER_SEARCH_LINES = 5


@dataclass
class PaperSpan:
    paper_id: int
    paper_name: str | None
    start_page: int
    end_page: int
    status: str = "OK"              # "OK" | "AMBIGUOUS"
    ambiguity_reason: str | None = None


def _page_header_paper_number(page_text: str) -> int | None:
    """Returns the paper number if this page's TOP carries a genuine,
    title-like paper-header line; None otherwise. Deliberately
    conservative — only the first few non-blank lines are considered, and
    the matched line must be short and not sentence-like, to avoid a
    running header/footer repeat or a mid-explanation mention (both
    confirmed to occur in the real AFCAT PDF) being mistaken for a fresh
    boundary."""
    lines = [l for l in page_text.split("\n") if l.strip()][:HEADER_SEARCH_LINES]
    for line in lines:
        s = line.strip()
        if len(s) > MAX_HEADER_LINE_LEN:
            continue
        if s and s[-1] in ".?!,:;":
            continue
        m = PAPER_HEADER_RE.match(s)
        if m:
            return int(m.group(1))
    return None


def detect_paper_spans(doc_result: DocumentExtractionResult) -> list[PaperSpan]:
    """
    Scans every page for a paper-header line, then builds spans from the
    FIRST page each distinct paper number is seen. Never invents a
    boundary: if the sequence of first-seen paper numbers isn't a clean,
    gap-free, strictly increasing run (1, 2, 3, ... in order of
    appearance), every span is returned with status="AMBIGUOUS" and a
    reason, rather than silently accepting a partial/garbled reading.
    """
    first_seen_page: dict[int, int] = {}
    for p in doc_result.pages:
        num = _page_header_paper_number(p.text)
        if num is not None and num not in first_seen_page:
            first_seen_page[num] = p.page_number

    total_pages = len(doc_result.pages)

    if not first_seen_page:
        # No structural paper headers anywhere -> whole document is one
        # paper. This is the path every document without this header
        # style takes today (e.g. SSC) — behavior is unchanged.
        return [PaperSpan(paper_id=1, paper_name=None, start_page=1, end_page=total_pages)]

    ordered = sorted(first_seen_page.items(), key=lambda kv: kv[1])  # by start page
    numbers_in_order = [n for n, _ in ordered]
    expected = list(range(numbers_in_order[0], numbers_in_order[0] + len(numbers_in_order)))

    is_clean = numbers_in_order == expected

    spans: list[PaperSpan] = []
    for i, (num, start_page) in enumerate(ordered):
        end_page = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else total_pages
        if end_page < start_page:
            end_page = start_page
        spans.append(
            PaperSpan(
                paper_id=num,
                paper_name=f"Paper {num}",
                start_page=start_page,
                end_page=end_page,
            )
        )

    if not is_clean:
        reason = (
            f"paper header numbers appeared out of order or with gaps "
            f"(saw {numbers_in_order} in page order) — boundaries between "
            f"them cannot be confidently determined"
        )
        for span in spans:
            span.status = "AMBIGUOUS"
            span.ambiguity_reason = reason

    return spans
