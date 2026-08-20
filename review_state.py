"""
review_state.py — pure, Streamlit-free state management for the human
review/edit workflow.

Kept separate from app.py on purpose: this is the actual business logic
(apply an edit, reset a question, decide what's blocking vs a warning,
merge edited data for export) and has NO Streamlit import, so it's
independently unit-testable (see tests/test_review_state.py) without a
running Streamlit session. app.py only ever renders this state and wires
widgets to its methods — it never reimplements this logic inline.

Design (per spec): PDF -> extractor -> parser -> validator once at
"Analyze" time. The result is frozen as `original_questions` and NEVER
touched again — no edit, reset, or re-validate call in this module ever
re-runs extraction, OCR, or parsing. All edits live in a separate
`edited` overlay; `effective_questions()` merges the two for validation
and export.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field


def _snapshot(q: dict) -> dict:
    return copy.deepcopy(q)


# ---------------------------------------------------------------------------
# Difficulty Level — bulk question-number-range assignment.
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = ("easy", "medium", "hard")

_RANGE_TOKEN_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_SINGLE_TOKEN_RE = re.compile(r"^\d+$")


@dataclass
class DifficultyParseResult:
    numbers: set = field(default_factory=set)
    errors: list = field(default_factory=list)


def parse_question_number_list(text: str, valid_numbers) -> DifficultyParseResult:
    """
    Parses a comma-separated list of individual numbers and/or ranges
    ("1-10, 15, 18, 22") into a set of question numbers, validating each
    token against `valid_numbers` (this paper's actual question numbers).

    Never guesses at a malformed token — "abc", "5-", "10-2" (reversed
    range) all produce a clear error and contribute no number, rather
    than being silently skipped or partially interpreted.
    """
    valid_numbers = set(valid_numbers)
    errors: list[str] = []
    numbers: list[int] = []
    seen_in_field: set[int] = set()

    for raw_token in (text or "").split(","):
        token = raw_token.strip()
        if not token:
            continue

        m_range = _RANGE_TOKEN_RE.match(token)
        if m_range:
            start, end = int(m_range.group(1)), int(m_range.group(2))
            if start > end:
                errors.append(f"malformed range '{token}' (start is greater than end)")
                continue
            token_numbers = list(range(start, end + 1))
        elif _SINGLE_TOKEN_RE.match(token):
            token_numbers = [int(token)]
        else:
            errors.append(f"invalid entry '{token}'")
            continue

        for n in token_numbers:
            if n in seen_in_field:
                errors.append(f"duplicate question number {n} within this field")
                continue
            seen_in_field.add(n)
            if n not in valid_numbers:
                errors.append(f"question number {n} does not exist in this paper")
                continue
            numbers.append(n)

    return DifficultyParseResult(numbers=set(numbers), errors=errors)


class PaperState:
    """
    All review/edit state for ONE paper. Multi-paper isolation is
    structural, not a rule to remember: each paper gets its own
    PaperState with its own `original_questions`/`edited` dicts, so
    editing one paper can never reach another's data.
    """

    def __init__(
        self,
        span,
        mode: str,
        questions: list[dict],
        marking_scheme_detected: dict,
        paper_anomalies: list[str],
    ):
        self.span = span
        self.mode = mode
        # `questions` is already validate_document()-annotated (each dict
        # carries "confidence"/"validation_flags" from the initial
        # analysis pass) — snapshot AFTER that, so the frozen original
        # reflects what the parser+validator actually found, and nothing
        # ever mutates it again (validate_document mutates the dicts it's
        # given in place, so re-validating later must run on fresh
        # copies from effective_questions(), never on this list).
        self.original_questions: list[dict] = [_snapshot(q) for q in questions]
        self.original_by_number: dict[int, dict] = {
            q["number"]: q for q in self.original_questions
        }
        self.question_order: list[int] = [q["number"] for q in self.original_questions]
        self.edited: dict[int, dict] = {}
        self.marking_scheme_detected = marking_scheme_detected
        self.paper_anomalies = paper_anomalies
        # Bulk difficulty-level assignment, THIS paper's own — never
        # shared with any other PaperState. Raw text per level, exactly
        # as the user typed it (e.g. "1-10, 15, 18, 22"); parsed on
        # demand by difficulty_summary() so edits to one field are
        # immediately reflected without extra bookkeeping.
        self.difficulty_text: dict[str, str] = {level: "" for level in DIFFICULTY_LEVELS}

    # -- question-level edit state -----------------------------------------

    def get_current(self, number: int) -> dict:
        """The question as it currently stands — the edited version if
        one exists, else the original. Never a reference the caller
        could mutate the frozen original through (callers that want to
        mutate should go through apply_edit)."""
        if number in self.edited:
            return self.edited[number]
        return self.original_by_number[number]

    def apply_edit(self, number: int, updates: dict) -> None:
        """Merge `updates` onto this question's current state and store
        the result as the edited version. Supports partial updates (e.g.
        only `passage`, for "apply passage change to all") as well as a
        full edit-form save. Never touches original_questions."""
        base = _snapshot(self.get_current(number))
        base.update(updates)
        self.edited[number] = base

    def reset_question(self, number: int) -> None:
        """Restore this question to its original extracted value and
        clear its MODIFIED status. Never reprocesses the PDF."""
        self.edited.pop(number, None)

    def reset_all(self) -> None:
        """Restore every question in this paper to its original
        extracted value."""
        self.edited.clear()

    def is_modified(self, number: int) -> bool:
        return number in self.edited

    # -- image state (V3) -------------------------------------------------
    # Images are stored as an extra "images" key on the question dict,
    # merged through the SAME apply_edit() every other edit uses — no new
    # storage mechanism, so Reset Question/Reset All, is_modified(), and
    # multi-paper isolation all cover images for free. Only CONFIRMED
    # images (auto-detected-and-accepted, or manually uploaded) ever live
    # here; unconfirmed detection candidates are a separate, ephemeral
    # cache the caller (app.py) keeps outside PaperState entirely — this
    # is what guarantees an unconfirmed auto-detection can never silently
    # reach Excel (excel_writer.py only ever reads this "images" key).

    def get_images(self, number: int) -> list[dict]:
        """Confirmed image artifacts for this question, one per
        destination ("question"/"option_a".."option_d")."""
        return list(self.get_current(number).get("images", []))

    def set_image(self, number: int, destination: str, artifact: dict) -> None:
        """Stores/replaces the confirmed image for ONE destination on this
        question. Never leaves two competing entries for the same
        destination — a manual upload for "question" replaces a prior
        auto-detected "question" image outright, exactly as required."""
        current = [im for im in self.get_images(number) if im.get("destination") != destination]
        current.append(dict(artifact, destination=destination))
        self.apply_edit(number, {"images": current})

    def clear_image(self, number: int, destination: str) -> None:
        current = [im for im in self.get_images(number) if im.get("destination") != destination]
        self.apply_edit(number, {"images": current})

    def unconfirmed_image_questions(self) -> list[int]:
        """Question numbers the parser flagged as image-based that still
        have NO successfully uploaded image for any destination — used to
        warn (not silently pass over) before Excel generation. A question
        whose only image attempt(s) FAILED (status == "failed") counts as
        unconfirmed too, not just one with zero attempts at all — an
        attempted-but-failed upload is exactly the kind of thing Excel
        generation must still surface, not silently treat as "resolved"
        just because *an* entry exists in `images`."""
        return [
            n for n in self.question_order
            if self.get_current(n).get("is_image")
            and not any(im.get("status") == "uploaded" for im in self.get_images(n))
        ]

    def was_flagged(self, number: int) -> bool:
        """Whether the ORIGINAL parser+validator output flagged this
        question — based only on the frozen original, so editing a
        question never erases the fact that it was originally flagged
        (the UI shows that as resolved, not gone)."""
        original = self.original_by_number.get(number)
        return bool(original) and original.get("confidence") != "HIGH"

    def status_badges(self, number: int) -> list[str]:
        """✓ Extracted | ⚠ Needs Review | ✎ Modified | ✓ Manually
        Confirmed — combined per spec: a resolved-but-originally-flagged
        question shows BOTH "Manually Confirmed" and "Modified", not one
        replacing the other."""
        modified = self.is_modified(number)
        flagged = self.was_flagged(number)
        if modified and flagged:
            return ["✓ Manually Confirmed", "✎ Modified"]
        if modified:
            return ["✎ Modified"]
        if flagged:
            return ["⚠ Needs Review"]
        return ["✓ Extracted"]

    # -- whole-paper views ---------------------------------------------------

    def effective_questions(self) -> list[dict]:
        """Fresh copies, in original order — the CURRENT EDITED DATA.
        Safe to pass to validate_document() or write_excel() repeatedly:
        each call gets new dict objects, so re-validating never mutates
        original_questions or the edited overlay in place."""
        return [_snapshot(self.get_current(n)) for n in self.question_order]

    def flagged_numbers(self) -> list[int]:
        return [n for n in self.question_order if self.was_flagged(n)]

    def review_counts(self) -> dict:
        flagged = self.flagged_numbers()
        resolved = [n for n in flagged if self.is_modified(n)]
        return {
            "needs_review": len(flagged),
            "resolved": len(resolved),
            "remaining": len(flagged) - len(resolved),
        }

    # -- passages -------------------------------------------------------

    def get_passage_groups(self) -> list[dict]:
        """Groups the CURRENT (post-edit) questions by identical passage
        text. Grouping is exact-text-based on purpose: editing one
        question's passage away from the shared text naturally splits it
        out of the group on the next render, with no separate group-id
        bookkeeping needed."""
        groups: dict[str, list[int]] = {}
        order: list[str] = []
        for n in self.question_order:
            passage = self.get_current(n).get("passage")
            if not passage:
                continue
            if passage not in groups:
                groups[passage] = []
                order.append(passage)
            groups[passage].append(n)
        return [
            {"passage_text": key, "question_numbers": sorted(groups[key])}
            for key in order
        ]

    def apply_passage_edit(self, question_numbers: list[int], new_text: str) -> None:
        """Updates ONLY the `passage` field for exactly the given
        question numbers — used for both "save for this question only"
        (a list of one) and "apply to all" (the whole group's numbers),
        never touching any question outside that list."""
        for n in question_numbers:
            self.apply_edit(n, {"passage": new_text})

    # -- difficulty level (bulk, per paper) ------------------------------

    def set_difficulty_text(self, level: str, text: str) -> None:
        assert level in DIFFICULTY_LEVELS, level
        self.difficulty_text[level] = text or ""

    def difficulty_summary(self) -> dict:
        """
        Parses all three fields against THIS paper's own question numbers
        and cross-checks them against each other. Returns:
          - results: {level: DifficultyParseResult} (per-field parse errors)
          - conflicts: {number: [levels]} for any number claimed by more
            than one level — never silently resolved to either
          - resolved: {number: level} for every unambiguously assigned number
          - unassigned: sorted list of numbers claimed by no level
          - counts: {level: count} of unambiguously assigned questions
        """
        valid_numbers = set(self.question_order)
        results = {
            level: parse_question_number_list(self.difficulty_text[level], valid_numbers)
            for level in DIFFICULTY_LEVELS
        }

        membership: dict[int, list[str]] = {}
        for level, result in results.items():
            for n in result.numbers:
                membership.setdefault(n, []).append(level)

        conflicts = {n: levels for n, levels in membership.items() if len(levels) > 1}
        resolved = {n: levels[0] for n, levels in membership.items() if len(levels) == 1}
        unassigned = sorted(valid_numbers - set(membership.keys()))
        counts = {level: sum(1 for lvl in resolved.values() if lvl == level) for level in DIFFICULTY_LEVELS}

        return {
            "results": results,
            "conflicts": conflicts,
            "resolved": resolved,
            "unassigned": unassigned,
            "counts": counts,
        }

    def difficulty_errors(self) -> list[str]:
        """Hard-blocking messages — malformed/invalid entries and
        cross-field conflicts. These are NEVER bypassable via "Generate
        Excel Anyway": unlike an unassigned question (which can honestly
        export with a blank Difficulty Level cell), a conflict or an
        invalid number means we don't even know what the user intended,
        so there is no safe non-fabricated value to fall back to."""
        summary = self.difficulty_summary()
        errors: list[str] = []
        for level, result in summary["results"].items():
            for e in result.errors:
                errors.append(f"Difficulty ({level.capitalize()}): {e}")
        for n, levels in sorted(summary["conflicts"].items()):
            levels_str = " and ".join(lvl.capitalize() for lvl in levels)
            errors.append(f"Question {n} is assigned to both {levels_str} — remove it from one.")
        return errors

    def clear_difficulty(self) -> None:
        self.difficulty_text = {level: "" for level in DIFFICULTY_LEVELS}

    def apply_difficulty(self, questions: list[dict]) -> list[dict]:
        """Sets `difficulty_level` (lowercase "easy"/"medium"/"hard", or
        "" if unassigned) on each of the given question dicts — intended
        to be called on fresh copies from effective_questions(), never on
        original_questions or the edited overlay directly."""
        resolved = self.difficulty_summary()["resolved"]
        for q in questions:
            q["difficulty_level"] = resolved.get(q["number"], "")
        return questions


def compute_blocking_errors(
    effective_questions: list[dict], marking_scheme: dict, difficulty_errors: list[str] | None = None
) -> list[str]:
    """
    Errors that MUST be fixed before Excel export is allowed — never just
    a warning. Kept small and explicit per spec: a missing or duplicate
    question number, an answer letter that isn't A/B/C/D, or a
    non-positive/negative marking-scheme value. Everything else (missing
    answers, missing options, low confidence, unresolved passage
    ambiguity, image questions) is a non-blocking warning, consistent
    with the existing validator.py precedent of warn-but-allow for
    per-question issues.
    """
    errors: list[str] = []

    numbers = [q.get("number") for q in effective_questions]
    if any(n is None or n == "" for n in numbers):
        errors.append("One or more questions are missing a question number.")

    counts: dict = {}
    for n in numbers:
        if n is not None:
            counts[n] = counts.get(n, 0) + 1
    dupes = sorted(n for n, c in counts.items() if c > 1)
    if dupes:
        errors.append(f"Duplicate question numbers: {dupes}")

    for q in effective_questions:
        ans = (q.get("correct_answer") or "").strip()
        if ans and ans not in ("A", "B", "C", "D"):
            errors.append(f"Q{q.get('number')}: correct answer '{ans}' is not A/B/C/D.")

    mm = marking_scheme.get("maximum_marks")
    mpq = marking_scheme.get("marks_per_question")
    neg = marking_scheme.get("negative_marks")
    if mm is not None and mm <= 0:
        errors.append("Maximum Marks must be a positive number.")
    if mpq is not None and mpq <= 0:
        errors.append("Marks per Question must be a positive number.")
    if neg is not None and neg < 0:
        errors.append("Negative Marks cannot be negative.")

    if difficulty_errors:
        errors.extend(difficulty_errors)

    return errors
