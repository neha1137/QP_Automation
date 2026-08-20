"""
validator.py — Runs the required checks over parsed questions before Excel
generation, assigns a confidence level per question, and produces the
summary report shown in the UI.

Never silently repairs anything — every problem found is recorded as a
flag on the question (and rolled into confidence), not auto-corrected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

VALID_LETTERS = {"A", "B", "C", "D"}


@dataclass
class QuestionFlags:
    flags: list[str] = field(default_factory=list)
    confidence: str = "HIGH"  # HIGH | MEDIUM | LOW

    def add(self, msg: str, downgrade_to: str = "LOW"):
        self.flags.append(msg)
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        if order[downgrade_to] > order[self.confidence]:
            self.confidence = downgrade_to


def validate_question(q: dict) -> QuestionFlags:
    qf = QuestionFlags()

    # 1. question number exists
    if not q.get("number"):
        qf.add("missing question number", "LOW")

    # 2. question text exists
    if not q.get("stem", "").strip():
        qf.add("empty question text", "LOW")

    # 9. image questions not falsely treated as text
    if q["is_image"]:
        qf.add("image-based question — options/figure not extracted (V1 scope)", "LOW")
    else:
        # 4. MCQ has four options
        empties = [letter for letter, v in q["options"].items() if not v.strip()]
        if empties:
            qf.add(f"missing option(s): {', '.join(empties)}", "LOW")

    # 5 & 6. correct answer valid and refers to an existing option
    ans = q.get("correct_answer", "")
    if not ans:
        qf.add("no correct answer found in answer key", "MEDIUM")
    elif ans not in VALID_LETTERS:
        qf.add(f"correct answer '{ans}' is not A/B/C/D", "LOW")
    elif not q["is_image"] and not q["options"].get(ans, "").strip():
        qf.add(f"correct answer '{ans}' points to an empty option", "LOW")

    # 7. passage questions contain their passage
    if q["q_type"] in ("PASSAGE_MCQ", "PASSAGE_FILL_BLANK") and not q.get("passage", "").strip():
        qf.add("passage-type question but no passage text attached", "LOW")

    # explanation presence (soft signal, not a hard failure)
    if not q.get("explanation", "").strip():
        qf.add("no explanation found", "MEDIUM")

    # parser-level anomalies (e.g. option list restarted, unclassified text).
    # An answer-key conflict (compact key vs. solutions disagreeing) is a
    # stronger signal than an ordinary structural anomaly — the correct
    # answer itself is unknown, not just uncertain — so it downgrades
    # straight to LOW rather than MEDIUM.
    for note in q.get("anomaly_notes", []):
        if note.startswith("answer key conflict"):
            qf.add(f"parser anomaly: {note}", "LOW")
        else:
            qf.add(f"parser anomaly: {note}", "MEDIUM")

    # OCR-derived content is inherently less certain than native extraction,
    # but "OCR was used" alone isn't grounds for LOW — only poor OCR
    # confidence is (per spec: don't assume OCR output is perfect, but
    # don't blanket-downgrade good OCR either).
    if q.get("source_method") == "ocr":
        ocr_conf = q.get("source_confidence", 0.0)
        if ocr_conf < 0.6:
            qf.add(f"OCR fallback with low confidence ({ocr_conf:.0%})", "LOW")
        else:
            qf.add(f"extracted via OCR fallback ({ocr_conf:.0%} confidence)", "MEDIUM")

    # everything checked out
    if not qf.flags:
        pass  # stays HIGH
    return qf


def validate_document(questions: list[dict]) -> dict:
    numbers = [q["number"] for q in questions]
    counts = Counter(numbers)
    duplicates = sorted(n for n, c in counts.items() if c > 1)

    if numbers:
        expected_range = set(range(min(numbers), max(numbers) + 1))
        missing_numbers = sorted(expected_range - set(numbers))
    else:
        missing_numbers = []

    per_question = {}
    for q in questions:
        qf = validate_question(q)
        per_question[q["number"]] = qf
        q["confidence"] = qf.confidence
        q["validation_flags"] = qf.flags

    text_questions = [q for q in questions if not q["is_image"]]
    image_questions = [q for q in questions if q["is_image"]]
    high = [q for q in questions if q["confidence"] == "HIGH"]
    medium = [q for q in questions if q["confidence"] == "MEDIUM"]
    low = [q for q in questions if q["confidence"] == "LOW"]
    missing_answers = [q["number"] for q in questions if not q.get("correct_answer")]
    missing_options = [
        q["number"] for q in questions
        if not q["is_image"] and any(not v.strip() for v in q["options"].values())
    ]
    passage_questions = [q["number"] for q in questions if q.get("passage")]
    passage_groups = len({q["passage"][:60] for q in questions if q.get("passage")})

    report = {
        "total_detected": len(questions),
        "text_questions": len(text_questions),
        "image_questions": len(image_questions),
        "high_confidence": len(high),
        "medium_confidence": len(medium),
        "low_confidence": len(low),
        "missing_answers": len(missing_answers),
        "missing_answers_numbers": missing_answers,
        "missing_options": len(missing_options),
        "missing_options_numbers": missing_options,
        "duplicate_question_numbers": duplicates,
        "missing_question_numbers": missing_numbers,
        "passage_questions": len(passage_questions),
        "passage_groups": passage_groups,
    }
    return report


def validate_multi_paper(papers: list[tuple[object, list[dict]]]) -> dict:
    """
    Multi-paper rollup — does NOT modify validate_document() or its
    single-paper contract. Each paper's question list is already
    independently numbered 1..N by the time it gets here, so calling
    validate_document() once per paper is correct BY CONSTRUCTION (its
    "one contiguous range" assumption is true for a single paper). A
    shared Counter across all papers' 1..100 numbers would wrongly flag
    every number as an N-way "duplicate" — that bug is avoided entirely
    by never building one, rather than by teaching validate_document to
    special-case it.

    `papers` is a list of (span, questions) pairs, where `span` has at
    least `.paper_id`, `.paper_name`, `.status`, `.ambiguity_reason`
    attributes (a parser.PaperSpan, or anything shaped like one).
    """
    per_paper = []
    totals: dict[str, int] = {}
    for span, questions in papers:
        report = validate_document(questions)
        per_paper.append({
            "paper_id": span.paper_id,
            "paper_name": span.paper_name,
            "status": getattr(span, "status", "OK"),
            "ambiguity_reason": getattr(span, "ambiguity_reason", None),
            "report": report,
        })
        for key, value in report.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value

    return {
        "papers_detected": len(papers),
        "per_paper": per_paper,
        **totals,
    }


def format_report_text(report: dict) -> str:
    lines = [
        f"Total detected: {report['total_detected']}",
        f"Text questions: {report['text_questions']}",
        f"Image questions: {report['image_questions']}",
        f"High confidence: {report['high_confidence']}",
        f"Medium confidence: {report['medium_confidence']}",
        f"Needs review (low confidence): {report['low_confidence']}",
        f"Missing answers: {report['missing_answers']}",
        f"Missing options: {report['missing_options']}",
        f"Duplicate question numbers: {len(report['duplicate_question_numbers'])}",
        f"Missing question numbers: {len(report['missing_question_numbers'])}",
        f"Passage groups: {report['passage_groups']} ({report['passage_questions']} questions)",
    ]
    return "\n".join(lines)
