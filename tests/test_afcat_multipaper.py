"""
test_afcat_multipaper.py — the AFCAT PDF must be recognized as 5
independent 100-question papers, each correctly parsed, validated, and
answered — this is the multi-paper generalization the whole change exists
for.

The 5 page ranges below are hardcoded here as GROUND TRUTH to assert
against (confirmed by hand from the actual PDF's own "Sample Question
Paper-N" headers) — this is expected and correct for a test. Production
code (paper_segmenter.py) must still derive them structurally, never by
looking up this table.
"""

from paper_segmenter import detect_paper_spans
from parser import parse_paper
from validator import validate_document, validate_multi_paper

EXPECTED_SPANS = [(1, 18), (19, 36), (37, 55), (56, 73), (74, 89)]


def test_paper_spans(afcat_doc_result):
    spans = detect_paper_spans(afcat_doc_result)
    assert [(s.start_page, s.end_page) for s in spans] == EXPECTED_SPANS
    assert all(s.status == "OK" for s in spans)
    assert [s.paper_id for s in spans] == [1, 2, 3, 4, 5]


def _parse_all_papers(afcat_doc_result):
    spans = detect_paper_spans(afcat_doc_result)
    return [(span, parse_paper(afcat_doc_result, span)) for span in spans]


def test_each_paper_has_100_questions_no_gaps_no_dupes(afcat_doc_result):
    for span, result in _parse_all_papers(afcat_doc_result):
        assert result["mode"] == "letter", f"paper {span.paper_id} should be letter-mode"
        questions = result["questions"]
        report = validate_document(questions)

        assert report["total_detected"] == 100, span.paper_id
        assert report["duplicate_question_numbers"] == [], span.paper_id
        assert report["missing_question_numbers"] == [], span.paper_id
        assert report["missing_answers"] == 0, span.paper_id

        numbers = sorted(q["number"] for q in questions)
        assert numbers == list(range(1, 101)), span.paper_id


def test_papers_dont_cross_wire_answers(afcat_doc_result):
    """Paper N's Q1 and Paper M's Q1 must not collide — answers are keyed
    by (paper, number), not number alone."""
    for span, result in _parse_all_papers(afcat_doc_result):
        q1 = next(q for q in result["questions"] if q["number"] == 1)
        assert q1["correct_answer"] in ("A", "B", "C", "D")


def test_sections_are_structural_and_data_driven(afcat_doc_result):
    """Every paper must show exactly the 4 real SECTION-X|Name headings
    with their actual (data-driven, not hardcoded) question counts —
    never a garbage free-text fragment like the pre-fix "(a) Facilitated"
    or "There are"."""
    for span, result in _parse_all_papers(afcat_doc_result):
        sections = {}
        for q in result["questions"]:
            sections[q["section"]] = sections.get(q["section"], 0) + 1
        assert set(sections) == {
            "SECTION-A | VERBAL ABILITY",
            "SECTION-B | NUMERAL ABILITY",
            "SECTION-C | REASONING & MILITARY APTITUDE",
            "SECTION-D | GENERAL KNOWLEDGE",
        }, (span.paper_id, sections)


def test_q23_typo_is_flagged_not_crashed(afcat_doc_result):
    """Paper 1's Q23 has a real content typo in the source PDF (a
    duplicated "(b)" where "(c)" should be) — must degrade to a flagged
    incomplete question, never a crash or a silently wrong pairing."""
    spans = detect_paper_spans(afcat_doc_result)
    result = parse_paper(afcat_doc_result, spans[0])
    q23 = next(q for q in result["questions"] if q["number"] == 23)
    assert q23["options"]["A"] and q23["options"]["B"]
    assert any("incomplete option list" in note for note in q23["anomaly_notes"])


def test_at_least_one_passage_attached_via_directions_header(afcat_doc_result):
    spans = detect_paper_spans(afcat_doc_result)
    result = parse_paper(afcat_doc_result, spans[0])  # Paper 1 has a clean Q2-6 cloze passage
    passage_questions = [q for q in result["questions"] if q.get("passage")]
    assert len(passage_questions) >= 5


def test_validate_multi_paper_no_false_cross_paper_duplicates(afcat_doc_result):
    papers = [(span, result["questions"]) for span, result in _parse_all_papers(afcat_doc_result)]
    rollup = validate_multi_paper(papers)
    assert rollup["papers_detected"] == 5
    assert rollup["total_detected"] == 500
    for entry in rollup["per_paper"]:
        assert entry["report"]["duplicate_question_numbers"] == []
