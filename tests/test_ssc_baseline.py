"""
test_ssc_baseline.py — THE regression gate. The SSC Stenographer MTP-2
PDF's numbers must stay byte-for-byte identical to what the pipeline
produced before multi-paper support was added, because it exercises the
legacy digit-option state machine, which multi-paper support must never
touch.
"""

from parser import parse_document
from paper_segmenter import detect_paper_spans
from validator import validate_document

from conftest import SSC_PDF  # noqa: F401 (path constant, not used directly)


def test_extraction_counts(ssc_doc_result):
    assert len(ssc_doc_result.pages) == 21
    assert ssc_doc_result.native_page_count == 21
    assert ssc_doc_result.ocr_page_count == 0


def test_single_synthetic_paper(ssc_doc_result):
    spans = detect_paper_spans(ssc_doc_result)
    assert len(spans) == 1
    span = spans[0]
    assert span.start_page == 1
    assert span.end_page == 21
    assert span.status == "OK"
    assert span.paper_name is None


def test_question_and_answer_counts(ssc_doc_result):
    parsed = parse_document(ssc_doc_result)
    assert parsed["mode"] == "digit"
    questions = parsed["questions"]
    report = validate_document(questions)

    assert report["total_detected"] == 200
    assert report["text_questions"] == 194
    assert report["image_questions"] == 6
    assert report["passage_groups"] == 5
    assert report["missing_answers"] == 0
    assert report["duplicate_question_numbers"] == []
    assert report["missing_question_numbers"] == []

    numbers = sorted(q["number"] for q in questions)
    assert numbers == list(range(1, 201))
