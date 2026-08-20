"""
tests/test_review_state_images.py — V3 image-state additions to
review_state.PaperState (get_images/set_image/clear_image/
unconfirmed_image_questions), verified against the REAL parsed SSC/AFCAT
data (same convention as test_review_state.py) rather than a toy fixture.
"""

from __future__ import annotations

from paper_segmenter import detect_paper_spans
from parser import parse_paper
from review_state import PaperState
from validator import validate_document

SSC_Q5_IMAGE_QUESTIONS = [5, 39, 44, 45, 46, 47]


def _build_paper_state(doc_result, span):
    result = parse_paper(doc_result, span)
    questions = result["questions"]
    validate_document(questions)
    return PaperState(span, result["mode"], questions, {}, result["paper_anomalies"])


def _ssc_paper_state(ssc_doc_result):
    span = detect_paper_spans(ssc_doc_result)[0]
    return _build_paper_state(ssc_doc_result, span)


def _uploaded_artifact(destination: str, url: str, source: str = "auto_detected") -> dict:
    return {
        "destination": destination, "source": source, "region": None,
        "confidence": "high", "image_bytes_hash": "sha-" + url, "content_type": "image/png",
        "s3_key": f"mock-tests/images/{url}", "image_url": url,
        "status": "uploaded", "error": None, "user_confirmed": True,
    }


def test_new_paper_state_has_no_confirmed_images_yet(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    for n in SSC_Q5_IMAGE_QUESTIONS:
        assert ps.get_images(n) == []


def test_unconfirmed_image_questions_lists_every_image_question_before_confirmation(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    unconfirmed = ps.unconfirmed_image_questions()
    for n in SSC_Q5_IMAGE_QUESTIONS:
        assert n in unconfirmed


def test_set_image_confirms_and_removes_from_unconfirmed_list(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    ps.set_image(5, "question", _uploaded_artifact("question", "https://example.com/q5.png"))
    assert 5 not in ps.unconfirmed_image_questions()
    images = ps.get_images(5)
    assert len(images) == 1
    assert images[0]["destination"] == "question"
    assert images[0]["image_url"] == "https://example.com/q5.png"


def test_manual_upload_replaces_auto_detected_for_same_destination(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    ps.set_image(5, "question", _uploaded_artifact("question", "https://example.com/auto.png", source="auto_detected"))
    ps.set_image(5, "question", _uploaded_artifact("question", "https://example.com/manual.png", source="manual_upload"))

    images = ps.get_images(5)
    question_images = [im for im in images if im["destination"] == "question"]
    assert len(question_images) == 1  # never two competing entries for the same destination
    assert question_images[0]["image_url"] == "https://example.com/manual.png"
    assert question_images[0]["source"] == "manual_upload"


def test_different_destinations_coexist_independently(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    ps.set_image(5, "question", _uploaded_artifact("question", "https://example.com/q.png"))
    ps.set_image(5, "option_a", _uploaded_artifact("option_a", "https://example.com/a.png"))
    images = {im["destination"]: im["image_url"] for im in ps.get_images(5)}
    assert images == {"question": "https://example.com/q.png", "option_a": "https://example.com/a.png"}


def test_reset_question_clears_images_back_to_unconfirmed(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    ps.set_image(5, "question", _uploaded_artifact("question", "https://example.com/q5.png"))
    assert ps.get_images(5) != []

    ps.reset_question(5)
    assert ps.get_images(5) == []
    assert 5 in ps.unconfirmed_image_questions()
    assert not ps.is_modified(5)


def test_reset_all_clears_every_question_image(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    for n in SSC_Q5_IMAGE_QUESTIONS:
        ps.set_image(n, "question", _uploaded_artifact("question", f"https://example.com/{n}.png"))
    assert ps.unconfirmed_image_questions() == []

    ps.reset_all()
    for n in SSC_Q5_IMAGE_QUESTIONS:
        assert ps.get_images(n) == []


def test_clear_image_removes_only_that_destination(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    ps.set_image(5, "question", _uploaded_artifact("question", "https://example.com/q.png"))
    ps.set_image(5, "option_a", _uploaded_artifact("option_a", "https://example.com/a.png"))

    ps.clear_image(5, "question")
    dests = {im["destination"] for im in ps.get_images(5)}
    assert dests == {"option_a"}


def test_original_snapshot_never_mutated_by_set_image(ssc_doc_result):
    ps = _ssc_paper_state(ssc_doc_result)
    ps.set_image(5, "question", _uploaded_artifact("question", "https://example.com/q.png"))
    assert "images" not in ps.original_by_number[5] or ps.original_by_number[5].get("images") in (None, [])


# -- multi-paper isolation (AFCAT: 5 independent PaperState objects) -------

def test_multi_paper_image_isolation(afcat_doc_result):
    spans = detect_paper_spans(afcat_doc_result)
    paper_states = [_build_paper_state(afcat_doc_result, span) for span in spans]
    assert len(paper_states) >= 2

    p1, p2 = paper_states[0], paper_states[1]
    p1.set_image(1, "question", _uploaded_artifact("question", "https://example.com/paper1-q1.png"))

    assert p1.get_images(1) != []
    assert p2.get_images(1) == []  # a completely separate PaperState object — never touched
