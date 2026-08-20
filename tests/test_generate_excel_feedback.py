"""
tests/test_generate_excel_feedback.py — Fix 2 regression tests: Generate
Excel / ZIP must show visible "generating" feedback, an accurate
success/failure terminal state, a working download only on real success,
and never a stale download button left over from a previous attempt.

All tests exercise the REAL app.py UI via Streamlit's AppTest (button
clicks, not direct session_state pokes), consistent with the rest of
test_app_ui.py's approach — these live in their own file since Fix 2 is a
distinct concern from the review/edit and image-detection tests there.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import openpyxl
from streamlit.testing.v1 import AppTest

import excel_writer
from excel_writer import write_excel as real_write_excel
from extractor import extract_document, detect_marking_scheme_for_range
from paper_segmenter import detect_paper_spans
from parser import parse_paper
from review_state import PaperState
from validator import validate_document

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(ROOT / "app.py")
SSC_PDF = str(ROOT / "SSC Stenographer MTP-2.pdf")
AFCAT_PDF = str(ROOT / "templates" / "AFCAT_Mock_1-5_SP_2026_Final_File.pdf")


def _build_paper_states(pdf_path: str):
    doc = extract_document(pdf_path)
    spans = detect_paper_spans(doc)
    states = []
    for span in spans:
        result = parse_paper(doc, span)
        questions = result["questions"]
        validate_document(questions)
        marking = detect_marking_scheme_for_range(pdf_path, span.start_page, span.end_page)
        states.append(PaperState(span, result["mode"], questions, marking, result["paper_anomalies"]))
    return doc, states


def _seeded_app(pdf_path: str, name: str) -> AppTest:
    doc, states = _build_paper_states(pdf_path)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["analyzed"] = True
    at.session_state["doc_result"] = doc
    at.session_state["paper_states"] = states
    at.session_state["uploaded_name"] = name
    at.session_state["uploaded_key"] = (name, 1)
    at.run()
    assert not at.exception, at.exception
    return at


# ---------------------------------------------------------------------------
# 1. Generate Excel visibly enters a "generating" state before writing —
#    proven directly, not just inferred: a wrapper around the real
#    write_excel asserts session_state already reads "generating" at the
#    exact moment it's invoked, then delegates to the real implementation
#    so the rest of the flow (success/file-on-disk) is still genuine.
# ---------------------------------------------------------------------------

def test_generate_excel_shows_generating_state_before_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "templates").symlink_to(ROOT / "templates")

    captured = {}

    def _observing_write_excel(questions, output_path, template_path=None, marking_scheme=None):
        import streamlit as st
        captured["status_during_call"] = st.session_state["generation_1"]["status"]
        return real_write_excel(questions, output_path, template_path=template_path, marking_scheme=marking_scheme)

    monkeypatch.setattr(excel_writer, "write_excel", _observing_write_excel)

    at = _seeded_app(SSC_PDF, "SSC.pdf")
    assert "generation_1" not in at.session_state or at.session_state["generation_1"]["status"] == "idle"

    at.button(key="gen_btn_1").click().run()
    assert not at.exception
    assert captured["status_during_call"] == "generating"
    assert at.session_state["generation_1"]["status"] == "success"  # terminal by the time .run() returns


# ---------------------------------------------------------------------------
# 2/3/4. Successful generation -> success state, download button present,
#    file genuinely exists on disk and is readable.
# ---------------------------------------------------------------------------

def test_generate_excel_success_shows_success_state_and_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "templates").symlink_to(ROOT / "templates")
    at = _seeded_app(SSC_PDF, "SSC.pdf")

    at.button(key="gen_btn_1").click().run()
    assert not at.exception

    state = at.session_state["generation_1"]
    assert state["status"] == "success"
    assert state["error"] is None

    # file genuinely exists and is readable, not just claimed
    path = Path(state["path"])
    assert path.exists()
    wb = openpyxl.load_workbook(path)
    assert "questions" in wb.sheetnames

    # success message rendered
    assert any("Excel generated successfully" in m.value for m in at.markdown) or \
           any("Excel generated successfully" in getattr(el, "value", "") for el in at.success)

    # download button present and wired to real bytes
    dl_buttons = [b for b in at.get("download_button") if b.proto.label == "DOWNLOAD EXCEL"]
    assert dl_buttons, "download button not found"
    assert dl_buttons[0].proto.url  # AppTest exposes downloadable data via a mock media URL, not raw bytes


# ---------------------------------------------------------------------------
# 5/6. Failed generation -> visible error with the ACTUAL exception text.
# ---------------------------------------------------------------------------

def test_generate_excel_failure_shows_actual_error_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "templates").symlink_to(ROOT / "templates")

    def _broken_write_excel(*args, **kwargs):
        raise RuntimeError("simulated disk failure while writing Excel")

    monkeypatch.setattr(excel_writer, "write_excel", _broken_write_excel)

    at = _seeded_app(SSC_PDF, "SSC.pdf")
    at.button(key="gen_btn_1").click().run()
    assert not at.exception  # the app itself must not crash — the error is caught and shown

    state = at.session_state["generation_1"]
    assert state["status"] == "failed"
    assert "simulated disk failure while writing Excel" in state["error"]

    # the actual reason text reaches the rendered UI, not a generic message
    error_texts = [e.value for e in at.error]
    assert any("simulated disk failure while writing Excel" in t for t in error_texts)
    assert any("Excel generation failed" in t for t in error_texts)


# ---------------------------------------------------------------------------
# 7. Failed generation must not show a stale download button from an
#    earlier successful attempt.
# ---------------------------------------------------------------------------

def test_failed_generation_does_not_expose_stale_download_button(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "templates").symlink_to(ROOT / "templates")
    at = _seeded_app(SSC_PDF, "SSC.pdf")

    # succeed once
    at.button(key="gen_btn_1").click().run()
    assert at.session_state["generation_1"]["status"] == "success"
    assert [b for b in at.get("download_button") if b.proto.label == "DOWNLOAD EXCEL"]

    # now make the NEXT attempt fail
    def _broken_write_excel(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(excel_writer, "write_excel", _broken_write_excel)
    at.button(key="gen_btn_1").click().run()
    assert not at.exception

    assert at.session_state["generation_1"]["status"] == "failed"
    assert not [b for b in at.get("download_button") if b.proto.label == "DOWNLOAD EXCEL"]  # no stale button


def test_blocking_error_after_prior_success_clears_stale_download(tmp_path, monkeypatch):
    """A different staleness path: generation succeeds once, then a LATER
    edit makes the current data blocking-invalid (e.g. a duplicate
    question number) before the user clicks Generate again. The stale
    success/download from before must not still be shown once the data is
    known to be invalid."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "templates").symlink_to(ROOT / "templates")
    at = _seeded_app(SSC_PDF, "SSC.pdf")

    at.button(key="gen_btn_1").click().run()
    assert at.session_state["generation_1"]["status"] == "success"

    ps = at.session_state["paper_states"][0]
    # force a duplicate question number -> a real blocking error
    ps.apply_edit(2, {"number": 1})
    at.run()
    assert not at.exception

    assert not [b for b in at.get("download_button") if b.proto.label == "DOWNLOAD EXCEL"]
    assert at.session_state["generation_1"]["status"] == "idle"
    assert any("Cannot generate Excel" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# 8. ZIP generation has equivalent success/error behavior.
# ---------------------------------------------------------------------------

def test_zip_generation_success_state_and_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "templates").symlink_to(ROOT / "templates")
    at = _seeded_app(AFCAT_PDF, "AFCAT.pdf")

    zip_button = next(b for b in at.button if b.label == "GENERATE ALL (ZIP)")
    zip_button.click().run()
    assert not at.exception

    state = at.session_state["zip_generation"]
    assert state["status"] == "success"
    assert state["included"] == 5

    with zipfile.ZipFile(io.BytesIO(state["bytes"])) as zf:
        assert len(zf.namelist()) == 5

    dl_buttons = [b for b in at.get("download_button") if "DOWNLOAD ALL" in b.proto.label]
    assert dl_buttons


def test_zip_generation_failure_shows_actual_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "templates").symlink_to(ROOT / "templates")

    def _broken_write_excel(*args, **kwargs):
        raise RuntimeError("simulated zip-time write failure")

    monkeypatch.setattr(excel_writer, "write_excel", _broken_write_excel)

    at = _seeded_app(AFCAT_PDF, "AFCAT.pdf")
    zip_button = next(b for b in at.button if b.label == "GENERATE ALL (ZIP)")
    zip_button.click().run()
    assert not at.exception

    state = at.session_state["zip_generation"]
    assert state["status"] == "failed"
    assert "simulated zip-time write failure" in state["error"]
    assert not [b for b in at.get("download_button") if "DOWNLOAD ALL" in b.proto.label]
    error_texts = [e.value for e in at.error]
    assert any("simulated zip-time write failure" in t for t in error_texts)
