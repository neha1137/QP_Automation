"""
app.py — Streamlit UI for the Mock Test PDF -> Excel Question Bank
Generator.

This file is UI/orchestration only. Extraction, paper segmentation,
parsing, and validation happen once, at "Analyze" time, in extractor.py /
paper_segmenter.py / parser.py / validator.py — this file never re-runs
any of that. All human review/edit state lives in review_state.PaperState
(Streamlit-free, independently unit-tested) — this file only renders that
state and wires widgets to its methods.

Workflow: Upload -> Analyze -> Extraction Summary -> Question Analysis ->
Needs Review -> Review/Edit -> Validate -> Generate Excel -> Download.
Excel is always generated from PaperState.effective_questions() — the
CURRENT EDITED DATA — never from a fresh re-parse.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

from extractor import (
    extract_document,
    check_tesseract_available,
    TesseractNotAvailableError,
    detect_marking_scheme_for_range,
)
from paper_segmenter import detect_paper_spans
from parser import parse_paper
from validator import validate_document
from excel_writer import write_excel, build_marking_scheme, TEMPLATE_PATH, QUESTION_TYPE_OPTIONS
from review_state import PaperState, compute_blocking_errors
import api_uploader
from image_locator import locate_image_regions
from image_extractor import build_artifact, hash_and_type_for_upload

# V3: destination keys used throughout the image UI/state — must match
# review_state.PaperState.set_image()'s `destination` values and
# excel_writer._IMAGE_DESTINATIONS exactly.
IMAGE_DESTINATIONS = ["question", "option_a", "option_b", "option_c", "option_d"]
IMAGE_DEST_LABELS = {
    "question": "Question Image",
    "option_a": "Option A",
    "option_b": "Option B",
    "option_c": "Option C",
    "option_d": "Option D",
    "ambiguous": "Needs Review",
}

st.set_page_config(page_title="Mock Test → Question Bank Generator", layout="wide", page_icon="📄")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .qb-section-label {
        font-size: 0.95rem; font-weight: 600; color: #555;
        margin-top: 0.4rem; margin-bottom: 0.2rem;
    }
    .qb-badge {
        display: inline-block; padding: 0.1rem 0.5rem; border-radius: 0.4rem;
        font-size: 0.8rem; font-weight: 600; margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Mock Test → Question Bank Generator")
st.caption("Upload a mock test PDF, review/correct anything the parser flagged, then export the question bank.")

# ── TEMPORARY — DEBUG IMAGE DETECTION ───────────────────────────────────
# Diagnostic-only toggle, safe to delete once image detection is visually
# confirmed working end-to-end in the real running app. Runs the raw
# PDF -> image_locator -> image_extractor -> PNG bytes pipeline directly
# (bypassing the normal is_image/detected-candidate gating entirely) and
# renders every candidate's full detail for the CURRENTLY selected
# question, independent of whether the normal review UI would show
# anything. Never uploads anything on its own — upload status is a read-only
# credential-presence check, never a real network call.
st.sidebar.markdown("### 🔧 Debug")
st.session_state.setdefault("debug_image_detection", False)
st.session_state["debug_image_detection"] = st.sidebar.checkbox(
    "Debug: Image Detection", value=st.session_state["debug_image_detection"],
    help="Shows raw image_locator/image_extractor output for the currently selected question — bbox, confidence, rendered preview, hash, upload status.",
)


# ── session state ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "uploaded_key": None,
        "uploaded_name": None,
        "analyzed": False,
        "analyze_error": None,
        "doc_result": None,
        "paper_states": None,   # list[PaperState]
        "pdf_bytes": None,      # V3: the analyzed PDF's own bytes, kept for
                                 # image_locator to reopen on later reruns —
                                 # the old tmp_path at analyze time was a
                                 # local variable, never persisted, so image
                                 # cropping had nothing to read from after
                                 # the initial "Analyze PDF" click completed.
        "image_detection_cache": {},   # (paper_id, q_number) -> [candidate dicts], ephemeral (never uploaded until confirmed)
        "image_upload_cache": {},      # sha256 -> {"url":..., "key":...}, dedup across reruns/questions
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _reset_analysis():
    for k in ["analyzed", "analyze_error", "doc_result", "paper_states", "pdf_bytes"]:
        st.session_state[k] = None
    st.session_state["analyzed"] = False
    st.session_state["image_detection_cache"] = {}
    # image_upload_cache is intentionally NOT cleared here — it's keyed by
    # content hash, so it stays valid (and keeps saving redundant S3 calls)
    # even across a fresh PDF upload/re-analysis in the same session.


def _parse_optional_number(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def _format_source_caption(current, detected, source: str) -> str:
    if source == "ambiguous":
        return "⚠ Ambiguous — conflicting values found in the PDF, needs review"
    if current is None:
        return "⚠ Not specified in PDF"
    if current == detected:
        return {
            "pdf": "✓ Extracted from PDF",
            "derived": "✓ Derived from Maximum Marks / Question Count",
        }.get(source, "✓ Extracted from PDF")
    return "✎ Manually entered"


def _current_marking_scheme(ps: PaperState) -> dict:
    """Reads the confirmed marking-scheme values for this paper straight
    out of its (possibly not-currently-rendered) widget state, falling
    back to the detected values for any widget the user hasn't touched
    yet this session — used both for display and for Excel generation,
    so "Generate All (ZIP)" reflects every paper's own current values
    even when only one paper's card is on screen at a time."""
    pid = ps.span.paper_id
    detected = ps.marking_scheme_detected
    max_marks = _parse_optional_number(
        st.session_state.get(f"max_marks_{pid}", "" if detected.get("max_marks") is None else str(detected["max_marks"]))
    )
    mpq = _parse_optional_number(
        st.session_state.get(f"mpq_{pid}", "" if detected.get("marks_per_question") is None else str(detected["marks_per_question"]))
    )
    neg = _parse_optional_number(
        st.session_state.get(f"neg_{pid}", "" if detected.get("negative_marks") is None else str(detected["negative_marks"]))
    )
    ms = build_marking_scheme(total_questions=len(ps.question_order), max_marks=max_marks, negative_marks=neg)
    ms["marks_per_question"] = mpq
    if max_marks != detected.get("max_marks"):
        ms["source"]["maximum_marks"] = "manual" if max_marks is not None else "not_specified"
    if neg != detected.get("negative_marks"):
        ms["source"]["negative_marks"] = "manual" if neg is not None else "not_specified"
    if mpq != detected.get("marks_per_question"):
        ms["source"]["marks_per_question"] = "manual" if mpq is not None else "not_specified"
    return ms


def _badge_html(badges: list[str]) -> str:
    colors = {
        "✓ Extracted": "#e6f4ea;color:#1e7e34",
        "⚠ Needs Review": "#fff3cd;color:#8a6500",
        "✎ Modified": "#e2e3ff;color:#3730a3",
        "✓ Manually Confirmed": "#e6f4ea;color:#1e7e34",
    }
    return " ".join(
        f'<span class="qb-badge" style="background:{colors.get(b, "#eee")}">{b}</span>' for b in badges
    )


def _jump_to_question(pid: int, number: int):
    st.session_state[f"search_{pid}"] = ""
    st.session_state[f"qselect_{pid}"] = number
    st.session_state[f"edit_mode_{pid}"] = True


# ── Generate Excel / ZIP lifecycle (Fix 2) ──────────────────────────────
# Single source of truth per paper: st.session_state[f"generation_{pid}"]
# = {"status": "idle"|"success"|"failed", "path": str|None, "error": str|None}.
# Exactly the CLICK -> clear -> "Generating..." -> validate/checks ->
# write -> verify -> store -> success/failure sequence, every time —
# never a state left over from a previous click, never a "success" shown
# before the file actually exists and is readable.

def _generation_state_key(pid) -> str:
    return f"generation_{pid}"


def _run_excel_generation(pid, effective_now, current_marking, out_path: str) -> None:
    """Runs one paper's Excel generation with visible progress feedback
    and leaves session_state in an accurate terminal state (never
    "generating" left lingering, never a stale prior success surviving a
    new failed attempt)."""
    gkey = _generation_state_key(pid)
    st.session_state[gkey] = {"status": "generating", "path": None, "error": None}
    with st.spinner("⏳ Generating Excel. Please wait..."):
        try:
            Path("output").mkdir(exist_ok=True)
            write_excel(effective_now, out_path, template_path=TEMPLATE_PATH, marking_scheme=current_marking)
            # Verify the file actually exists and is readable before ever
            # calling this a success — never claim success on faith alone.
            if not Path(out_path).exists():
                raise RuntimeError(f"Excel file was not found at {out_path} after generation.")
            with open(out_path, "rb") as f:
                if not f.read(4):
                    raise RuntimeError(f"Excel file at {out_path} is empty/unreadable.")
            st.session_state[gkey] = {"status": "success", "path": out_path, "error": None}
        except Exception as e:
            st.session_state[gkey] = {"status": "failed", "path": None, "error": str(e)}


# ── V3: image detection / upload ────────────────────────────────────────
# Detection (image_locator + image_extractor) is local/fast and runs once
# per question, cached in session_state — never re-run just because the
# user edited an unrelated field or Streamlit reran the script. Nothing
# here ever touches S3 or PaperState on its own: candidates stay purely
# local/ephemeral until the user explicitly clicks "Use Detected Image" or
# uploads manually (see _confirm_and_upload) — an unconfirmed detection
# can never reach Excel (excel_writer.py only reads PaperState's
# "images" key, which only _confirm_and_upload ever writes to).

def _open_analyzed_pdf() -> "fitz.Document | None":
    pdf_bytes = st.session_state.get("pdf_bytes")
    if not pdf_bytes:
        return None
    return fitz.open(stream=pdf_bytes, filetype="pdf")


def _questions_on_page(ps: PaperState, page_number: int) -> list[int]:
    """Question numbers on this page, in the paper's own reading order —
    used to find "the next question's marker" for image_locator, without
    it ever having to re-derive page/order information the parser already
    determined."""
    return [
        n for n in ps.question_order
        if ps.get_current(n).get("source_page") == page_number
    ]


def _detect_images_for_question(ps: PaperState, pid: int, number: int, q: dict) -> list[dict]:
    """Runs image_locator + image_extractor for one question, returns a
    list of plain dicts (not dataclasses, so they're trivially cacheable
    in st.session_state) with local preview bytes — never uploads
    anything. Returns [] on ANY failure (bad PDF state, locator exception,
    no region found) — the caller then shows "automatic detection failed"
    and falls back to manual upload only, never a guessed crop."""
    try:
        doc = _open_analyzed_pdf()
        if doc is None:
            return []
        page_number = q.get("source_page")
        if not page_number or page_number < 1 or page_number > len(doc):
            return []
        page = doc[page_number - 1]
        on_page = _questions_on_page(ps, page_number)
        idx = on_page.index(number) if number in on_page else -1
        next_number = on_page[idx + 1] if 0 <= idx < len(on_page) - 1 else None

        region_candidates = locate_image_regions(page, number, next_number)
        out = []
        for rc in region_candidates:
            artifact = build_artifact(page, rc)
            if artifact is None:
                continue
            out.append({
                "destination": artifact.destination,
                "region": artifact.region,
                "confidence": artifact.confidence,
                "source_type": artifact.source_type,
                "reason": artifact.reason,
                "content_type": artifact.content_type,
                "image_bytes": artifact.image_bytes,
                "sha256": artifact.sha256,
            })
        return out
    except Exception:
        # Detection is best-effort assistance, never allowed to crash the
        # review UI — any failure here degrades to "manual upload only".
        return []


def _confirm_and_upload(ps: PaperState, number: int, destination: str, image_bytes: bytes,
                         content_type: str, source: str, region, confidence, reason: str = "") -> dict:
    """The ONE path both auto-confirm and manual-upload go through —
    upload (dedup-cached) then store as the CONFIRMED image for this
    destination via PaperState.set_image(), which always replaces any
    prior image for that same destination (never two competing URLs)."""
    result = api_uploader.upload_image(
        image_bytes, content_type, cache=st.session_state["image_upload_cache"],
    )
    artifact = {
        "destination": destination,
        "source": source,
        "region": list(region) if region else None,
        "confidence": confidence,
        "reason": reason,
        "image_bytes_hash": result["sha256"],
        "content_type": content_type,
        "s3_key": result.get("key"),
        "image_url": result.get("url"),
        "status": result["status"],
        "error": result.get("error"),
        "user_confirmed": result["status"] == "uploaded",
    }
    ps.set_image(number, destination, artifact)
    return result


def _render_manual_upload(ps: PaperState, pid: int, number: int, default_dest: str, widget_suffix: str):
    dest_key = f"manualdest_{pid}_{number}_{widget_suffix}"
    default_idx = IMAGE_DESTINATIONS.index(default_dest) if default_dest in IMAGE_DESTINATIONS else 0
    sel = st.selectbox(
        "Image belongs to", options=IMAGE_DESTINATIONS, index=default_idx,
        format_func=lambda d: IMAGE_DEST_LABELS[d], key=dest_key,
    )
    file = st.file_uploader(
        "Upload Image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"],
        key=f"manualfile_{pid}_{number}_{widget_suffix}",
    )
    if file is not None and st.button("Upload & Use Image", key=f"manualbtn_{pid}_{number}_{widget_suffix}"):
        raw = file.getvalue()
        _sha, content_type = hash_and_type_for_upload(raw, file.name)
        result = _confirm_and_upload(
            ps, number, sel, raw, content_type,
            source="manual_upload", region=None, confidence="manual",
            reason="manually uploaded by reviewer",
        )
        if result["status"] == "uploaded":
            st.success(f"✓ Manual image uploaded — {IMAGE_DEST_LABELS[sel]}")
        else:
            st.error(f"✗ Upload failed: {result['error']}")
        st.rerun()


def _render_image_section(ps: PaperState, pid: int, number: int, q: dict):
    confirmed = {im["destination"]: im for im in ps.get_images(number)}

    st.markdown("**Automatic Image Detection**")
    cache = st.session_state["image_detection_cache"]
    cache_key = f"{pid}:{number}"
    if cache_key not in cache:
        cache[cache_key] = _detect_images_for_question(ps, pid, number, q)
    candidates = cache[cache_key]

    if not candidates:
        st.warning("⚠ Automatic image detection failed — no visual region could be confidently located.")
    else:
        for idx, cand in enumerate(candidates):
            dest = cand["destination"]
            with st.container(border=True):
                st.image(cand["image_bytes"], width=350)
                if dest == "ambiguous":
                    st.warning("⚠ Image detection needs review — destination is unclear. " + cand.get("reason", ""))
                elif cand["confidence"] == "high":
                    st.success(f"✓ Image detected — ✓ Crop looks valid — suggested destination: {IMAGE_DEST_LABELS[dest]}")
                else:
                    st.info(f"Image detected (confidence: {cand['confidence']}) — suggested destination: {IMAGE_DEST_LABELS.get(dest, dest)}. " + cand.get("reason", ""))

                dest_options = IMAGE_DESTINATIONS
                default_idx = dest_options.index(dest) if dest in dest_options else 0
                sel_key = f"imgdest_{pid}_{number}_{idx}"
                chosen_dest = st.selectbox(
                    "Destination", options=dest_options, index=default_idx,
                    format_func=lambda d: IMAGE_DEST_LABELS[d], key=sel_key,
                )

                st.caption("Does this image look correct?")
                bcol1, bcol2 = st.columns(2)
                use_clicked = bcol1.button("✓ Use Detected Image", key=f"usedet_{pid}_{number}_{idx}")
                manual_toggle_key = f"manualmode_{pid}_{number}_{idx}"
                manual_clicked = bcol2.button("✎ Upload Image Manually", key=f"manual_{pid}_{number}_{idx}")

                if use_clicked:
                    result = _confirm_and_upload(
                        ps, number, chosen_dest, cand["image_bytes"], cand["content_type"],
                        source="auto_detected", region=cand.get("region"), confidence=cand["confidence"],
                        reason=cand.get("reason", ""),
                    )
                    if result["status"] == "uploaded":
                        st.success(f"✓ Uploaded — URL generated for {IMAGE_DEST_LABELS[chosen_dest]}")
                    else:
                        st.error(f"✗ Upload failed: {result['error']}")
                    st.rerun()

                if manual_clicked:
                    st.session_state[manual_toggle_key] = True
                if st.session_state.get(manual_toggle_key):
                    _render_manual_upload(ps, pid, number, default_dest=chosen_dest, widget_suffix=str(idx))

    with st.expander("Manual upload (always available, even if detection succeeded)"):
        _render_manual_upload(ps, pid, number, default_dest="question", widget_suffix="always")

    if confirmed:
        st.markdown("**Confirmed Images for This Question**")
        for dest in IMAGE_DESTINATIONS:
            im = confirmed.get(dest)
            if not im:
                continue
            label = IMAGE_DEST_LABELS.get(dest, dest)
            if im.get("status") == "uploaded":
                st.success(f"✓ {label} — public URL generated")
                # Read-only text_input rather than st.success text: a long
                # URL is selectable/copyable here, and doesn't wrap into an
                # unreadable block inside a colored callout.
                st.text_input(
                    f"{label} image URL", value=im["image_url"], disabled=True,
                    key=f"confirmedurl_{pid}_{number}_{dest}",
                )
            else:
                st.error(f"✗ {label}: upload failed — {im.get('error')}")


# ── TEMPORARY — DEBUG IMAGE DETECTION renderer ──────────────────────────
def _render_debug_image_detection(ps: PaperState, pid: int, number: int, q: dict):
    """Raw PDF -> image_locator -> image_extractor -> PNG bytes -> preview,
    stopping there — never calls api_uploader.upload_image(). Runs
    unconditionally for whichever question is currently selected,
    completely independent of the normal is_image/detected-candidate
    gating in the review panel above, so it can show exactly what
    detection does (or doesn't) find even when the normal UI shows
    nothing. Safe with zero upload configuration: "Upload status" below is only
    ever a read-only check of whether credentials are present, never a
    real upload attempt."""
    with st.expander(f"🔧 DEBUG: Image Detection — Q{number}", expanded=True):
        st.caption(
            "Temporary diagnostic view. Pipeline: PDF → image_locator → "
            "image_extractor → PNG bytes → preview. Never uploads on its own."
        )

        pdf_bytes = st.session_state.get("pdf_bytes")
        if pdf_bytes:
            st.write(f"**pdf_bytes available:** yes ({len(pdf_bytes)} bytes)")
        else:
            st.error("**pdf_bytes available:** NO — image detection cannot run without it. "
                      "Re-run 'Analyze PDF' (this session's PDF bytes may not have been captured).")
            return

        st.write(f"**Question number:** {number}")
        st.write(f"**source_page:** {q.get('source_page')}")
        st.write(f"**is_image (legacy parser signal — does NOT gate detection):** {q.get('is_image')}")

        on_page = _questions_on_page(ps, q.get("source_page"))
        idx_on_page = on_page.index(number) if number in on_page else -1
        next_number = on_page[idx_on_page + 1] if 0 <= idx_on_page < len(on_page) - 1 else None
        st.write(f"**Questions on this page (order):** {on_page}")
        st.write(f"**Next question boundary used:** {next_number}")

        # Deliberately NOT read from the session cache — always a fresh
        # recomputation, so this view can never show stale/cached data.
        candidates = _detect_images_for_question(ps, pid, number, q)
        st.write(f"**Candidates found:** {len(candidates)}")

        if not candidates:
            st.warning("⚠ No visual candidates detected for this question — image_locator found "
                       "no raster image or vector drawing intersecting any region of this question.")
            return

        confirmed_by_dest = {im["destination"]: im for im in ps.get_images(number)}

        for idx, cand in enumerate(candidates):
            dest = cand["destination"]
            st.markdown(f"---\n**Candidate {idx} — destination = `{dest}`**")

            region = cand.get("region")
            if region:
                st.code(
                    f"bbox: x0={region[0]:.1f}, y0={region[1]:.1f}, x1={region[2]:.1f}, y1={region[3]:.1f}",
                    language=None,
                )
            st.write(f"**confidence:** {cand['confidence']}")
            st.write(f"**source_type:** {cand['source_type']}")

            st.image(cand["image_bytes"], caption=f"Q{number} / {dest} — rendered preview", width=450)
            st.write(f"**image bytes:** {len(cand['image_bytes'])}")
            st.write(f"**SHA-256:** `{cand['sha256']}`")

            confirmed = confirmed_by_dest.get(dest)
            is_confirmed = bool(confirmed and confirmed.get("status") == "uploaded")
            st.write(f"**confirmed:** {'yes' if is_confirmed else 'no'}")

            if confirmed and confirmed.get("status") == "uploaded":
                st.write("**Upload status:** ✓ Uploaded")
                st.write(f"**image_url:** {confirmed['image_url']}")
            elif confirmed and confirmed.get("status") == "failed":
                st.write(f"**Upload status:** ✗ Failed — {confirmed.get('error')}")
            else:
                try:
                    api_uploader.get_config()
                    st.write("**Upload status:** configured (not uploaded from this debug view — "
                              "use \"Use Detected Image\" above to actually upload)")
                except api_uploader.MissingAPIConfigError:
                    st.write("**Upload status:** Not configured")

            ext = "png" if cand["content_type"] == "image/png" else "jpg"
            st.download_button(
                "Download Debug Image",
                data=cand["image_bytes"],
                file_name=f"debug_q{number}_{dest}.{ext}",
                mime=cand["content_type"],
                key=f"debugdl_{pid}_{number}_{idx}",
            )


def _render_image_url_summary(ps: PaperState, pid: int):
    """Every confirmed image in this paper and the public URL it produced.

    Only reads PaperState (the confirmed store excel_writer.py also reads),
    so what this panel shows is exactly what will land in the Excel image
    columns — never an unconfirmed detection candidate."""
    rows = []
    for n in ps.question_order:
        for im in ps.get_images(n):
            rows.append((n, im))

    uploaded = [(n, im) for n, im in rows if im.get("status") == "uploaded" and im.get("image_url")]
    failed = [(n, im) for n, im in rows if im.get("status") != "uploaded"]

    if not rows:
        with st.expander("🖼 Image URLs (0 generated)", expanded=False):
            st.info(
                "No image URLs yet. Detecting an image does NOT upload it — open a "
                "question with an image, then click **✓ Use Detected Image** (or upload "
                "one manually) to generate its public URL."
            )
        return

    label = f"🖼 Image URLs — {len(uploaded)} generated"
    if failed:
        label += f", {len(failed)} failed"

    with st.expander(label, expanded=bool(failed)):
        if failed:
            st.error(f"{len(failed)} image(s) failed to upload — these will be BLANK in the Excel.")
            for n, im in failed:
                dest_label = IMAGE_DEST_LABELS.get(im.get("destination"), im.get("destination"))
                st.write(f"**Q{n}** · {dest_label} — {im.get('error') or 'unknown error'}")
            st.markdown("---")

        if uploaded:
            st.caption(
                "These exact URLs are what gets written into the Excel image columns."
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "Q": n,
                        "Destination": IMAGE_DEST_LABELS.get(im.get("destination"), im.get("destination")),
                        "Image URL": im["image_url"],
                    }
                    for n, im in uploaded
                ]),
                width="stretch", hide_index=True,
            )


# ── Step 1: Upload ──────────────────────────────────────────────────────
st.header("1. Upload Question Paper PDF")
uploaded = st.file_uploader("Upload Question Paper PDF", type=["pdf"], label_visibility="collapsed")

if uploaded is not None:
    file_key = (uploaded.name, uploaded.size)
    if st.session_state["uploaded_key"] != file_key:
        st.session_state["uploaded_key"] = file_key
        _reset_analysis()

    st.success(f"✓ PDF uploaded — **{uploaded.name}**")

    analyze_clicked = st.button("Analyze PDF", type="primary")

    if analyze_clicked and not st.session_state["analyzed"]:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Extracting text (native, OCR fallback where needed)..."):
                doc_result = extract_document(tmp_path)

            with st.spinner("Detecting paper boundaries, sections, questions, answers..."):
                spans = detect_paper_spans(doc_result)
                paper_states = []
                for span in spans:
                    result = parse_paper(doc_result, span)
                    questions = result["questions"]
                    validate_document(questions)  # annotates confidence/flags in place
                    marking = detect_marking_scheme_for_range(tmp_path, span.start_page, span.end_page)
                    paper_states.append(
                        PaperState(span, result["mode"], questions, marking, result["paper_anomalies"])
                    )

                if not any(ps.question_order for ps in paper_states):
                    raise ValueError(
                        "No questions could be detected in this PDF. Please check that "
                        "it's a valid, text-based mock test question paper."
                    )

            st.session_state["doc_result"] = doc_result
            st.session_state["paper_states"] = paper_states
            st.session_state["uploaded_name"] = uploaded.name
            st.session_state["pdf_bytes"] = uploaded.getvalue()  # V3: kept for image_locator on later reruns
            st.session_state["analyzed"] = True
            st.session_state["analyze_error"] = None

        except TesseractNotAvailableError as e:
            st.session_state["analyze_error"] = "OCR was needed for this PDF but couldn't run.\n\n" + str(e)
        except Exception as e:
            st.session_state["analyze_error"] = (
                "This PDF could not be processed. It may be corrupted, empty, or in an "
                "unsupported format.\n\nDetails: " + str(e)
            )

    if st.session_state["analyze_error"]:
        st.error(st.session_state["analyze_error"])

if not st.session_state["analyzed"]:
    st.stop()

doc_result = st.session_state["doc_result"]
paper_states: list[PaperState] = st.session_state["paper_states"]
multi = len(paper_states) > 1
stem = Path(st.session_state["uploaded_name"]).stem

# ── Step 2: Extraction status ───────────────────────────────────────────
st.header("2. Extraction Summary")
summary = doc_result.summary()
cols = st.columns(5 if multi else 4)
cols[0].metric("Pages", summary["total_pages"])
cols[1].metric("Native text pages", summary["native_pages"])
cols[2].metric("OCR pages", summary["ocr_pages"])
if multi:
    cols[3].metric("Papers detected", len(paper_states))
    cols[4].metric("Questions detected", sum(len(ps.question_order) for ps in paper_states))
else:
    cols[3].metric("Questions detected", sum(len(ps.question_order) for ps in paper_states))

ocr_ok, ocr_help = check_tesseract_available()
if ocr_ok:
    st.markdown("**OCR Status:** 🟢 Tesseract available")
else:
    st.markdown("**OCR Status:** 🟡 Tesseract unavailable")
    st.caption("Native-text PDFs still work normally.")

# ── Step 3: Question Analysis (aggregate) ───────────────────────────────
st.header("3. Question Analysis")
agg_reports = [validate_document(ps.effective_questions()) for ps in paper_states]
total_text = sum(r["text_questions"] for r in agg_reports)
total_image = sum(r["image_questions"] for r in agg_reports)
total_passage_groups = sum(r["passage_groups"] for r in agg_reports)
total_needs_review = sum(ps.review_counts()["needs_review"] for ps in paper_states)
total_resolved = sum(ps.review_counts()["resolved"] for ps in paper_states)

if multi:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Papers detected", len(paper_states))
    m2.metric("Questions detected", sum(len(ps.question_order) for ps in paper_states))
    m3.metric("Text questions", total_text)
    m4.metric("Image questions", total_image)
    m5.metric("Passage groups", total_passage_groups)
    m6.metric("Needs Review", total_needs_review)
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Text questions", total_text)
    m2.metric("Image questions", total_image)
    m3.metric("Passage groups", total_passage_groups)
    m4.metric("Needs Review", total_needs_review)

remaining = total_needs_review - total_resolved
if total_needs_review == 0:
    st.success("✓ No questions were flagged for review.")
elif remaining == 0:
    st.success(f"✓ All {total_needs_review} flagged questions reviewed")
else:
    st.warning(f"Needs Review: {total_needs_review}   ·   Resolved: {total_resolved}   ·   Remaining: {remaining}")

# ── Step 4: Paper selector (multi-paper only) ───────────────────────────
if multi:
    st.header("4. Select a Paper to Review")
    paper_labels = [f"{ps.span.paper_name or f'Paper {ps.span.paper_id}'} (pages {ps.span.start_page}–{ps.span.end_page})" for ps in paper_states]
    active_idx = st.selectbox("Paper", options=list(range(len(paper_states))), format_func=lambda i: paper_labels[i], key="active_paper_idx")
    active_ps = paper_states[active_idx]
    step_no = 5
else:
    active_ps = paper_states[0]
    step_no = 4

pid = active_ps.span.paper_id

if active_ps.span.status == "AMBIGUOUS":
    st.error(f"⚠ Needs Review — pages {active_ps.span.start_page}–{active_ps.span.end_page}")
    st.write(active_ps.span.ambiguity_reason or "Paper boundary could not be confidently determined.")
    st.caption("Excel generation is disabled for this paper — generating one now risks silently mis-splitting content between two papers' files.")
    st.stop()

report = validate_document(active_ps.effective_questions())
review_counts = active_ps.review_counts()

with st.container(border=True):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Questions", report["total_detected"])
    m2.metric("Text Questions", report["text_questions"])
    m3.metric("Image Questions", report["image_questions"])
    m4.metric("Passage Groups", report["passage_groups"])
    m5.metric("Needs Review", review_counts["remaining"])

    st.markdown('<div class="qb-section-label">Sections detected</div>', unsafe_allow_html=True)
    seen_sections, section_counts = [], {}
    for n in active_ps.question_order:
        sec = active_ps.get_current(n)["section"] or "Unknown Section"
        if sec not in section_counts:
            seen_sections.append(sec)
            section_counts[sec] = 0
        section_counts[sec] += 1
    for sec in seen_sections:
        st.markdown(f"- **{sec}** — {section_counts[sec]}")

    if active_ps.paper_anomalies:
        with st.expander(f"⚠ {len(active_ps.paper_anomalies)} structural note(s) for this paper"):
            for note in active_ps.paper_anomalies:
                st.markdown(f"- {note}")

    # ── NEEDS REVIEW (first-class) ──────────────────────────────────────
    st.markdown(f'<div class="qb-section-label">⚠ Needs Review ({review_counts["needs_review"]})</div>', unsafe_allow_html=True)
    if review_counts["needs_review"] == 0:
        st.caption("No questions were flagged for review in this paper.")
    else:
        with st.expander(f"Needs Review: {review_counts['needs_review']}   ·   Resolved: {review_counts['resolved']}   ·   Remaining: {review_counts['remaining']}", expanded=review_counts["remaining"] > 0):
            for n in active_ps.flagged_numbers():
                original = active_ps.original_by_number[n]
                flags = "; ".join(original.get("validation_flags", [])) or "flagged"
                badges = _badge_html(active_ps.status_badges(n))
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**Q{n}** {badges}<br><span style='color:#666'>⚠ {flags}</span>", unsafe_allow_html=True)
                with c2:
                    st.button("Review / Edit", key=f"needsreview_jump_{pid}_{n}", on_click=_jump_to_question, args=(pid, n))

    # ── Image URL summary ────────────────────────────────────────────────
    # Answers "did the detected image actually get a public URL?" for the
    # whole paper at once. Without this the only place a URL appears is
    # inside one question's review panel, so confirming 40 images meant
    # clicking into 40 questions to verify any of them worked.
    _render_image_url_summary(active_ps, pid)

    # ── Search / navigate ────────────────────────────────────────────────
    st.markdown('<div class="qb-section-label">Review &amp; Edit Questions</div>', unsafe_allow_html=True)
    search = st.text_input("Search question number/text", key=f"search_{pid}", placeholder="e.g. 23, or a word from the question")

    def _matches(n: int) -> bool:
        if not search.strip():
            return True
        s = search.strip().lower()
        q = active_ps.get_current(n)
        return s in str(n) or s in (q.get("stem") or "").lower()

    options = [n for n in active_ps.question_order if _matches(n)]

    if not options:
        st.info("No questions match your search.")
    else:
        qkey = f"qselect_{pid}"
        if qkey not in st.session_state or st.session_state[qkey] not in options:
            st.session_state[qkey] = options[0]

        def _fmt(n: int) -> str:
            badges = active_ps.status_badges(n)
            prefix = "⚠ " if "⚠ Needs Review" in badges else ("✎ " if "✎ Modified" in badges else "")
            return f"{prefix}Q{n}"

        selected_n = st.selectbox("Question", options=options, key=qkey, format_func=_fmt)
        q = active_ps.get_current(selected_n)
        edit_key = f"edit_mode_{pid}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = False

        with st.container(border=True):
            badges = active_ps.status_badges(selected_n)
            st.markdown(f"### Question {selected_n} {_badge_html(badges)}", unsafe_allow_html=True)

            # Visual detection is decided SPATIALLY (does image_locator
            # actually find a drawing/raster object in some region of this
            # question?), never by q["is_image"] alone — that flag only
            # means "the parser couldn't extract option text", which is a
            # completely different, unrelated signal (a text question can
            # still have a diagram in its stem — see SSC Q49/Q50 — and an
            # is_image question's "options" might genuinely have nothing
            # to show if detection finds no real visual object either).
            # is_image is kept ONLY as a legacy "needs review" signal
            # (still drives the badge/needs-review flow elsewhere) and as
            # one of three reasons to still show this section even with no
            # detection hit yet (so the manual-upload path stays reachable).
            detect_cache = st.session_state["image_detection_cache"]
            detect_cache_key = f"{pid}:{selected_n}"
            if detect_cache_key not in detect_cache:
                detect_cache[detect_cache_key] = _detect_images_for_question(active_ps, pid, selected_n, q)
            detected_candidates = detect_cache[detect_cache_key]
            has_confirmed_images = bool(active_ps.get_images(selected_n))

            if q["is_image"]:
                st.markdown("🖼 **IMAGE QUESTION**")
            elif detected_candidates or has_confirmed_images:
                st.markdown("🖼 **Visual Content Detected**")

            if q["is_image"] or detected_candidates or has_confirmed_images:
                _render_image_section(active_ps, pid, selected_n, q)

            # TEMPORARY — DEBUG IMAGE DETECTION: runs regardless of the
            # gating above, so it shows exactly what detection finds (or
            # doesn't) even for a question the normal review UI shows
            # nothing for.
            if st.session_state.get("debug_image_detection"):
                _render_debug_image_detection(active_ps, pid, selected_n, q)

            original = active_ps.original_by_number[selected_n]
            if any(note.startswith("answer key conflict") for note in original.get("anomaly_notes", [])):
                if active_ps.is_modified(selected_n):
                    st.success("✓ Manually confirmed — answer key conflict resolved.")
                else:
                    st.error("⚠ Answer key conflict: the compact answer key and the solutions section disagree. Pick the correct answer below.")

            if not st.session_state[edit_key]:
                st.markdown("**Question:**")
                st.write(q["stem"] or "_(no text)_")
                for letter in "ABCD":
                    val = q["options"].get(letter, "")
                    st.write(f"{letter}. {val if val else '_(blank)_'}")
                st.markdown(f"**Correct Answer:** {q['correct_answer'] or '_(none found)_'}")
                st.markdown(f"**Section:** {q['section'] or 'Unknown Section'}")
                if q.get("explanation"):
                    with st.expander("Explanation"):
                        st.write(q["explanation"])
                if q.get("passage"):
                    with st.expander(f"Passage (shared with this question's group)"):
                        st.write(q["passage"])
                        pkey = f"edit_passage_{pid}"
                        if st.button("Edit Passage", key=f"editpassage_btn_{pid}_{selected_n}"):
                            st.session_state[pkey] = selected_n
                        if st.session_state.get(pkey) == selected_n:
                            group = next((g for g in active_ps.get_passage_groups() if selected_n in g["question_numbers"]), None)
                            new_passage = st.text_area("Passage text", value=q["passage"], key=f"passage_text_{pid}_{selected_n}", height=150)
                            pc1, pc2, pc3 = st.columns(3)
                            if pc1.button("Save for this question only", key=f"passage_one_{pid}_{selected_n}"):
                                active_ps.apply_passage_edit([selected_n], new_passage)
                                st.session_state[pkey] = None
                                st.rerun()
                            if group and len(group["question_numbers"]) > 1 and pc2.button("Apply to all questions in this group", key=f"passage_all_{pid}_{selected_n}"):
                                active_ps.apply_passage_edit(group["question_numbers"], new_passage)
                                st.session_state[pkey] = None
                                st.rerun()
                            if pc3.button("Cancel", key=f"passage_cancel_{pid}_{selected_n}"):
                                st.session_state[pkey] = None
                                st.rerun()

                if st.button("Edit Question", type="primary", key=f"editbtn_{pid}_{selected_n}"):
                    st.session_state[edit_key] = True
                    st.rerun()
            else:
                st.markdown("**Edit Question**")
                with st.form(key=f"editform_{pid}_{selected_n}"):
                    new_number = st.number_input("Question Number", value=int(q["number"]), step=1)
                    new_stem = st.text_area("Question Text", value=q["stem"], height=120)
                    new_a = st.text_input("Option A", value=q["options"].get("A", ""))
                    new_b = st.text_input("Option B", value=q["options"].get("B", ""))
                    new_c = st.text_input("Option C", value=q["options"].get("C", ""))
                    new_d = st.text_input("Option D", value=q["options"].get("D", ""))
                    answer_choices = ["", "A", "B", "C", "D"]
                    cur_ans = q["correct_answer"] if q["correct_answer"] in answer_choices else ""
                    new_answer = st.selectbox("Correct Answer", options=answer_choices, index=answer_choices.index(cur_ans), format_func=lambda v: v or "(none)")
                    new_explanation = st.text_area("Explanation", value=q.get("explanation", ""), height=100)
                    new_section = st.text_input("Section", value=q.get("section") or "")
                    current_qtype = q.get("excel_question_type") or "objective_single_choice"
                    if current_qtype not in QUESTION_TYPE_OPTIONS:
                        current_qtype = QUESTION_TYPE_OPTIONS[0]
                    new_qtype = st.selectbox("Question Type", options=QUESTION_TYPE_OPTIONS, index=QUESTION_TYPE_OPTIONS.index(current_qtype))

                    save_col, cancel_col = st.columns(2)
                    save_clicked = save_col.form_submit_button("Save Changes", type="primary")
                    cancel_clicked = cancel_col.form_submit_button("Cancel")

                if save_clicked:
                    active_ps.apply_edit(int(selected_n), {
                        "number": int(new_number),
                        "stem": new_stem,
                        "options": {"A": new_a, "B": new_b, "C": new_c, "D": new_d},
                        "correct_answer": new_answer,
                        "explanation": new_explanation,
                        "section": new_section,
                        "excel_question_type": new_qtype,
                    })
                    st.session_state[edit_key] = False
                    st.success(f"✓ Q{selected_n} saved.")
                    st.rerun()
                if cancel_clicked:
                    st.session_state[edit_key] = False
                    st.rerun()

            rc1, rc2 = st.columns(2)
            if active_ps.is_modified(selected_n):
                if rc1.button("Reset Question", key=f"resetq_{pid}_{selected_n}"):
                    active_ps.reset_question(selected_n)
                    st.session_state[edit_key] = False
                    st.rerun()

    # ── Compact scan table (read-only) ──────────────────────────────────
    with st.expander(f"All {len(active_ps.question_order)} questions — scan table"):
        rows = [{
            "Q": n,
            "Section": active_ps.get_current(n)["section"] or "Unknown Section",
            "Status": " / ".join(active_ps.status_badges(n)),
            "Answer": active_ps.get_current(n)["correct_answer"],
        } for n in active_ps.question_order]
        st.dataframe(pd.DataFrame(rows), width="stretch", height=420)
        st.caption("Use the Question selector above to edit a specific question.")

    # ── Reset All ────────────────────────────────────────────────────────
    if active_ps.edited:
        confirm_key = f"confirm_reset_all_{pid}"
        if st.button("Reset All Edits", key=f"resetall_btn_{pid}"):
            st.session_state[confirm_key] = True
        if st.session_state.get(confirm_key):
            st.warning("Are you sure you want to discard all manual changes for this paper?")
            cc1, cc2 = st.columns(2)
            if cc1.button("Reset All", key=f"resetall_confirm_{pid}", type="primary"):
                active_ps.reset_all()
                st.session_state[confirm_key] = False
                st.rerun()
            if cc2.button("Cancel", key=f"resetall_cancel_{pid}"):
                st.session_state[confirm_key] = False
                st.rerun()

    # ── Difficulty Level (bulk assignment, per paper) ───────────────────
    st.markdown('<div class="qb-section-label">Difficulty Level</div>', unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        easy_text = st.text_input("Easy — Question numbers", value=active_ps.difficulty_text["easy"], key=f"diff_easy_{pid}", placeholder="e.g. 1-10, 15, 18, 22")
        active_ps.set_difficulty_text("easy", easy_text)
    with dc2:
        medium_text = st.text_input("Medium — Question numbers", value=active_ps.difficulty_text["medium"], key=f"diff_medium_{pid}", placeholder="e.g. 11-14, 16-17, 19-21")
        active_ps.set_difficulty_text("medium", medium_text)
    with dc3:
        hard_text = st.text_input("Hard — Question numbers", value=active_ps.difficulty_text["hard"], key=f"diff_hard_{pid}", placeholder="e.g. 23-30")
        active_ps.set_difficulty_text("hard", hard_text)

    diff_summary = active_ps.difficulty_summary()
    diff_errs = active_ps.difficulty_errors()

    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Easy", diff_summary["counts"]["easy"])
    dm2.metric("Medium", diff_summary["counts"]["medium"])
    dm3.metric("Hard", diff_summary["counts"]["hard"])
    dm4.metric("Unassigned", len(diff_summary["unassigned"]))

    if diff_errs:
        st.error("⚠ Conflict / invalid entries — must be resolved before export:\n\n" + "\n".join(f"- {e}" for e in diff_errs))
    elif diff_summary["unassigned"]:
        st.warning(f"⚠ {len(diff_summary['unassigned'])} question(s) do not have a difficulty level: {diff_summary['unassigned']}")
    else:
        st.success("✓ Every question has a difficulty level assigned.")

    if any(active_ps.difficulty_text.values()):
        if st.button("Clear Difficulty Assignments", key=f"clear_diff_{pid}"):
            active_ps.clear_difficulty()
            for lvl in ("easy", "medium", "hard"):
                st.session_state.pop(f"diff_{lvl}_{pid}", None)
            st.rerun()

    # ── Marking scheme ───────────────────────────────────────────────────
    st.markdown('<div class="qb-section-label">Marking Scheme</div>', unsafe_allow_html=True)
    marking = active_ps.marking_scheme_detected
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        max_marks_str = st.text_input("Maximum Marks", value="" if marking.get("max_marks") is None else str(marking["max_marks"]), key=f"max_marks_{pid}")
        cur_mm = _parse_optional_number(max_marks_str)
        st.caption(_format_source_caption(cur_mm, marking.get("max_marks"), marking.get("source", {}).get("max_marks", "not_specified")))
    with mc2:
        default_mpq = "" if marking.get("marks_per_question") is None else str(marking["marks_per_question"])
        mpq_str = st.text_input("Marks per Question", value=default_mpq, key=f"mpq_{pid}")
        cur_mpq = _parse_optional_number(mpq_str)
        st.caption(_format_source_caption(cur_mpq, marking.get("marks_per_question"), marking.get("source", {}).get("marks_per_question", "not_specified")))
    with mc3:
        default_neg = "" if marking.get("negative_marks") is None else str(marking["negative_marks"])
        neg_str = st.text_input("Negative Marking", value=default_neg, placeholder="blank", key=f"neg_{pid}")
        cur_neg = _parse_optional_number(neg_str)
        st.caption(_format_source_caption(cur_neg, marking.get("negative_marks"), marking.get("source", {}).get("negative_marks", "not_specified")))

    # ── Validate ─────────────────────────────────────────────────────────
    st.markdown('<div class="qb-section-label">Validate</div>', unsafe_allow_html=True)
    if st.button("✓ Validate Before Export", key=f"validate_btn_{pid}", type="primary"):
        effective = active_ps.effective_questions()
        fresh_report = validate_document(effective)
        current_marking = _current_marking_scheme(active_ps)
        blocking = compute_blocking_errors(effective, {
            "maximum_marks": current_marking.get("maximum_marks"),
            "marks_per_question": current_marking.get("marks_per_question"),
            "negative_marks": current_marking.get("negative_marks"),
        }, difficulty_errors=active_ps.difficulty_errors())
        unassigned_count = len(active_ps.difficulty_summary()["unassigned"])
        unconfirmed_images_count = len(active_ps.unconfirmed_image_questions())
        st.session_state[f"last_validation_{pid}"] = {
            "report": fresh_report, "blocking": blocking, "unassigned_difficulty": unassigned_count,
            "unconfirmed_images": unconfirmed_images_count,
        }

    last_val = st.session_state.get(f"last_validation_{pid}")
    if last_val:
        r, blocking = last_val["report"], last_val["blocking"]
        unassigned_n = last_val.get("unassigned_difficulty", 0)
        unconfirmed_images_n = last_val.get("unconfirmed_images", 0)
        all_clean = (
            not blocking and r["low_confidence"] == 0 and r["missing_answers"] == 0
            and r["duplicate_question_numbers"] == [] and r["missing_question_numbers"] == []
            and unassigned_n == 0 and unconfirmed_images_n == 0
        )
        if all_clean:
            st.success(
                "✓ Questions: {0}/{0}\n\n✓ No duplicate question numbers\n\n✓ No missing question numbers\n\n"
                "✓ Answers present\n\n✓ Options present where expected\n\n✓ Sections valid\n\n"
                "✓ Every question has a difficulty level\n\n✓ Marking scheme valid\n\n✓ All image questions have a confirmed image".format(r["total_detected"])
            )
        else:
            if blocking:
                st.error("Blocking errors — fix before export:\n\n" + "\n".join(f"- {e}" for e in blocking))
            warn_bits = []
            if r["low_confidence"] + r["missing_answers"]:
                warn_bits.append(f"{r['low_confidence'] + r['missing_answers']} question(s) still need review (low confidence and/or missing answers)")
            if unassigned_n:
                warn_bits.append(f"{unassigned_n} question(s) do not have a difficulty level")
            if unconfirmed_images_n:
                warn_bits.append(f"{unconfirmed_images_n} image question(s) have no confirmed image")
            if warn_bits:
                st.warning("⚠ " + "; ".join(warn_bits) + ".")

    # ── Generate ─────────────────────────────────────────────────────────
    st.markdown('<div class="qb-section-label">Generate Excel</div>', unsafe_allow_html=True)
    effective_now = active_ps.effective_questions()
    current_marking = _current_marking_scheme(active_ps)
    diff_errs_now = active_ps.difficulty_errors()
    blocking_now = compute_blocking_errors(effective_now, {
        "maximum_marks": current_marking.get("maximum_marks"),
        "marks_per_question": current_marking.get("marks_per_question"),
        "negative_marks": current_marking.get("negative_marks"),
    }, difficulty_errors=diff_errs_now)

    if blocking_now:
        st.error("Cannot generate Excel — fix these first:\n\n" + "\n".join(f"- {e}" for e in blocking_now))
    else:
        # Difficulty Level is folded into these fresh, throwaway copies
        # only — never into original_questions or the edited overlay.
        active_ps.apply_difficulty(effective_now)
        rep_now = validate_document([dict(x) for x in effective_now])
        unassigned_now = len(active_ps.difficulty_summary()["unassigned"])

        unconfirmed_images = active_ps.unconfirmed_image_questions()

        warn_bits = []
        if rep_now["low_confidence"]:
            warn_bits.append(f"{rep_now['low_confidence']} question(s) still need review")
        if unassigned_now:
            warn_bits.append(f"{unassigned_now} question(s) have no difficulty level (will export with a blank Difficulty Level cell)")
        if unconfirmed_images:
            warn_bits.append(
                f"{len(unconfirmed_images)} image question(s) have no confirmed image yet "
                f"(will export with a blank Image URL cell): {unconfirmed_images}"
            )
        if warn_bits:
            st.warning("⚠ " + "; ".join(warn_bits) + ".")

        gen_label = f"GENERATE EXCEL — {active_ps.span.paper_name}" if multi else "GENERATE EXCEL"
        gen_clicked = st.button("GENERATE EXCEL ANYWAY" if warn_bits else gen_label, type="primary", key=f"gen_btn_{pid}")
        if gen_clicked:
            out_name = f"{stem}_{active_ps.span.paper_name.replace(' ', '')}_question_bank.xlsx" if multi else f"{stem}_question_bank.xlsx"
            out_path = f"output/{out_name}"
            _run_excel_generation(pid, effective_now, current_marking, out_path)
            st.session_state[f"final_marking_scheme_{pid}"] = current_marking

    gen_state = st.session_state.get(_generation_state_key(pid), {"status": "idle"})
    # Current data can't produce a valid file right now (blocking_now) ->
    # a PREVIOUS success must never be shown as if it still applies to the
    # CURRENT (now-blocked) edited data — clear it rather than leave a
    # stale download button up.
    if blocking_now and gen_state.get("status") == "success":
        st.session_state[_generation_state_key(pid)] = {"status": "idle", "path": None, "error": None}
        gen_state = st.session_state[_generation_state_key(pid)]

    if gen_state["status"] == "failed":
        st.error(f"❌ Excel generation failed.\n\nReason: {gen_state['error']}")
    elif gen_state["status"] == "success":
        st.success("✅ Excel generated successfully! 📥 Download ready.")
        with open(gen_state["path"], "rb") as f:
            data = f.read()
        st.download_button(
            "DOWNLOAD EXCEL",
            data=data,
            file_name=Path(gen_state["path"]).name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            key=f"dl_btn_{pid}",
        )

# ── Generate All (ZIP) — multi-paper convenience ────────────────────────
if multi:
    st.header(f"{step_no}. Generate All Papers")
    ok_states = [ps for ps in paper_states if ps.span.status == "OK"]
    ambiguous_states = [ps for ps in paper_states if ps.span.status == "AMBIGUOUS"]
    if ambiguous_states:
        st.caption(f"⚠ {len(ambiguous_states)} paper(s) with an ambiguous boundary are excluded from this ZIP.")

    if st.button("GENERATE ALL (ZIP)", disabled=not ok_states):
        # Same CLICK -> clear -> "Generating..." -> write -> verify ->
        # store -> success/failure sequence as single-paper generation
        # (Fix 2) — one dict, one source of truth, never a stale success
        # left over from an earlier attempt.
        st.session_state["zip_generation"] = {"status": "generating"}
        with st.spinner("⏳ Generating ZIP. Please wait..."):
            try:
                Path("output").mkdir(exist_ok=True)
                buf = io.BytesIO()
                skipped = []
                image_warnings = []
                included = 0
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for ps in ok_states:
                        current_marking = _current_marking_scheme(ps)
                        effective = ps.effective_questions()
                        blocking = compute_blocking_errors(effective, {
                            "maximum_marks": current_marking.get("maximum_marks"),
                            "marks_per_question": current_marking.get("marks_per_question"),
                            "negative_marks": current_marking.get("negative_marks"),
                        }, difficulty_errors=ps.difficulty_errors())
                        if blocking:
                            skipped.append((ps.span.paper_name, blocking))
                            continue
                        ps.apply_difficulty(effective)
                        unconfirmed = ps.unconfirmed_image_questions()
                        if unconfirmed:
                            image_warnings.append((ps.span.paper_name, unconfirmed))
                        out_name = f"{stem}_{ps.span.paper_name.replace(' ', '')}_question_bank.xlsx"
                        out_path = f"output/{out_name}"
                        write_excel(effective, out_path, template_path=TEMPLATE_PATH, marking_scheme=current_marking)
                        if not Path(out_path).exists():
                            raise RuntimeError(f"Excel file was not found at {out_path} after generation.")
                        zf.write(out_path, arcname=out_name)
                        included += 1

                zip_bytes = buf.getvalue() if included else None
                if not zip_bytes:
                    raise RuntimeError(
                        "No papers could be included in the ZIP — every paper was skipped due to blocking errors."
                        if skipped else "No papers were available to include in the ZIP."
                    )
                st.session_state["zip_generation"] = {
                    "status": "success",
                    "bytes": zip_bytes,
                    "name": f"{stem}_all_papers.zip",
                    "included": included,
                    "skipped": skipped,
                    "image_warnings": image_warnings,
                }
            except Exception as e:
                st.session_state["zip_generation"] = {"status": "failed", "error": str(e)}

    zip_state = st.session_state.get("zip_generation", {"status": "idle"})

    if zip_state["status"] == "failed":
        st.error(f"❌ ZIP generation failed.\n\nReason: {zip_state['error']}")
    elif zip_state["status"] == "success":
        if zip_state.get("image_warnings"):
            for name, numbers in zip_state["image_warnings"]:
                st.warning(f"⚠ {name}: {len(numbers)} image question(s) exported with a blank Image URL cell (no confirmed image): {numbers}")

        if zip_state.get("skipped"):
            for name, errs in zip_state["skipped"]:
                st.error(f"⚠ {name} excluded from ZIP — blocking errors:\n\n" + "\n".join(f"- {e}" for e in errs))

        st.success(f"✅ ZIP generated successfully! {zip_state['included']} paper(s) included — 📥 Download ready.")
        st.download_button(
            "DOWNLOAD ALL (ZIP)",
            data=zip_state["bytes"],
            file_name=zip_state["name"],
            mime="application/zip",
            type="primary",
        )
