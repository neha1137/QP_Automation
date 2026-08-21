"""
conftest.py — shared fixtures for the regression suite.

Extraction (native text + OCR fallback where needed) is the slow step, so
each real PDF is extracted at most once per test session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extractor import extract_document  # noqa: E402

# The real SSC Stenographer MTP-2 mock-test PDF used by the regression
# suite now lives at templates/sample_input.pdf (moved there; same file,
# never duplicated or renamed on disk). SSC_PDF is the single source of
# truth for its path — every test module that needs it should import
# this constant (or the sample_pdf_path fixture below) rather than
# hardcoding the filename or an absolute path, so the suite works
# regardless of where the repo is cloned.
SSC_PDF = str(ROOT / "templates" / "sample_input.pdf")
AFCAT_PDF = str(ROOT / "templates" / "AFCAT_Mock_1-5_SP_2026_Final_File.pdf")


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    path = ROOT / "templates" / "sample_input.pdf"
    assert path.exists(), f"Missing test fixture: {path}"
    return path


@pytest.fixture(scope="session")
def ssc_doc_result():
    return extract_document(SSC_PDF)


@pytest.fixture(scope="session")
def afcat_doc_result():
    return extract_document(AFCAT_PDF)
