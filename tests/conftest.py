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

SSC_PDF = str(ROOT / "SSC Stenographer MTP-2.pdf")
AFCAT_PDF = str(ROOT / "templates" / "AFCAT_Mock_1-5_SP_2026_Final_File.pdf")


@pytest.fixture(scope="session")
def ssc_doc_result():
    return extract_document(SSC_PDF)


@pytest.fixture(scope="session")
def afcat_doc_result():
    return extract_document(AFCAT_PDF)
