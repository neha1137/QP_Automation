"""Both real PDFs in this repo are fully native-text — OCR must never
fire for either of them."""


def test_ssc_is_fully_native(ssc_doc_result):
    assert ssc_doc_result.ocr_page_count == 0
    assert ssc_doc_result.native_page_count == len(ssc_doc_result.pages)


def test_afcat_is_fully_native(afcat_doc_result):
    assert afcat_doc_result.ocr_page_count == 0
    assert afcat_doc_result.native_page_count == len(afcat_doc_result.pages)
