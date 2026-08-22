# Mock Test PDF → Excel Question Bank Generator

Upload a mock test / question paper PDF; the app extracts questions,
sections, passages, and answers/explanations, and populates the provided
Excel template.

## Run

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py --server.port 8080
```

The app is configured (via `.streamlit/config.toml`) to default to port
8080 even without the flag — `streamlit run app.py` and
`python3 -m streamlit run app.py` both work too.

## Pipeline

```
PDF → page-level extraction (native / OCR) → question/section/passage
parser → answer-key extraction → validation → Excel template
```

See `extractor.py`, `parser.py`, `validator.py`, `excel_writer.py`.

## OCR Architecture

Each page is extracted independently — OCR is a per-page fallback, never a
default:

```
PDF page
   │
   ▼
Native text extraction (PyMuPDF, page.get_text())
   │
   ▼
Quality check (compute_quality_score in extractor.py)
   — length, printable/alnum ratio, word coherence, presence of
     question/option numbering patterns. NOT just "len(text) > 0".
   │
   ├── score ≥ 0.55 ──► use native text (fast, exact source text, no OCR)
   │
   └── score < 0.55 ──► render page to an image (PyMuPDF, 280 DPI)
                          │
                          ▼
                        Tesseract OCR (via pytesseract, --psm 6)
                          │
                          ▼
                        light whitespace/line-break normalization
                          (never spelling/word "autocorrection" —
                          that could silently change a question's meaning)
```

A 21-page, fully native-text PDF processes as **21 native / 0 OCR** —
OCR never runs unless a specific page's native extraction actually fails
the quality check. A mixed document (some scanned pages, some not) is
handled per page: `native=7, ocr=13` is normal for a partly-scanned paper.

**Engine**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), a
free, local, open-source OCR engine — not a cloud API. `pytesseract` is
only a thin Python wrapper: it shells out to the `tesseract` binary on
your system and parses its output. No project code calls Gemini, OpenAI,
Anthropic, or any other cloud/LLM API for extraction or answer-solving —
answers and explanations are read directly out of the PDF's own answer
key (see `parser.py`'s `ANSWER_KEY_RE`), never independently solved.

**No API key, no internet required.** Everything runs locally once
Tesseract is installed. Rendering is done with PyMuPDF (also local); OCR
runs as a local subprocess.

**Install** (system binary, separate from the `pytesseract` pip package):

| OS | Command |
|---|---|
| macOS | `brew install tesseract` |
| Ubuntu/Debian | `sudo apt-get install tesseract-ocr` |
| Windows | https://github.com/UB-Mannheim/tesseract/wiki |

**If Tesseract isn't installed**: native-text PDFs are completely
unaffected — the app works normally, since OCR is never invoked for them.
If a page genuinely needs OCR (scanned/image-only) and Tesseract can't be
found, `extractor.py` raises a `TesseractNotAvailableError` with the
install instructions above, surfaced as a clear message in the Streamlit
UI (`app.py` also shows an OCR-availability indicator on startup) —
rather than a silent blank page or a raw subprocess stack trace.

DPI (280) and Tesseract config (`--psm 6`, chosen because the default
automatic page segmentation reorders short left-aligned option lines in
dense question-paper layouts) are set in `extractor.py`
(`OCR_RENDER_DPI`, `ocr_config`).

## Image Upload

Detected (or manually uploaded) question images are pushed **one at a
time** to the Oswaal admin upload API, which stores the file and returns
its public URL. That URL is written straight into the Excel image
columns — the app never constructs a URL itself.

```
PDF region → image_locator → image_extractor (PNG bytes)
   → api_uploader.upload_image()  ── POST multipart ──►  /api/v1/admin/upload_image
   ◄── {"status":"uploaded","image_url":"https://..."} ──┘
   → PaperState.set_image()  →  Excel "Image URL" columns
```

Uploads are deduplicated by SHA-256 of the image bytes within a session,
so re-visiting a question or editing an unrelated field never re-uploads
the same image.

### Configuration

Config lives in `.env` and uses the **same keys as the
`ai-education-admin` frontend**, so one file serves both projects
(`src/lib/mockTestApiClient.ts` reads exactly these):

```bash
VITE_API_BASE_URL=https://stagingaibackend.oswaal360.com
VITE_X_API_KEY=<x-api-key>
```

The bearer token is the one thing that cannot carry over: the admin app
reads it from the `authToken` browser cookie set at login, and Streamlit
has no such cookie. So provide it one of two ways —

```bash
# Option A — paste a token (expires; see the JWT's `exp`)
OSWAAL_API_TOKEN=<bearer token>

# Option B — let it log in automatically via POST /api/v1/login,
#            the same call the admin Login page makes. Recommended:
#            the token is minted, cached, and refreshed on expiry.
OSWAAL_USERNAME=<username>
OSWAAL_PASSWORD=<password>
```

Real environment variables override `.env`. See `.env.example`.
`.env` is gitignored — it holds a live key and must not be committed.

If uploads are unconfigured they fail with a clear message in the UI;
text extraction and Excel generation still work.
