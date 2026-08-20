"""
image_extractor.py — Turns an image_locator.RegionCandidate into actual
bytes: either the original embedded raster (lossless, when the region is
covered by exactly one raster image) or a rendered bitmap of the page
region (the common case for this project's real PDFs — see
image_locator.py's module docstring for the empirical vector-vs-raster
finding).

Kept separate from image_locator.py: locator answers "where" (page
coordinates), this module answers "what bytes, at what quality, with what
hash" — aws_uploader.py only ever depends on this module's output, never
on PyMuPDF directly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import fitz  # PyMuPDF

# 220 DPI: crisp enough to read fine diagram labels/axis numbers on a
# rendered crop without producing unreasonably large PNGs for a
# question-sized region (a typical few-hundred-point-tall band renders to
# well under 1MB at this DPI).
RENDER_DPI = 220

# Small padding (points) added around a tight content bbox before
# rendering, so line strokes/anti-aliasing right at the union's edge
# aren't clipped.
_CROP_PADDING = 4.0


@dataclass
class ImageArtifact:
    destination: str
    region: tuple | None
    confidence: str
    source_type: str
    reason: str
    content_type: str
    image_bytes: bytes
    sha256: str


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _padded_clip(bbox: tuple, page_rect) -> "fitz.Rect":
    x0, y0, x1, y1 = bbox
    r = fitz.Rect(
        max(page_rect.x0, x0 - _CROP_PADDING),
        max(page_rect.y0, y0 - _CROP_PADDING),
        min(page_rect.x1, x1 + _CROP_PADDING),
        min(page_rect.y1, y1 + _CROP_PADDING),
    )
    return r


def render_region(page, bbox: tuple, dpi: int = RENDER_DPI) -> bytes:
    """Renders the page region as a PNG — works regardless of whether the
    content underneath is vector paths, raster images, or a mix; it's a
    screenshot of the area, not a re-interpretation of PDF objects."""
    clip = _padded_clip(bbox, page.rect)
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    return pix.tobytes("png")


def extract_raster(page, xref: int) -> tuple[bytes, str] | None:
    """Lossless path: pulls the original embedded image bytes straight out
    of the PDF (no re-rendering/recompression). Returns (bytes, ext) or
    None if extraction fails for any reason (caller falls back to
    render_region)."""
    try:
        doc = page.parent
        info = doc.extract_image(xref)
        return info["image"], info["ext"]
    except Exception:
        return None


_EXT_TO_CONTENT_TYPE = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


def build_artifact(page, candidate) -> ImageArtifact | None:
    """
    Turns a locator RegionCandidate into an ImageArtifact with real bytes.
    Prefers the lossless embedded-raster path when the candidate says
    exactly one raster image covers the region; otherwise renders the
    region as PNG (the path that actually fires for this project's real
    sample PDFs — both are vector-diagram-only, per image_locator.py's
    docstring).

    Returns None only if rendering itself fails (e.g. a degenerate/empty
    bbox) — never returns a placeholder or blank image.
    """
    bbox = candidate.bbox
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    if candidate.source_type == "embedded_raster" and candidate.raster_xref is not None:
        raw = extract_raster(page, candidate.raster_xref)
        if raw is not None:
            data, ext = raw
            content_type = _EXT_TO_CONTENT_TYPE.get(ext.lower())
            if content_type:
                return ImageArtifact(
                    destination=candidate.destination,
                    region=bbox,
                    confidence=candidate.confidence,
                    source_type="embedded_raster",
                    reason=candidate.reason,
                    content_type=content_type,
                    image_bytes=data,
                    sha256=compute_sha256(data),
                )
        # fall through to render_region if lossless extraction didn't pan out

    try:
        data = render_region(page, bbox)
    except Exception:
        return None
    if not data:
        return None
    return ImageArtifact(
        destination=candidate.destination,
        region=bbox,
        confidence=candidate.confidence,
        source_type="pdf_region_render",
        reason=candidate.reason,
        content_type="image/png",
        image_bytes=data,
        sha256=compute_sha256(data),
    )


def hash_and_type_for_upload(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """For manually uploaded files: (sha256, content_type), derived from
    the actual uploaded filename's extension — never blindly labeled
    'image/jpeg' regardless of the real format."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = _EXT_TO_CONTENT_TYPE.get(ext, "image/png")
    return compute_sha256(file_bytes), content_type
