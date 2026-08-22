"""
api_uploader.py — Uploads extracted/manually-provided question images to
the Oswaal admin `upload_image` endpoint and returns the public URL that
the backend hands back.

Config mirrors the ai-education-admin frontend exactly, reading the SAME
`.env` keys it does (VITE_API_BASE_URL / VITE_X_API_KEY) so there is one
place to change the environment, not two:

    ai-education-admin/src/lib/mockTestApiClient.ts
        API_BASE_URL = VITE_API_BASE_URL
        API_KEY      = VITE_X_API_KEY
        getAuthHeaders() -> Authorization: Bearer <authToken cookie>
                            X-API-Key: <API_KEY>
        uploadImage(file) -> POST {BASE}/api/v1/admin/upload_image
                             FormData field name "image"
                          -> {status, image_url}

The one thing that cannot carry over is the bearer token: the admin app
reads it from a browser cookie set at login, and this is a Streamlit app
with no such cookie. So the token is obtained the same way the frontend
originally got it — POST /api/v1/login with credentials — and cached for
the process. A pasted OSWAAL_API_TOKEN still wins if present, which is
what makes short-lived manual tokens usable without a password.

Credentials come from `.env`/environment only — never hardcoded, never
logged, never returned to a caller. `upload_image` is the only function
that touches the network, and it degrades to a `status: "failed"` dict on
any error rather than letting an exception reach the Streamlit UI thread.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import requests

# Same default as the admin app's .env, used only if nothing is configured.
DEFAULT_BASE_URL = "https://stagingaibackend.oswaal360.com"
UPLOAD_PATH = "/api/v1/admin/upload_image"
LOGIN_PATH = "/api/v1/login"

# The backend derives the stored filename itself; this is only what the
# multipart part is labeled with, chosen from the real content type so a
# JPEG is never announced as a PNG.
_EXT_FOR_CONTENT_TYPE = {
    "image/png": "png",
    "image/jpeg": "jpg",
}

# Generous enough for a large rendered diagram on a slow link, bounded so
# a hung backend can't freeze a review session indefinitely.
UPLOAD_TIMEOUT_SECONDS = 60
LOGIN_TIMEOUT_SECONDS = 30

# Process-wide token cache. Streamlit reruns the whole script on every
# interaction, so without this a review session would re-login on each
# rerun. Lock guards against Streamlit's threaded reruns racing to login.
_token_lock = threading.Lock()
_cached_token: str | None = None


class MissingAPIConfigError(RuntimeError):
    """Raised when required configuration is absent — surfaced as a clear,
    specific message rather than letting a bare 401 reach the UI."""


def _strip(value: str | None) -> str:
    """.env values routinely carry trailing whitespace or wrapping quotes;
    an API key with a stray quote fails as a confusing 401."""
    if value is None:
        return ""
    # Strip whitespace again after unquoting: `KEY=" abc "` must yield
    # `abc`, not ` abc ` — a key with a leading space fails as a 401.
    return value.strip().strip('"').strip("'").strip()


def load_dotenv(path: str | Path = ".env") -> dict:
    """Minimal KEY=VALUE reader — no python-dotenv dependency added just
    for this. Real environment variables always win over the file, so a
    deployment can override without editing it.

    Silently returns {} when the file is absent: the app must still start
    (text extraction and Excel generation work without uploads).
    """
    values: dict[str, str] = {}
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = _strip(value)
    return values


def _setting(key: str, dotenv: dict, default: str = "") -> str:
    return _strip(os.environ.get(key)) or dotenv.get(key, "") or default


def normalize_base_url(url: str) -> str:
    """Mirrors the frontend's normalizeBaseURL: a trailing slash would
    produce a double slash in the request path."""
    return url.rstrip("/")


def get_config(dotenv_path: str | Path = ".env") -> dict:
    """API key is required (it identifies the vendor); the token is
    optional here because it can be minted by login() instead."""
    dotenv = load_dotenv(dotenv_path)

    base_url = normalize_base_url(_setting("VITE_API_BASE_URL", dotenv, DEFAULT_BASE_URL))
    api_key = _setting("VITE_X_API_KEY", dotenv)

    if not api_key:
        raise MissingAPIConfigError(
            "Missing VITE_X_API_KEY. Add it to this project's .env file "
            "(the same value the ai-education-admin app uses) or set it as "
            "an environment variable."
        )

    return {
        "base_url": base_url,
        "upload_url": base_url + UPLOAD_PATH,
        "login_url": base_url + LOGIN_PATH,
        "api_key": api_key,
        "token": _setting("OSWAAL_API_TOKEN", dotenv),
        "username": _setting("OSWAAL_USERNAME", dotenv),
        "password": _setting("OSWAAL_PASSWORD", dotenv),
    }


def login(config: dict) -> tuple[str | None, str | None]:
    """(token, error) from POST /api/v1/login — the same call the admin
    frontend's Login page makes, whose `access_token` it stores in the
    `authToken` cookie."""
    if not config["username"] or not config["password"]:
        return None, (
            "No API token available and no login credentials configured. "
            "Set OSWAAL_API_TOKEN to a bearer token, or set OSWAAL_USERNAME "
            "and OSWAAL_PASSWORD so a token can be obtained automatically."
        )
    try:
        response = requests.post(
            config["login_url"],
            headers={"Content-Type": "application/json", "accept": "application/json"},
            json={"username": config["username"], "password": config["password"]},
            timeout=LOGIN_TIMEOUT_SECONDS,
        )
    except Exception as e:
        return None, f"Could not reach login endpoint: {e}"

    if response.status_code != 200:
        return None, f"Login failed (HTTP {response.status_code}): {(response.text or '')[:200]}"

    try:
        token = response.json().get("access_token")
    except ValueError:
        return None, "Login endpoint returned a non-JSON response"

    if not token:
        return None, "Login response contained no access_token"
    return token, None


def get_token(config: dict, force_refresh: bool = False) -> tuple[str | None, str | None]:
    """(token, error). An explicitly configured token is used as-is; only
    when none is configured is one minted via login() and cached.

    `force_refresh` bypasses both, so an expired cached token can be
    replaced after a 401 instead of failing every subsequent upload.
    """
    global _cached_token

    if config["token"] and not force_refresh:
        return config["token"], None

    with _token_lock:
        if _cached_token and not force_refresh:
            return _cached_token, None
        token, error = login(config)
        if token:
            _cached_token = token
        return token, error


def reset_token_cache() -> None:
    """Drops the cached token — used by tests, and after a 401."""
    global _cached_token
    with _token_lock:
        _cached_token = None


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_filename(sha256_hex: str, content_type: str) -> str:
    ext = _EXT_FOR_CONTENT_TYPE.get(content_type, "png")
    return f"{sha256_hex}.{ext}"


def build_headers(token: str, api_key: str) -> dict:
    """Same two headers as the frontend's getAuthHeaders(). Content-Type is
    deliberately NOT set — requests generates the multipart boundary
    itself, and a hardcoded boundary produces a body the server cannot
    parse."""
    return {"Authorization": f"Bearer {token}", "X-API-Key": api_key}


def parse_response(payload) -> tuple[str | None, str | None]:
    """(url, error). Reads the documented `image_url` key (the shape
    mockTestApiClient.ts declares) and common alternates, so a backend
    field rename degrades to a clear error instead of silently writing an
    empty URL into the Excel column."""
    if not isinstance(payload, dict):
        return None, f"Unexpected response type from upload API: {type(payload).__name__}"
    for key in ("image_url", "url", "public_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), None
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("image_url", "url", "public_url"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), None
    return None, f"Upload API response contained no image URL: {payload}"


def _failed(error: str, sha: str, key: str | None = None) -> dict:
    return {"status": "failed", "url": None, "key": key, "sha256": sha,
            "error": error, "reused": False}


def _post_image(config: dict, token: str, filename: str, image_bytes: bytes,
                content_type: str):
    return requests.post(
        config["upload_url"],
        headers=build_headers(token, config["api_key"]),
        files={"image": (filename, image_bytes, content_type)},
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )


def upload_image(image_bytes: bytes, content_type: str, cache: dict | None = None) -> dict:
    """
    Content-hash-deduplicated upload — posts ONE image per call, waits for
    the response, and returns the public URL the backend assigned.

    `cache`, if given, is expected to be a dict living in the caller's
    session state (sha256 -> {"url", "key"}), reused across Streamlit
    reruns so editing an unrelated field, or revisiting the same question,
    never re-uploads the same bytes.

    Returns a dict, always with these keys — never raises:
        status:   "uploaded" | "failed"
        url:      public URL, or None on failure
        key:      the filename sent to the backend (for debugging)
        sha256:   hex digest of image_bytes
        error:    human-readable message, or None on success
        reused:   True on an in-session cache hit
    """
    sha = compute_sha256(image_bytes)

    if cache is not None and sha in cache:
        cached = cache[sha]
        return {"status": "uploaded", "url": cached["url"], "key": cached["key"],
                "sha256": sha, "error": None, "reused": True}

    try:
        config = get_config()
    except MissingAPIConfigError as e:
        return _failed(str(e), sha)

    token, error = get_token(config)
    if token is None:
        return _failed(error, sha)

    filename = build_filename(sha, content_type)

    try:
        response = _post_image(config, token, filename, image_bytes, content_type)
        # A cached token outlives its expiry silently; re-login once and
        # retry rather than failing every upload for the rest of the
        # session. Only worth doing when the token wasn't pinned by config.
        if response.status_code in (401, 403) and not config["token"]:
            token, error = get_token(config, force_refresh=True)
            if token is None:
                return _failed(f"Authentication expired and re-login failed: {error}", sha, filename)
            response = _post_image(config, token, filename, image_bytes, content_type)
    except requests.exceptions.Timeout:
        return _failed(f"Upload timed out after {UPLOAD_TIMEOUT_SECONDS}s", sha, filename)
    except requests.exceptions.ConnectionError as e:
        return _failed(f"Could not reach upload API: {e}", sha, filename)
    except Exception as e:
        return _failed(f"Upload request failed: {e}", sha, filename)

    if response.status_code != 200:
        # Body is truncated: an HTML error page from a proxy would
        # otherwise dump kilobytes of markup into a Streamlit error box.
        return _failed(
            f"Upload API returned HTTP {response.status_code}: {(response.text or '')[:300]}",
            sha, filename,
        )

    try:
        payload = response.json()
    except ValueError:
        return _failed(
            f"Upload API returned non-JSON response: {(response.text or '')[:300]}", sha, filename
        )

    url, error = parse_response(payload)
    if url is None:
        return _failed(error, sha, filename)

    if cache is not None:
        cache[sha] = {"url": url, "key": filename}
    return {"status": "uploaded", "url": url, "key": filename, "sha256": sha,
            "error": None, "reused": False}
