"""
tests/test_api_uploader_mocked.py — api_uploader.py, entirely mocked.

No real network call is ever made here: `requests.post` is replaced, so
these tests assert the branching api_uploader.py actually does (auth
headers, multipart filename, response parsing, dedup, every failure
path) without depending on the staging backend being reachable.
"""

import json

import pytest
import requests

import api_uploader


@pytest.fixture(autouse=True)
def clean_token_cache():
    """The token cache is process-wide, so it must not leak between tests."""
    api_uploader.reset_token_cache()
    yield
    api_uploader.reset_token_cache()


@pytest.fixture
def api_env(monkeypatch):
    """Config comes from the same .env keys the ai-education-admin app
    uses; real env vars take precedence, which is what these tests set."""
    monkeypatch.setenv("VITE_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("VITE_X_API_KEY", "test-api-key")
    monkeypatch.setenv("OSWAAL_API_TOKEN", "test-token")
    for var in ("OSWAAL_USERNAME", "OSWAAL_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def recorder(monkeypatch):
    """Captures the outgoing request and returns a configurable response."""
    calls = []

    def fake_post(url, headers=None, files=None, timeout=None):
        calls.append({"url": url, "headers": headers, "files": files, "timeout": timeout})
        return fake_post.response

    fake_post.response = FakeResponse(
        payload={"status": "uploaded", "image_url": "https://cdn.example.com/img/abc.png"}
    )
    fake_post.calls = calls
    monkeypatch.setattr(api_uploader.requests, "post", fake_post)
    return fake_post


def test_missing_api_key_returns_failed_not_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env to fall back to
    for var in ("VITE_X_API_KEY", "OSWAAL_API_TOKEN", "OSWAAL_USERNAME", "OSWAAL_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert result["url"] is None
    assert "VITE_X_API_KEY" in result["error"]


def test_no_token_and_no_credentials_returns_failed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VITE_X_API_KEY", "test-api-key")
    for var in ("OSWAAL_API_TOKEN", "OSWAAL_USERNAME", "OSWAAL_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert "OSWAAL_USERNAME" in result["error"]


def test_config_reads_dotenv_file(monkeypatch, tmp_path):
    """The project's existing .env — the same file ai-education-admin
    uses — is the config source; no duplicate variable names."""
    monkeypatch.chdir(tmp_path)
    for var in ("VITE_API_BASE_URL", "VITE_X_API_KEY", "OSWAAL_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text(
        "VITE_API_BASE_URL=https://staging.example.com\n"
        "VITE_X_API_KEY=key-from-dotenv\n"
    )
    config = api_uploader.get_config()
    assert config["api_key"] == "key-from-dotenv"
    assert config["upload_url"] == "https://staging.example.com/api/v1/admin/upload_image"


def test_real_env_var_overrides_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("VITE_X_API_KEY=key-from-dotenv\n")
    monkeypatch.setenv("VITE_X_API_KEY", "key-from-environment")
    assert api_uploader.get_config()["api_key"] == "key-from-environment"


def test_dotenv_values_are_stripped_of_quotes_and_whitespace(tmp_path):
    (tmp_path / ".env").write_text('VITE_X_API_KEY="  quoted-key  " \n')
    assert api_uploader.load_dotenv(tmp_path / ".env")["VITE_X_API_KEY"] == "quoted-key"


def test_missing_dotenv_file_is_not_an_error(tmp_path):
    assert api_uploader.load_dotenv(tmp_path / "nope.env") == {}


def test_base_url_trailing_slash_does_not_double_up(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VITE_API_BASE_URL", "https://api.example.com/")
    monkeypatch.setenv("VITE_X_API_KEY", "k")
    assert api_uploader.get_config()["upload_url"] == (
        "https://api.example.com/api/v1/admin/upload_image"
    )


def test_successful_upload_returns_backend_url(api_env, recorder):
    result = api_uploader.upload_image(b"png-bytes", "image/png")
    assert result["status"] == "uploaded"
    assert result["url"] == "https://cdn.example.com/img/abc.png"
    assert result["error"] is None
    assert result["reused"] is False
    assert result["sha256"] == api_uploader.compute_sha256(b"png-bytes")


def test_request_carries_auth_headers_and_endpoint(api_env, recorder):
    api_uploader.upload_image(b"png-bytes", "image/png")
    call = recorder.calls[0]
    assert call["url"] == "https://api.example.com/api/v1/admin/upload_image"
    assert call["headers"]["Authorization"] == "Bearer test-token"
    assert call["headers"]["X-API-Key"] == "test-api-key"
    # requests must build the multipart boundary itself — a hardcoded
    # Content-Type would produce a body the server cannot parse.
    assert "Content-Type" not in call["headers"]
    assert call["timeout"] == api_uploader.UPLOAD_TIMEOUT_SECONDS


def test_multipart_field_name_and_content_type(api_env, recorder):
    api_uploader.upload_image(b"jpeg-bytes", "image/jpeg")
    filename, data, content_type = recorder.calls[0]["files"]["image"]
    assert data == b"jpeg-bytes"
    assert content_type == "image/jpeg"
    assert filename.endswith(".jpg")
    assert filename.startswith(api_uploader.compute_sha256(b"jpeg-bytes"))


def test_png_filename_uses_png_extension(api_env, recorder):
    api_uploader.upload_image(b"png-bytes", "image/png")
    filename = recorder.calls[0]["files"]["image"][0]
    assert filename.endswith(".png")


def test_base_url_is_overridable_by_env(api_env, recorder, monkeypatch):
    monkeypatch.setenv("VITE_API_BASE_URL", "https://prod.example.com")
    api_uploader.upload_image(b"data", "image/png")
    assert recorder.calls[0]["url"] == "https://prod.example.com/api/v1/admin/upload_image"


def test_cache_hit_skips_second_network_call(api_env, recorder):
    cache = {}
    r1 = api_uploader.upload_image(b"same-bytes", "image/png", cache=cache)
    r2 = api_uploader.upload_image(b"same-bytes", "image/png", cache=cache)
    assert len(recorder.calls) == 1        # exactly one real upload
    assert r2["reused"] is True
    assert r2["url"] == r1["url"]


def test_different_bytes_upload_separately(api_env, recorder):
    cache = {}
    api_uploader.upload_image(b"image-one", "image/png", cache=cache)
    api_uploader.upload_image(b"image-two", "image/png", cache=cache)
    assert len(recorder.calls) == 2
    assert len(cache) == 2


def test_non_200_returns_failed_with_status_in_error(api_env, recorder):
    recorder.response = FakeResponse(status_code=401, text="Unauthorized")
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert result["url"] is None
    assert "401" in result["error"]


def test_non_json_response_returns_failed(api_env, recorder):
    recorder.response = FakeResponse(status_code=200, payload=None, text="<html>gateway</html>")
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert "non-JSON" in result["error"]


def test_json_without_url_returns_failed_not_none_url(api_env, recorder):
    """A backend field rename must fail loudly, never silently write an
    empty URL into the Excel column."""
    recorder.response = FakeResponse(payload={"status": "uploaded"})
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert result["url"] is None
    assert "no image URL" in result["error"]


def test_failed_upload_is_not_cached(api_env, recorder):
    cache = {}
    recorder.response = FakeResponse(status_code=500, text="boom")
    api_uploader.upload_image(b"data", "image/png", cache=cache)
    assert cache == {}


def test_timeout_returns_failed(api_env, monkeypatch):
    def raise_timeout(*a, **k):
        raise requests.exceptions.Timeout()
    monkeypatch.setattr(api_uploader.requests, "post", raise_timeout)
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert "timed out" in result["error"]


def test_connection_error_returns_failed(api_env, monkeypatch):
    def raise_conn(*a, **k):
        raise requests.exceptions.ConnectionError("dns failure")
    monkeypatch.setattr(api_uploader.requests, "post", raise_conn)
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert "Could not reach upload API" in result["error"]


@pytest.mark.parametrize("payload", [
    {"url": "https://cdn.example.com/a.png"},
    {"public_url": "https://cdn.example.com/a.png"},
    {"data": {"image_url": "https://cdn.example.com/a.png"}},
])
def test_alternate_url_keys_are_accepted(api_env, recorder, payload):
    recorder.response = FakeResponse(payload=payload)
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "uploaded"
    assert result["url"] == "https://cdn.example.com/a.png"


# ── Automatic login (the admin app reads this token from a cookie; a
#    Streamlit app has none, so it mints one the same way Login.tsx does) ──

@pytest.fixture
def login_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VITE_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("VITE_X_API_KEY", "test-api-key")
    monkeypatch.delenv("OSWAAL_API_TOKEN", raising=False)
    monkeypatch.setenv("OSWAAL_USERNAME", "oswaal360")
    monkeypatch.setenv("OSWAAL_PASSWORD", "secret")


def _routing_post(monkeypatch, login_response, upload_response):
    """Routes /login and /upload_image to separate canned responses."""
    calls = []

    def fake_post(url, headers=None, files=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "files": files, "json": json})
        return login_response if url.endswith("/login") else upload_response

    monkeypatch.setattr(api_uploader.requests, "post", fake_post)
    return calls


def test_logs_in_when_no_token_configured(login_env, monkeypatch):
    calls = _routing_post(
        monkeypatch,
        FakeResponse(payload={"access_token": "minted-token"}),
        FakeResponse(payload={"status": "uploaded", "image_url": "https://cdn.example.com/x.png"}),
    )
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "uploaded"
    assert calls[0]["url"] == "https://api.example.com/api/v1/login"
    assert calls[0]["json"] == {"username": "oswaal360", "password": "secret"}
    assert calls[1]["headers"]["Authorization"] == "Bearer minted-token"


def test_token_is_cached_across_uploads(login_env, monkeypatch):
    calls = _routing_post(
        monkeypatch,
        FakeResponse(payload={"access_token": "minted-token"}),
        FakeResponse(payload={"status": "uploaded", "image_url": "https://cdn.example.com/x.png"}),
    )
    api_uploader.upload_image(b"one", "image/png")
    api_uploader.upload_image(b"two", "image/png")
    assert sum(1 for c in calls if c["url"].endswith("/login")) == 1


def test_failed_login_returns_clear_error(login_env, monkeypatch):
    _routing_post(
        monkeypatch,
        FakeResponse(status_code=401, text="bad credentials"),
        FakeResponse(payload={}),
    )
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert "Login failed" in result["error"]


def test_expired_token_triggers_relogin_and_retry(login_env, monkeypatch):
    """A cached token outliving its expiry must not fail every remaining
    upload in the session."""
    state = {"uploads": 0}

    def fake_post(url, headers=None, files=None, json=None, timeout=None):
        if url.endswith("/login"):
            return FakeResponse(payload={"access_token": "fresh-token"})
        state["uploads"] += 1
        if state["uploads"] == 1:
            return FakeResponse(status_code=401, text="expired")
        return FakeResponse(payload={"status": "uploaded", "image_url": "https://cdn.example.com/x.png"})

    monkeypatch.setattr(api_uploader.requests, "post", fake_post)
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "uploaded"
    assert state["uploads"] == 2


def test_pinned_token_is_not_retried_on_401(api_env, recorder):
    """An explicitly configured token must not silently trigger a login."""
    recorder.response = FakeResponse(status_code=401, text="nope")
    result = api_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert "401" in result["error"]
    assert len(recorder.calls) == 1
