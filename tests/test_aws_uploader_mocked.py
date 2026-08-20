"""
tests/test_aws_uploader_mocked.py — aws_uploader.py, entirely mocked.

No test in this file makes a real network call or requires real AWS
credentials — every boto3 S3 client is replaced with an in-memory fake.
This is a deliberate, explicit separation from a real S3 integration test
(which this project does not have, since no AWS credentials are available
in this environment — see the V3 planning notes).
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

import aws_uploader


class FakeS3Client:
    """Records calls; simulates a real bucket's HEAD/PUT semantics with an
    in-memory dict, including a real ClientError with a 404 code on a
    HEAD for an object that hasn't been PUT yet — matching what
    aws_uploader.py actually branches on."""

    def __init__(self):
        self.objects = {}  # key -> {"Body":..., "ContentType":...}
        self.head_calls = []
        self.put_calls = []

    def head_object(self, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key]["Body"])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.put_calls.append((Bucket, Key, Body, ContentType))
        self.objects[Key] = {"Body": Body, "ContentType": ContentType}


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_S3_BUCKET", "ai-education-s3-public-media")
    monkeypatch.setenv("AWS_S3_KEY_PREFIX", "mock-tests/images/")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(aws_uploader, "build_client", lambda config: client)
    return client


# -- missing configuration ---------------------------------------------------

def test_missing_aws_config_surfaces_clear_error(monkeypatch):
    for var in aws_uploader.REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    result = aws_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert "AWS_ACCESS_KEY_ID" in result["error"] or "Missing required AWS configuration" in result["error"]
    assert result["url"] is None


# -- correct bucket / prefix / content-type / URL format --------------------

def test_upload_uses_correct_bucket_prefix_and_content_type(aws_env, fake_client):
    result = aws_uploader.upload_image(b"png-bytes", "image/png")
    assert result["status"] == "uploaded"
    assert len(fake_client.put_calls) == 1
    bucket, key, body, content_type = fake_client.put_calls[0]
    assert bucket == "ai-education-s3-public-media"
    assert key.startswith("mock-tests/images/")
    assert key.endswith(".png")
    assert content_type == "image/png"
    assert body == b"png-bytes"


def test_public_url_matches_reference_format(aws_env, fake_client):
    result = aws_uploader.upload_image(b"jpeg-bytes", "image/jpeg")
    sha = aws_uploader.compute_sha256(b"jpeg-bytes")
    expected = f"https://ai-education-s3-public-media.s3.ap-south-1.amazonaws.com/mock-tests/images/{sha}.jpg"
    assert result["url"] == expected


def test_jpeg_content_type_never_mislabeled_as_png(aws_env, fake_client):
    result = aws_uploader.upload_image(b"some-jpeg-bytes", "image/jpeg")
    _, key, _, content_type = fake_client.put_calls[0]
    assert content_type == "image/jpeg"
    assert key.endswith(".jpg")


# -- SHA-256 deduplication ----------------------------------------------------

def test_head_before_put_skips_redundant_upload(aws_env, fake_client):
    data = b"identical-bytes"
    r1 = aws_uploader.upload_image(data, "image/png")
    assert len(fake_client.put_calls) == 1
    # Second upload of the SAME bytes, fresh call (no in-memory cache passed)
    # relies on HEAD finding the object already there.
    r2 = aws_uploader.upload_image(data, "image/png")
    assert len(fake_client.put_calls) == 1  # still just one PUT — HEAD found it
    assert len(fake_client.head_calls) == 2
    assert r1["url"] == r2["url"]
    assert r2["reused"] is True


def test_in_session_cache_avoids_even_the_head_call(aws_env, fake_client):
    cache = {}
    data = b"cache-me"
    aws_uploader.upload_image(data, "image/png", cache=cache)
    assert len(fake_client.put_calls) == 1
    assert len(fake_client.head_calls) == 1

    aws_uploader.upload_image(data, "image/png", cache=cache)
    # in-session cache hit — no additional HEAD or PUT at all
    assert len(fake_client.put_calls) == 1
    assert len(fake_client.head_calls) == 1


def test_different_images_get_different_keys(aws_env, fake_client):
    r1 = aws_uploader.upload_image(b"image-one", "image/png")
    r2 = aws_uploader.upload_image(b"image-two", "image/png")
    assert r1["key"] != r2["key"]
    assert r1["url"] != r2["url"]
    assert len(fake_client.put_calls) == 2


# -- failure handling ---------------------------------------------------------

def test_put_object_failure_is_surfaced_not_raised(aws_env, monkeypatch):
    class FailingClient(FakeS3Client):
        def put_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject")

    monkeypatch.setattr(aws_uploader, "build_client", lambda config: FailingClient())
    result = aws_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "failed"
    assert result["error"] is not None
    assert result["url"] is None


def test_head_object_403_does_not_block_upload_attempt(aws_env, monkeypatch):
    """A principal that can PutObject but not HeadObject/GetObject should
    still succeed — HEAD is a best-effort optimization only, never a hard
    requirement (see aws_uploader.py's comment on this)."""

    class ForbiddenHeadClient(FakeS3Client):
        def head_object(self, Bucket, Key):
            raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")

    client = ForbiddenHeadClient()
    monkeypatch.setattr(aws_uploader, "build_client", lambda config: client)
    result = aws_uploader.upload_image(b"data", "image/png")
    assert result["status"] == "uploaded"
    assert len(client.put_calls) == 1
