"""
aws_uploader.py — S3 upload for extracted/manually-provided question
images, with SHA-256-based deduplication.

No existing AWS/boto3 code was found anywhere in this repository (grepped
exhaustively — see V3 planning notes); this module is a from-scratch
addition, deliberately isolated so:
  - credentials are read from environment variables only, never hardcoded,
    never logged, never returned to a caller;
  - the public-URL assumption (matching the reference example's shape) is
    confined to `build_object_url()` alone — swapping to a presigned-URL
    scheme later (if the bucket turns out not to be public-read, which is
    UNVERIFIED as of this implementation) means changing this one function,
    not any caller.

Every function here is pure/deterministic except `upload_image`, which is
the only one that talks to the network — and even that degrades to a
clear `status: "failed"` dict on any error, never an uncaught exception
reaching the Streamlit UI thread.
"""

from __future__ import annotations

import hashlib
import os

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

REQUIRED_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_S3_BUCKET",
]
DEFAULT_KEY_PREFIX = "mock-tests/images/"

_EXT_FOR_CONTENT_TYPE = {
    "image/png": "png",
    "image/jpeg": "jpg",
}


class MissingAWSConfigError(RuntimeError):
    """Raised when one or more required AWS_* environment variables are
    absent — surfaced as a clear, specific message rather than letting a
    bare boto3 NoCredentialsError/KeyError reach the UI."""


def get_config() -> dict:
    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        raise MissingAWSConfigError(
            "Missing required AWS configuration: " + ", ".join(missing) + ". "
            "Set these as environment variables (or Streamlit secrets) before "
            "uploading images. See README for the exact list."
        )
    return {
        "access_key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region": os.environ["AWS_REGION"],
        "bucket": os.environ["AWS_S3_BUCKET"],
        "prefix": os.environ.get("AWS_S3_KEY_PREFIX", DEFAULT_KEY_PREFIX),
        "session_token": os.environ.get("AWS_SESSION_TOKEN") or None,
    }


def build_client(config: dict):
    kwargs = dict(
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        region_name=config["region"],
    )
    if config.get("session_token"):
        kwargs["aws_session_token"] = config["session_token"]
    return boto3.client("s3", **kwargs)


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_object_key(sha256_hex: str, content_type: str, prefix: str) -> str:
    ext = _EXT_FOR_CONTENT_TYPE.get(content_type, "png")
    return f"{prefix.rstrip('/')}/{sha256_hex}.{ext}"


def build_object_url(bucket: str, region: str, key: str) -> str:
    """Permanent public-URL form, matching the reference example's shape
    exactly. NOTE: whether this bucket actually grants anonymous
    s3:GetObject on this prefix has NOT been verified as of this
    implementation — see the module docstring. If it turns out not to be
    public, this is the only function that needs to change (to
    `client.generate_presigned_url(...)` instead)."""
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def upload_image(image_bytes: bytes, content_type: str, cache: dict | None = None) -> dict:
    """
    Content-hash-deduplicated upload. `cache`, if given, is expected to be
    a dict living in the caller's session state (sha256 -> {"url","key"})
    — reused across Streamlit reruns so editing an unrelated field, or
    revisiting the same question, never re-uploads the same bytes.

    Returns a dict, always with these keys — never raises:
        status:   "uploaded" | "failed"
        url:      public URL, or None on failure
        key:      the S3 object key (even on some failures, for debugging)
        sha256:   hex digest of image_bytes
        error:    human-readable message, or None on success
        reused:   True if this exact hash's object already existed
                   (in-session cache hit OR a HEAD-before-PUT hit)
    """
    sha = compute_sha256(image_bytes)

    if cache is not None and sha in cache:
        cached = cache[sha]
        return {
            "status": "uploaded", "url": cached["url"], "key": cached["key"],
            "sha256": sha, "error": None, "reused": True,
        }

    try:
        config = get_config()
    except MissingAWSConfigError as e:
        return {"status": "failed", "url": None, "key": None, "sha256": sha, "error": str(e), "reused": False}

    key = build_object_key(sha, content_type, config["prefix"])

    try:
        client = build_client(config)
    except Exception as e:
        return {"status": "failed", "url": None, "key": key, "sha256": sha, "error": f"Could not create S3 client: {e}", "reused": False}

    # HEAD is a best-effort optimization only: any failure here (404 = not
    # found, 403 = this principal can't HEAD but might still PUT, or a
    # transient error) just means "assume it doesn't exist yet" — it never
    # blocks the actual upload attempt, since the deterministic
    # content-hash key makes a redundant PUT harmless (same bytes, same
    # key, same result) rather than a duplicate object.
    already_exists = False
    try:
        client.head_object(Bucket=config["bucket"], Key=key)
        already_exists = True
    except ClientError:
        already_exists = False
    except Exception:
        already_exists = False

    try:
        if not already_exists:
            client.put_object(
                Bucket=config["bucket"], Key=key, Body=image_bytes, ContentType=content_type,
            )
        url = build_object_url(config["bucket"], config["region"], key)
        if cache is not None:
            cache[sha] = {"url": url, "key": key}
        return {
            "status": "uploaded", "url": url, "key": key, "sha256": sha,
            "error": None, "reused": already_exists,
        }
    except NoCredentialsError as e:
        return {"status": "failed", "url": None, "key": key, "sha256": sha, "error": f"AWS credentials rejected: {e}", "reused": False}
    except EndpointConnectionError as e:
        return {"status": "failed", "url": None, "key": key, "sha256": sha, "error": f"Could not reach S3: {e}", "reused": False}
    except ClientError as e:
        msg = e.response.get("Error", {}).get("Message", str(e))
        return {"status": "failed", "url": None, "key": key, "sha256": sha, "error": f"S3 upload failed: {msg}", "reused": False}
    except Exception as e:
        return {"status": "failed", "url": None, "key": key, "sha256": sha, "error": str(e), "reused": False}
