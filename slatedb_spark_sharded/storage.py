"""S3 small-object helpers used by the default publisher."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse
import os

from .errors import PublishManifestError


def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""

    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URL: {url}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URL must include key path: {url}")
    return parsed.netloc, key


def create_s3_client(s3_client_config: Mapping[str, Any] | None = None):
    """Create boto3 S3 client with optional explicit/ENV overrides."""

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise PublishManifestError(
            "boto3 is required for default S3 publishing"
        ) from exc

    config = dict(s3_client_config or {})
    endpoint_url = config.get("endpoint_url") or os.getenv("SLATEDB_S3_ENDPOINT_URL")
    region_name = config.get("region_name") or os.getenv("AWS_REGION")
    aws_access_key_id = config.get("aws_access_key_id") or os.getenv(
        "AWS_ACCESS_KEY_ID"
    )
    aws_secret_access_key = config.get("aws_secret_access_key") or os.getenv(
        "AWS_SECRET_ACCESS_KEY"
    )
    aws_session_token = config.get("aws_session_token") or os.getenv(
        "AWS_SESSION_TOKEN"
    )

    kwargs: dict[str, Any] = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if region_name:
        kwargs["region_name"] = region_name
    if aws_access_key_id:
        kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        kwargs["aws_session_token"] = aws_session_token

    return boto3.client("s3", **kwargs)


def put_bytes(
    url: str,
    payload: bytes,
    content_type: str,
    headers: Mapping[str, str] | None = None,
    *,
    s3_client=None,
) -> None:
    """PUT bytes to S3 URL."""

    client = s3_client or create_s3_client()
    bucket, key = parse_s3_url(url)

    put_kwargs: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": payload,
        "ContentType": content_type,
    }

    if headers:
        # Map optional artifact headers into S3 user metadata to preserve context.
        put_kwargs["Metadata"] = {str(k): str(v) for k, v in headers.items()}

    client.put_object(**put_kwargs)


def get_bytes(url: str, *, s3_client=None) -> bytes:
    """Read object bytes from S3 URL."""

    client = s3_client or create_s3_client()
    bucket, key = parse_s3_url(url)
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def try_get_bytes(url: str, *, s3_client=None) -> bytes | None:
    """Read object bytes and return None when object is not found."""

    client = s3_client or create_s3_client()
    bucket, key = parse_s3_url(url)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # pragma: no cover - runtime/SDK dependent
        code = None
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    return obj["Body"].read()
