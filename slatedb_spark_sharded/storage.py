"""S3 small-object helpers used by the default publisher."""

from __future__ import annotations

import os
from typing import Mapping, TypedDict
from urllib.parse import urlparse

from .errors import PublishManifestError
from .type_defs import S3ClientConfig


def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""

    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URL: {url}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URL must include key path: {url}")
    return parsed.netloc, key


class _S3ClientKwargs(TypedDict, total=False):
    endpoint_url: str
    region_name: str
    access_key_id: str
    secret_access_key: str
    session_token: str


def create_s3_client(s3_client_config: S3ClientConfig | None = None):
    """Create boto3 S3 client with optional explicit/ENV overrides."""

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise PublishManifestError(
            "boto3 is required for default S3 publishing"
        ) from exc

    config = s3_client_config or {}
    endpoint_url = config.get("endpoint_url") or os.getenv("SLATEDB_S3_ENDPOINT_URL")
    region_name = config.get("region_name") or os.getenv("S3_REGION") or os.getenv(
        "AWS_REGION"
    )
    access_key_id = config.get("access_key_id") or os.getenv("S3_ACCESS_KEY_ID") or os.getenv(
        "AWS_ACCESS_KEY_ID"
    )
    secret_access_key = (
        config.get("secret_access_key")
        or os.getenv("S3_SECRET_ACCESS_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    session_token = config.get("session_token") or os.getenv("S3_SESSION_TOKEN") or os.getenv(
        "AWS_SESSION_TOKEN"
    )

    kwargs: _S3ClientKwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if region_name:
        kwargs["region_name"] = region_name
    if access_key_id:
        kwargs["access_key_id"] = access_key_id
    if secret_access_key:
        kwargs["secret_access_key"] = secret_access_key
    if session_token:
        kwargs["session_token"] = session_token

    return boto3.client(
        "s3",
        endpoint_url=kwargs.get("endpoint_url"),
        region_name=kwargs.get("region_name"),
        aws_access_key_id=kwargs.get("access_key_id"),
        aws_secret_access_key=kwargs.get("secret_access_key"),
        aws_session_token=kwargs.get("session_token"),
    )


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

    put_kwargs: dict[str, object] = {
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
