"""S3 small-object helpers used by the default publisher."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any, TypedDict
from urllib.parse import urlparse

from .credentials import S3Credentials
from .errors import PublishManifestError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .metrics import MetricEvent, MetricsCollector
from .type_defs import RetryConfig, S3ConnectionOptions

_logger = get_logger(__name__)

# Retry defaults for transient S3 errors.
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_INITIAL_BACKOFF_S = 1.0
_DEFAULT_BACKOFF_MULTIPLIER = 2.0

# S3 error codes considered transient and safe to retry.
_TRANSIENT_S3_CODES: frozenset[str] = frozenset(
    {
        "RequestTimeout",
        "RequestTimeoutException",
        "InternalError",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
        "RequestLimitExceeded",
        "BandwidthLimitExceeded",
        "503",
        "500",
        "429",
    }
)


def _is_transient_s3_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a transient/retryable S3 error."""

    # Check boto3/botocore ClientError response codes.
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in _TRANSIENT_S3_CODES:
            return True
        http_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if http_code in {429, 500, 502, 503}:
            return True

    # Check common transient network/connection error types by name.
    exc_type_name = type(exc).__name__
    if exc_type_name in {
        "ConnectionError",
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "ConnectionClosedError",
    }:
        return True

    return False


def _resolve_retry_params(
    retry_config: RetryConfig | None,
) -> tuple[int, float, float]:
    """Extract (max_retries, initial_backoff, backoff_multiplier) from config or defaults."""
    if retry_config is not None:
        return (
            retry_config.max_retries,
            retry_config.initial_backoff_s,
            retry_config.backoff_multiplier,
        )
    return _DEFAULT_MAX_RETRIES, _DEFAULT_INITIAL_BACKOFF_S, _DEFAULT_BACKOFF_MULTIPLIER


def _retry_s3_operation(
    operation: Callable[[], Any],
    *,
    operation_name: str,
    url: str,
    metrics_collector: MetricsCollector | None = None,
    retry_config: RetryConfig | None = None,
):
    """Execute *operation* with exponential-backoff retries on transient S3 errors.

    Returns the result of *operation()* on success, or re-raises the last
    exception after exhausting all retries.
    """

    max_retries, initial_backoff, backoff_multiplier = _resolve_retry_params(
        retry_config
    )

    last_exc: BaseException | None = None
    delay = initial_backoff

    for attempt in range(max_retries + 1):
        try:
            result = operation()
            if attempt > 0:
                log_event(
                    "s3_retry_succeeded",
                    logger=_logger,
                    operation=operation_name,
                    url=url,
                    attempts=attempt + 1,
                )
            return result
        except Exception as exc:
            last_exc = exc
            if not _is_transient_s3_error(exc) or attempt == max_retries:
                if attempt > 0:
                    log_failure(
                        "s3_operation_failed_after_retries",
                        severity=FailureSeverity.ERROR,
                        logger=_logger,
                        error=exc,
                        operation=operation_name,
                        url=url,
                        attempts=attempt + 1,
                    )
                    if metrics_collector is not None:
                        metrics_collector.emit(
                            MetricEvent.S3_RETRY_EXHAUSTED,
                            {
                                "attempts": attempt + 1,
                            },
                        )
                raise

            log_failure(
                "s3_transient_failure",
                severity=FailureSeverity.TRANSIENT,
                logger=_logger,
                error=exc,
                operation=operation_name,
                url=url,
                attempt=attempt + 1,
                max_retries=max_retries,
                retry_delay_s=delay,
            )
            if metrics_collector is not None:
                metrics_collector.emit(
                    MetricEvent.S3_RETRY,
                    {
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "delay_s": delay,
                    },
                )
            time.sleep(delay)
            delay *= backoff_multiplier

    # Unreachable, but makes the type checker happy.
    raise last_exc  # type: ignore[misc]


def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""

    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URL: {url}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URL must include key path: {url}")
    return parsed.netloc, key


def join_s3(base: str, *parts: str) -> str:
    """Join S3 URL path segments, stripping redundant slashes."""
    clean = [base.rstrip("/")]
    clean.extend(part.strip("/") for part in parts if part)
    return "/".join(clean)


class _ResolvedS3Config(TypedDict, total=False):
    """Resolved S3 client kwargs shared by boto3 and aiobotocore."""

    endpoint_url: str
    region_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    config: object  # botocore.config.Config
    verify: bool | str


def _resolve_s3_config(
    credentials: S3Credentials | None = None,
    connection_options: S3ConnectionOptions | None = None,
) -> _ResolvedS3Config:
    """Resolve S3 client kwargs from explicit credentials/options and env vars.

    When *credentials* is provided, its fields are used as-is (no env-var
    fallback for identity — the provider is the single source of truth).
    When *credentials* is ``None``, identity falls back to env vars.
    Transport always falls back to env vars for endpoint/region.

    Returns a dict suitable for passing to ``boto3.client("s3", ...)`` or
    ``aiobotocore.get_session().create_client("s3", ...)``.
    """
    from botocore.config import Config as BotocoreConfig

    opts = connection_options or {}

    # Identity: if a provider resolved credentials, use them as-is.
    # Only fall back to env vars when no credentials were given at all.
    if credentials is not None:
        access_key_id = credentials.access_key_id
        secret_access_key = credentials.secret_access_key
        session_token = credentials.session_token
    else:
        access_key_id = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        session_token = os.getenv("S3_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")

    # Transport: explicit options → env-var fallback
    endpoint_url = opts.get("endpoint_url") or os.getenv("SLATEDB_S3_ENDPOINT_URL")
    region_name = (
        opts.get("region_name") or os.getenv("S3_REGION") or os.getenv("AWS_REGION")
    )

    # Assemble botocore Config from optional connection options
    boto_config_kwargs: dict[str, object] = {}
    s3_options: dict[str, object] = {}

    addressing_style = opts.get("addressing_style")
    if addressing_style:
        s3_options["addressing_style"] = addressing_style

    signature_version = opts.get("signature_version")
    if signature_version:
        boto_config_kwargs["signature_version"] = signature_version

    connect_timeout = opts.get("connect_timeout")
    if connect_timeout is not None:
        boto_config_kwargs["connect_timeout"] = connect_timeout

    read_timeout = opts.get("read_timeout")
    if read_timeout is not None:
        boto_config_kwargs["read_timeout"] = read_timeout

    max_attempts = opts.get("max_attempts")
    if max_attempts is not None:
        boto_config_kwargs["retries"] = {
            "max_attempts": max_attempts,
            "mode": "standard",
        }

    if s3_options:
        boto_config_kwargs["s3"] = s3_options

    botocore_config = (
        BotocoreConfig(**boto_config_kwargs) if boto_config_kwargs else None
    )

    result: _ResolvedS3Config = {}
    if endpoint_url:
        result["endpoint_url"] = endpoint_url
    if region_name:
        result["region_name"] = region_name
    if access_key_id:
        result["aws_access_key_id"] = access_key_id
    if secret_access_key:
        result["aws_secret_access_key"] = secret_access_key
    if session_token:
        result["aws_session_token"] = session_token
    if botocore_config is not None:
        result["config"] = botocore_config

    verify = opts.get("verify_ssl")
    if verify is not None and verify is not True:
        result["verify"] = verify

    return result


def create_s3_client(
    credentials: S3Credentials | None = None,
    connection_options: S3ConnectionOptions | None = None,
):
    """Create boto3 S3 client with optional explicit credentials and connection options."""

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise PublishManifestError(
            "boto3 is required for default S3 publishing"
        ) from exc

    resolved = _resolve_s3_config(credentials, connection_options)
    return boto3.client("s3", **resolved)


def delete_prefix(
    prefix_url: str,
    *,
    s3_client: Any = None,
    metrics_collector: MetricsCollector | None = None,
    retry_config: RetryConfig | None = None,
) -> int:
    """Delete all objects under an S3 prefix. Returns the number of objects deleted.

    When *retry_config* is ``None`` (the default), failures are logged but
    not raised (best-effort).  When a *retry_config* is provided, each
    list+delete page is wrapped in ``_retry_s3_operation()`` and transient
    errors are retried; non-transient errors propagate.
    """
    client = s3_client or create_s3_client()
    bucket, key_prefix = parse_s3_url(prefix_url)

    deleted = 0
    continuation_token: str | None = None

    try:
        while True:
            # Step 1: List one page of objects (retriable independently)
            list_kwargs: dict[str, Any] = {
                "Bucket": bucket,
                "Prefix": key_prefix,
                "MaxKeys": 1000,
            }
            if continuation_token is not None:
                list_kwargs["ContinuationToken"] = continuation_token

            if retry_config is not None:
                response = _retry_s3_operation(
                    lambda kw=list_kwargs: client.list_objects_v2(**kw),
                    operation_name="delete_prefix_list",
                    url=prefix_url,
                    metrics_collector=metrics_collector,
                    retry_config=retry_config,
                )
            else:
                response = client.list_objects_v2(**list_kwargs)

            contents = response.get("Contents", [])
            if not contents:
                break

            # Step 2: Delete the listed objects (retriable independently)
            objects = [{"Key": obj["Key"]} for obj in contents]
            delete_payload: dict[str, Any] = {
                "Bucket": bucket,
                "Delete": {"Objects": objects, "Quiet": True},
            }
            if retry_config is not None:
                _retry_s3_operation(
                    lambda kw=delete_payload: client.delete_objects(**kw),
                    operation_name="delete_prefix_delete",
                    url=prefix_url,
                    metrics_collector=metrics_collector,
                    retry_config=retry_config,
                )
            else:
                client.delete_objects(**delete_payload)

            deleted += len(objects)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
    except Exception as exc:
        if retry_config is not None:
            raise
        log_failure(
            "s3_delete_prefix_failed",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            error=exc,
            prefix_url=prefix_url,
            deleted_so_far=deleted,
        )

    return deleted


def list_prefixes(
    prefix_url: str,
    *,
    s3_client: Any = None,
    retry_config: RetryConfig | None = None,
) -> list[str]:
    """List immediate child prefixes under an S3 prefix (CommonPrefixes).

    Returns full S3 URLs for each child prefix, sorted lexicographically.
    Uses manual pagination with per-page retry for resilience against
    transient S3 errors mid-pagination.
    """
    client = s3_client or create_s3_client()
    bucket, key_prefix = parse_s3_url(prefix_url)
    if not key_prefix.endswith("/"):
        key_prefix += "/"

    prefixes: list[str] = []
    continuation_token: str | None = None

    while True:
        list_kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": key_prefix,
            "Delimiter": "/",
        }
        if continuation_token is not None:
            list_kwargs["ContinuationToken"] = continuation_token

        if retry_config is not None:
            response = _retry_s3_operation(
                lambda kw=list_kwargs: client.list_objects_v2(**kw),
                operation_name="list_prefixes",
                url=prefix_url,
                retry_config=retry_config,
            )
        else:
            response = client.list_objects_v2(**list_kwargs)

        for cp in response.get("CommonPrefixes", []):
            prefixes.append(f"s3://{bucket}/{cp['Prefix']}")

        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")

    prefixes.sort()
    return prefixes


def put_bytes(
    url: str,
    payload: bytes,
    content_type: str,
    headers: Mapping[str, str] | None = None,
    *,
    s3_client: Any = None,
    metrics_collector: MetricsCollector | None = None,
    retry_config: RetryConfig | None = None,
) -> None:
    """PUT bytes to S3 URL with automatic retry on transient S3 errors."""

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

    _retry_s3_operation(
        lambda: client.put_object(**put_kwargs),
        operation_name="put_object",
        url=url,
        metrics_collector=metrics_collector,
        retry_config=retry_config,
    )


def get_bytes(
    url: str,
    *,
    s3_client: Any = None,
    metrics_collector: MetricsCollector | None = None,
    retry_config: RetryConfig | None = None,
) -> bytes:
    """Read object bytes from S3 URL with automatic retry on transient errors."""

    client = s3_client or create_s3_client()
    bucket, key = parse_s3_url(url)

    def _do_get():
        obj = client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()

    return _retry_s3_operation(
        _do_get,
        operation_name="get_object",
        url=url,
        metrics_collector=metrics_collector,
        retry_config=retry_config,
    )


def try_get_bytes(
    url: str,
    *,
    s3_client: Any = None,
    metrics_collector: MetricsCollector | None = None,
    retry_config: RetryConfig | None = None,
) -> bytes | None:
    """Read object bytes and return None when object is not found.

    Transient S3 errors (throttling, timeouts) are retried automatically.
    """

    client = s3_client or create_s3_client()
    bucket, key = parse_s3_url(url)

    def _do_get():
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:  # broad catch intentional: supports duck-typed S3-compatible clients whose error types vary
            code = None
            response = getattr(exc, "response", None)
            if isinstance(response, dict):
                code = response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        return obj["Body"].read()

    return _retry_s3_operation(
        _do_get,
        operation_name="try_get_object",
        url=url,
        metrics_collector=metrics_collector,
        retry_config=retry_config,
    )
