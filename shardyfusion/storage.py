"""S3 small-object helpers and storage backend abstraction.

Provides :class:`StorageBackend` and :class:`AsyncStorageBackend` protocols,
plus :class:`ObstoreBackend` (obstore-backed) and :class:`MemoryBackend`
(in-memory test double) implementations.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Protocol
from urllib.parse import urlparse

from .credentials import S3Credentials
from .errors import PublishManifestError
from .logging import get_logger, log_event
from .type_defs import S3ConnectionOptions

_logger = get_logger(__name__)

# Shardyfusion-tuned obstore retry config.
# Aggressive: fast first retry (~200ms) with jitter, 4s cap, 5 retries, 30s timeout.
_DEFAULT_RETRY_CONFIG: dict[str, Any] = {
    "max_retries": 5,
    "backoff": {
        "init_backoff": timedelta(milliseconds=200),
        "max_backoff": timedelta(seconds=4),
        "base": 2.0,
    },
    "retry_timeout": timedelta(seconds=30),
}


class StorageBackend(Protocol):
    """Sync object storage abstraction. Operates on ``s3://`` URLs."""

    def put(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None: ...

    def get(self, url: str) -> bytes: ...

    def try_get(self, url: str) -> bytes | None: ...

    def list_prefixes(self, prefix_url: str) -> list[str]: ...

    def delete_prefix(self, prefix_url: str) -> int: ...


class AsyncStorageBackend(Protocol):
    """Async counterpart to :class:`StorageBackend`."""

    async def put(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None: ...

    async def get(self, url: str) -> bytes: ...

    async def try_get(self, url: str) -> bytes | None: ...

    async def list_prefixes(self, prefix_url: str) -> list[str]: ...

    async def delete_prefix(self, prefix_url: str) -> int: ...


class ObstoreBackend:
    """Sync obstore-backed :class:`StorageBackend` implementation."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def put(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        bucket, key = parse_s3_url(url)
        import obstore  # pyright: ignore[reportMissingImports]

        attributes: dict[str, str] = {}
        if content_type:
            attributes["Content-Type"] = content_type
        if headers:
            attributes.update(headers)

        # obstore retries 5xx/429 internally.  We add one fallback retry for
        # GenericError wrapping a PUT timeout that Rust declined to retry
        # (non-idempotent methods are not retried on timeout).
        try:
            obstore.put(self._store, key, payload, attributes=attributes or None)
        except Exception as exc:
            if _is_put_retryable(exc):
                log_event(
                    "s3_put_fallback_retry",
                    logger=_logger,
                    url=url,
                    delay_s=2.0,
                )
                time.sleep(2.0)
                obstore.put(self._store, key, payload, attributes=attributes or None)
            else:
                raise

    def get(self, url: str) -> bytes:
        bucket, key = parse_s3_url(url)
        import obstore  # pyright: ignore[reportMissingImports]

        result = obstore.get(self._store, key)
        return bytes(result.bytes())

    def try_get(self, url: str) -> bytes | None:
        bucket, key = parse_s3_url(url)
        import obstore  # pyright: ignore[reportMissingImports]
        from obstore.exceptions import (
            NotFoundError,  # pyright: ignore[reportMissingImports]
        )

        try:
            result = obstore.get(self._store, key)
            return bytes(result.bytes())
        except NotFoundError:
            return None

    def list_prefixes(self, prefix_url: str) -> list[str]:
        bucket, key_prefix = parse_s3_url(prefix_url)
        if not key_prefix.endswith("/"):
            key_prefix += "/"
        import obstore  # pyright: ignore[reportMissingImports]

        list_result = obstore.list_with_delimiter(self._store, prefix=key_prefix)
        prefixes = [
            f"s3://{bucket}/{cp}" for cp in list_result.get("common_prefixes", [])
        ]
        prefixes.sort()
        return prefixes

    def delete_prefix(self, prefix_url: str) -> int:
        bucket, key_prefix = parse_s3_url(prefix_url)
        import obstore  # pyright: ignore[reportMissingImports]

        # obstore.list is an auto-paginating iterator of chunks.
        paths: list[str] = []
        for chunk in obstore.list(self._store, prefix=key_prefix):
            for meta in chunk:
                paths.append(meta["path"])
        if not paths:
            return 0
        obstore.delete(self._store, paths)
        return len(paths)


class AsyncObstoreBackend:
    """Async obstore-backed :class:`AsyncStorageBackend` implementation."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def put(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        bucket, key = parse_s3_url(url)
        import obstore  # pyright: ignore[reportMissingImports]

        attributes: dict[str, str] = {}
        if content_type:
            attributes["Content-Type"] = content_type
        if headers:
            attributes.update(headers)

        try:
            await obstore.put_async(
                self._store, key, payload, attributes=attributes or None
            )
        except Exception as exc:
            if _is_put_retryable(exc):
                import asyncio

                log_event(
                    "s3_put_fallback_retry",
                    logger=_logger,
                    url=url,
                    delay_s=2.0,
                )
                await asyncio.sleep(2.0)
                await obstore.put_async(
                    self._store, key, payload, attributes=attributes or None
                )
            else:
                raise

    async def get(self, url: str) -> bytes:
        bucket, key = parse_s3_url(url)
        import obstore  # pyright: ignore[reportMissingImports]

        result = await obstore.get_async(self._store, key)
        return bytes(result.bytes())

    async def try_get(self, url: str) -> bytes | None:
        bucket, key = parse_s3_url(url)
        import obstore  # pyright: ignore[reportMissingImports]
        from obstore.exceptions import (
            NotFoundError,  # pyright: ignore[reportMissingImports]
        )

        try:
            result = await obstore.get_async(self._store, key)
            return bytes(result.bytes())
        except NotFoundError:
            return None

    async def list_prefixes(self, prefix_url: str) -> list[str]:
        bucket, key_prefix = parse_s3_url(prefix_url)
        if not key_prefix.endswith("/"):
            key_prefix += "/"
        import obstore  # pyright: ignore[reportMissingImports]

        list_result = await obstore.list_with_delimiter_async(
            self._store, prefix=key_prefix
        )
        prefixes = [
            f"s3://{bucket}/{cp}" for cp in list_result.get("common_prefixes", [])
        ]
        prefixes.sort()
        return prefixes

    async def delete_prefix(self, prefix_url: str) -> int:
        bucket, key_prefix = parse_s3_url(prefix_url)
        import obstore  # pyright: ignore[reportMissingImports]

        paths: list[str] = []
        async for chunk in obstore.list(self._store, prefix=key_prefix):
            for meta in chunk:
                paths.append(meta["path"])
        if not paths:
            return 0
        await obstore.delete_async(self._store, paths)
        return len(paths)


class _MemoryStore:
    """Shared in-memory store backing both sync and async test doubles."""

    def __init__(self) -> None:
        self._data: dict[
            str, tuple[bytes, str, Mapping[str, str] | None]
        ] = {}  # url → (payload, content_type, headers)

    def _put(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._data[url] = (payload, content_type, headers)

    def _get(self, url: str) -> bytes:
        if url not in self._data:
            raise FileNotFoundError(url)
        return self._data[url][0]

    def _try_get(self, url: str) -> bytes | None:
        return self._data.get(url, (None, "", None))[0]

    def _list_prefixes(self, prefix_url: str) -> list[str]:
        if not prefix_url.endswith("/"):
            prefix_url += "/"
        prefixes: set[str] = set()
        for url in self._data:
            if url.startswith(prefix_url):
                rest = url[len(prefix_url) :]
                if "/" in rest:
                    dir_name = prefix_url + rest.split("/")[0] + "/"
                    prefixes.add(dir_name)
        return sorted(prefixes)

    def _delete_prefix(self, prefix_url: str) -> int:
        to_delete = [url for url in self._data if url.startswith(prefix_url)]
        for url in to_delete:
            del self._data[url]
        return len(to_delete)


class MemoryBackend:
    """In-memory :class:`StorageBackend` test double."""

    def __init__(self) -> None:
        self._store = _MemoryStore()

    def put(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._store._put(url, payload, content_type, headers)

    def get(self, url: str) -> bytes:
        return self._store._get(url)

    def try_get(self, url: str) -> bytes | None:
        return self._store._try_get(url)

    def list_prefixes(self, prefix_url: str) -> list[str]:
        return self._store._list_prefixes(prefix_url)

    def delete_prefix(self, prefix_url: str) -> int:
        return self._store._delete_prefix(prefix_url)


class AsyncMemoryBackend:
    """In-memory :class:`AsyncStorageBackend` test double."""

    def __init__(self) -> None:
        self._store = _MemoryStore()

    async def put(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._store._put(url, payload, content_type, headers)

    async def get(self, url: str) -> bytes:
        return self._store._get(url)

    async def try_get(self, url: str) -> bytes | None:
        return self._store._try_get(url)

    async def list_prefixes(self, prefix_url: str) -> list[str]:
        return self._store._list_prefixes(prefix_url)

    async def delete_prefix(self, prefix_url: str) -> int:
        return self._store._delete_prefix(prefix_url)


def _is_put_retryable(exc: BaseException) -> bool:
    """Return True when *exc* looks like a transient error worth one PUT retry."""
    import obstore.exceptions  # pyright: ignore[reportMissingImports]

    if isinstance(exc, obstore.exceptions.GenericError):
        return True
    exc_name = type(exc).__name__
    return exc_name in {
        "ConnectionError",
        "TimeoutError",
        "ReadTimeoutError",
    }


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


def create_s3_store(
    bucket: str,
    credentials: S3Credentials | None = None,
    connection_options: S3ConnectionOptions | None = None,
) -> Any:
    """Create an :class:`obstore.store.S3Store` with shardyfusion config.

    Credentials, endpoint, region, and timeout options are mapped from
    shardyfusion's typed config to obstore's constructor.  The retry
    configuration uses shardyfusion's aggressive defaults (5 retries,
    200ms initial backoff with jitter, 4s cap, 30s timeout).
    """
    try:
        from obstore.store import S3Store  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise PublishManifestError(
            "obstore is required for S3 I/O. "
            "Install via: pip install 'shardyfusion[slatedb]' or any backend extra."
        ) from exc

    opts: S3ConnectionOptions = connection_options or {}
    kwargs: dict[str, Any] = {}
    client_options: dict[str, Any] = {}

    # Identity
    if credentials is not None:
        if credentials.access_key_id is not None:
            kwargs["access_key_id"] = credentials.access_key_id
        if credentials.secret_access_key is not None:
            kwargs["secret_access_key"] = credentials.secret_access_key
        if credentials.session_token is not None:
            kwargs["session_token"] = credentials.session_token
    else:
        access_key = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        session_token = os.getenv("S3_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")
        if access_key:
            kwargs["access_key_id"] = access_key
        if secret_key:
            kwargs["secret_access_key"] = secret_key
        if session_token:
            kwargs["session_token"] = session_token

    # Transport
    endpoint = opts.get("endpoint_url") or os.getenv("SLATEDB_S3_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint"] = endpoint
        if endpoint.startswith("http://"):
            client_options["allow_http"] = True

    region = (
        opts.get("region_name") or os.getenv("S3_REGION") or os.getenv("AWS_REGION")
    )
    if region:
        kwargs["region"] = region

    addressing = opts.get("addressing_style")
    if addressing == "path":
        kwargs["virtual_hosted_style_request"] = False
    elif addressing == "virtual":
        kwargs["virtual_hosted_style_request"] = True

    verify = opts.get("verify_ssl")
    if verify is False:
        client_options["allow_invalid_certificates"] = True
    elif isinstance(verify, str):
        # CA bundle path not directly supported; leave to SSL_CERT_FILE env var.
        pass

    connect_timeout = opts.get("connect_timeout")
    if connect_timeout is not None:
        client_options["connect_timeout"] = f"{int(connect_timeout)}s"

    read_timeout = opts.get("read_timeout")
    if read_timeout is not None:
        client_options["timeout"] = f"{int(read_timeout)}s"

    if client_options:
        kwargs["client_options"] = client_options

    # Retry — map max_attempts if explicitly set, otherwise use shardyfusion defaults
    max_attempts = opts.get("max_attempts")
    if max_attempts is not None:
        kwargs["retry_config"] = {"max_retries": max(0, int(max_attempts) - 1)}
    else:
        kwargs["retry_config"] = _DEFAULT_RETRY_CONFIG

    return S3Store(bucket, **kwargs)
