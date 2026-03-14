"""Async manifest store protocol and implementations.

``AsyncManifestStore`` is the read-only async counterpart of
``ManifestStore``.  ``AsyncS3ManifestStore`` uses *aiobotocore* for
native async S3 I/O.
"""

from __future__ import annotations

from typing import Any, Protocol

from .logging import FailureSeverity, get_logger, log_failure
from .manifest import ManifestRef, ParsedManifest
from .manifest_store import (
    ManifestStore,
    parse_json_manifest,
)
from .metrics import MetricsCollector
from .storage import join_s3, parse_s3_url
from .type_defs import S3ClientConfig

_logger = get_logger(__name__)


class AsyncManifestStore(Protocol):
    """Read-only async manifest loading (no publish — that's writer-side)."""

    async def load_current(self) -> ManifestRef | None: ...

    async def load_manifest(self, ref: str) -> ParsedManifest: ...

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]: ...


class AsyncS3ManifestStore:
    """Native async S3 manifest store using aiobotocore.

    Each ``load_current()`` / ``load_manifest()`` call creates a short-lived
    S3 client via the session context manager.  These calls are infrequent
    (init + refresh only), so per-call clients are simple and safe.
    """

    def __init__(
        self,
        s3_prefix: str,
        *,
        manifest_name: str = "manifest",
        current_name: str = "_CURRENT",
        s3_client_config: S3ClientConfig | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        import aiobotocore.session  # type: ignore[import-not-found]

        self.s3_prefix = s3_prefix.rstrip("/")
        self.manifest_name = manifest_name
        self.current_name = current_name
        self._session = aiobotocore.session.get_session()
        self._resolved = _resolve_s3_config_for_aiobotocore(s3_client_config)
        self._metrics = metrics_collector

    async def load_current(self) -> ManifestRef | None:
        from .manifest_store import parse_current_pointer_to_ref

        current_url = f"{self.s3_prefix}/{self.current_name}"
        payload = await self._try_get_bytes(current_url)
        if payload is None:
            return None
        return parse_current_pointer_to_ref(payload)

    async def load_manifest(self, ref: str) -> ParsedManifest:
        try:
            payload = await self._get_bytes(ref)
        except Exception as exc:
            log_failure(
                "manifest_s3_load_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                manifest_ref=ref,
            )
            raise
        return parse_json_manifest(payload)

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        from .manifest_store import parse_manifest_dir_entry

        manifests_prefix = join_s3(self.s3_prefix, "manifests") + "/"
        bucket, key_prefix = parse_s3_url(manifests_prefix)

        refs: list[ManifestRef] = []
        async with self._session.create_client("s3", **self._resolved) as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(  # type: ignore[union-attr]
                Bucket=bucket, Prefix=key_prefix, Delimiter="/"
            ):
                for entry in page.get("CommonPrefixes", []):
                    dir_name = entry["Prefix"][len(key_prefix) :]
                    ref = parse_manifest_dir_entry(
                        dir_name, self.s3_prefix, self.manifest_name
                    )
                    if ref is not None:
                        refs.append(ref)

        refs.sort(key=lambda r: r.published_at, reverse=True)
        return refs[:limit]

    async def _get_bytes(self, url: str) -> bytes:
        return await _async_retry_s3_operation(
            self._do_get_bytes,
            url,
            operation_name="get_object",
            url_for_log=url,
            metrics_collector=self._metrics,
        )

    async def _try_get_bytes(self, url: str) -> bytes | None:
        return await _async_retry_s3_operation(
            self._do_try_get_bytes,
            url,
            operation_name="try_get_object",
            url_for_log=url,
            metrics_collector=self._metrics,
        )

    async def _do_get_bytes(self, url: str) -> bytes:
        bucket, key = parse_s3_url(url)
        async with self._session.create_client("s3", **self._resolved) as client:
            obj = await client.get_object(Bucket=bucket, Key=key)  # type: ignore[misc]
            async with obj["Body"] as stream:
                return await stream.read()

    async def _do_try_get_bytes(self, url: str) -> bytes | None:
        bucket, key = parse_s3_url(url)
        async with self._session.create_client("s3", **self._resolved) as client:
            try:
                obj = await client.get_object(Bucket=bucket, Key=key)  # type: ignore[misc]
            except Exception as exc:
                code = None
                response = getattr(exc, "response", None)
                if isinstance(response, dict):
                    code = response.get("Error", {}).get("Code")
                if code in {"NoSuchKey", "404", "NotFound"}:
                    return None
                raise
            async with obj["Body"] as stream:
                return await stream.read()


class _SyncManifestStoreAdapter:
    """Wraps a sync ``ManifestStore`` for use with ``AsyncShardedReader``.

    S3 calls are offloaded to a thread via ``asyncio.to_thread()``.
    """

    def __init__(self, store: ManifestStore) -> None:
        self._store = store

    async def load_current(self) -> ManifestRef | None:
        import asyncio

        return await asyncio.to_thread(self._store.load_current)

    async def load_manifest(self, ref: str) -> ParsedManifest:
        import asyncio

        return await asyncio.to_thread(self._store.load_manifest, ref)

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        import asyncio

        return await asyncio.to_thread(self._store.list_manifests, limit=limit)


def _resolve_s3_config_for_aiobotocore(
    s3_client_config: S3ClientConfig | None = None,
) -> dict[str, Any]:
    """Resolve S3 config for aiobotocore's ``create_client()``."""
    from .storage import _resolve_s3_config

    return dict(_resolve_s3_config(s3_client_config))


async def _async_retry_s3_operation(
    operation: Any,
    *args: Any,
    operation_name: str,
    url_for_log: str,
    metrics_collector: MetricsCollector | None = None,
) -> Any:
    """Async equivalent of ``_retry_s3_operation`` with ``asyncio.sleep``."""
    import asyncio

    from .storage import (
        _DEFAULT_BACKOFF_MULTIPLIER,
        _DEFAULT_INITIAL_BACKOFF_S,
        _DEFAULT_MAX_RETRIES,
        _is_transient_s3_error,
    )

    last_exc: BaseException | None = None
    delay = _DEFAULT_INITIAL_BACKOFF_S

    for attempt in range(_DEFAULT_MAX_RETRIES + 1):
        try:
            result = await operation(*args)
            if attempt > 0:
                from .logging import log_event

                log_event(
                    "s3_retry_succeeded",
                    logger=_logger,
                    operation=operation_name,
                    url=url_for_log,
                    attempts=attempt + 1,
                )
            return result
        except Exception as exc:
            last_exc = exc
            if not _is_transient_s3_error(exc) or attempt == _DEFAULT_MAX_RETRIES:
                if attempt > 0:
                    log_failure(
                        "s3_operation_failed_after_retries",
                        severity=FailureSeverity.ERROR,
                        logger=_logger,
                        error=exc,
                        operation=operation_name,
                        url=url_for_log,
                        attempts=attempt + 1,
                    )
                    if metrics_collector is not None:
                        from .metrics import MetricEvent

                        metrics_collector.emit(
                            MetricEvent.S3_RETRY_EXHAUSTED,
                            {"attempts": attempt + 1},
                        )
                raise

            log_failure(
                "s3_transient_failure",
                severity=FailureSeverity.TRANSIENT,
                logger=_logger,
                error=exc,
                operation=operation_name,
                url=url_for_log,
                attempt=attempt + 1,
                max_retries=_DEFAULT_MAX_RETRIES,
                retry_delay_s=delay,
            )
            if metrics_collector is not None:
                from .metrics import MetricEvent

                metrics_collector.emit(
                    MetricEvent.S3_RETRY,
                    {
                        "attempt": attempt + 1,
                        "max_retries": _DEFAULT_MAX_RETRIES,
                        "delay_s": delay,
                    },
                )
            await asyncio.sleep(delay)
            delay *= _DEFAULT_BACKOFF_MULTIPLIER

    raise last_exc  # type: ignore[misc]
