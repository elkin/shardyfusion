"""Unified manifest persistence protocol and implementations.

ManifestStore replaces the previous ManifestPublisher + ManifestReader
pair with a single interface that accepts Pydantic models directly.
S3ManifestStore implements two-phase publish internally; database
backends can use a single atomic transaction.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import ValidationError

from .errors import ManifestParseError
from .logging import FailureSeverity, get_logger, log_failure
from .manifest import (
    CurrentPointer,
    JsonManifestBuilder,
    ManifestArtifact,
    ManifestBuilder,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .metrics import MetricsCollector
from .storage import create_s3_client, get_bytes, join_s3, put_bytes, try_get_bytes
from .type_defs import S3ClientConfig

_logger = get_logger(__name__)


class ManifestStore(Protocol):
    """Unified protocol for persisting and loading manifests."""

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        """Persist manifest + CURRENT atomically. Return a manifest reference."""
        ...

    def load_current(self) -> CurrentPointer | None:
        """Return the latest CURRENT pointer, or None if not published."""
        ...

    def load_manifest(self, ref: str) -> ParsedManifest:
        """Load and parse a manifest by reference."""
        ...


class S3ManifestStore:
    """Manifest store backed by S3 — merges publish and read in one class.

    Publish is two-phase: manifest file first, then ``_CURRENT`` pointer.
    """

    def __init__(
        self,
        s3_prefix: str,
        *,
        manifest_name: str = "manifest",
        current_name: str = "_CURRENT",
        manifest_builder: ManifestBuilder | None = None,
        s3_client_config: S3ClientConfig | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self.s3_prefix = s3_prefix.rstrip("/")
        self.manifest_name = manifest_name
        self.current_name = current_name
        self._builder = manifest_builder
        self._s3_client = create_s3_client(s3_client_config)
        self._metrics = metrics_collector

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        builder = self._builder or JsonManifestBuilder()
        artifact = builder.build(
            required_build=required_build,
            shards=shards,
            custom_fields=custom,
        )

        manifest_url = join_s3(
            self.s3_prefix,
            "manifests",
            f"run_id={run_id}",
            self.manifest_name,
        )
        put_bytes(
            manifest_url,
            artifact.payload,
            artifact.content_type,
            artifact.headers,
            s3_client=self._s3_client,
            metrics_collector=self._metrics,
        )

        current_artifact = _build_current_artifact(
            manifest_ref=manifest_url,
            manifest_content_type=artifact.content_type,
            run_id=run_id,
        )
        current_url = join_s3(self.s3_prefix, self.current_name)
        put_bytes(
            current_url,
            current_artifact.payload,
            current_artifact.content_type,
            current_artifact.headers,
            s3_client=self._s3_client,
            metrics_collector=self._metrics,
        )

        return manifest_url

    def load_current(self) -> CurrentPointer | None:
        current_url = f"{self.s3_prefix}/{self.current_name}"
        payload = try_get_bytes(
            current_url, s3_client=self._s3_client, metrics_collector=self._metrics
        )
        if payload is None:
            return None

        try:
            return CurrentPointer.model_validate_json(payload)
        except ValidationError as exc:
            raise ManifestParseError(
                f"CURRENT pointer validation failed: {exc}"
            ) from exc

    def load_manifest(self, ref: str) -> ParsedManifest:
        try:
            payload = get_bytes(
                ref, s3_client=self._s3_client, metrics_collector=self._metrics
            )
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


class InMemoryManifestStore:
    """In-memory manifest store for tests — replaces InMemoryPublisher."""

    def __init__(self) -> None:
        self._manifests: dict[str, ParsedManifest] = {}
        self._latest_ref: str | None = None
        self._latest_run_id: str | None = None

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        ref = f"mem://manifests/run_id={run_id}/manifest"
        self._manifests[ref] = ParsedManifest(
            required_build=required_build,
            shards=shards,
            custom=custom,
        )
        self._latest_ref = ref
        self._latest_run_id = run_id
        return ref

    def load_current(self) -> CurrentPointer | None:
        if self._latest_ref is None or self._latest_run_id is None:
            return None
        return CurrentPointer(
            manifest_ref=self._latest_ref,
            manifest_content_type="application/json",
            run_id=self._latest_run_id,
            updated_at="1970-01-01T00:00:00+00:00",
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifests[ref]


# ---------------------------------------------------------------------------
# Async manifest store protocol and implementations
# ---------------------------------------------------------------------------


class AsyncManifestStore(Protocol):
    """Read-only async manifest loading (no publish — that's writer-side)."""

    async def load_current(self) -> CurrentPointer | None: ...

    async def load_manifest(self, ref: str) -> ParsedManifest: ...


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
        current_name: str = "_CURRENT",
        s3_client_config: S3ClientConfig | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        import aiobotocore.session  # type: ignore[import-not-found]

        self.s3_prefix = s3_prefix.rstrip("/")
        self.current_name = current_name
        self._session = aiobotocore.session.get_session()
        self._resolved = _resolve_s3_config_for_aiobotocore(s3_client_config)
        self._metrics = metrics_collector

    async def load_current(self) -> CurrentPointer | None:
        current_url = f"{self.s3_prefix}/{self.current_name}"
        payload = await self._try_get_bytes(current_url)
        if payload is None:
            return None

        try:
            return CurrentPointer.model_validate_json(payload)
        except ValidationError as exc:
            raise ManifestParseError(
                f"CURRENT pointer validation failed: {exc}"
            ) from exc

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
        from .storage import parse_s3_url

        bucket, key = parse_s3_url(url)
        async with self._session.create_client("s3", **self._resolved) as client:
            obj = await client.get_object(Bucket=bucket, Key=key)  # type: ignore[misc]
            async with obj["Body"] as stream:
                return await stream.read()

    async def _do_try_get_bytes(self, url: str) -> bytes | None:
        from .storage import parse_s3_url

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

    async def load_current(self) -> CurrentPointer | None:
        import asyncio

        return await asyncio.to_thread(self._store.load_current)

    async def load_manifest(self, ref: str) -> ParsedManifest:
        import asyncio

        return await asyncio.to_thread(self._store.load_manifest, ref)


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


# ---------------------------------------------------------------------------
# Parsing helpers (migrated from manifest_readers.py)
# ---------------------------------------------------------------------------


def parse_json_manifest(payload: bytes) -> ParsedManifest:
    """Parse default JSON manifest payload into typed ParsedManifest."""

    try:
        parsed = ParsedManifest.model_validate_json(payload)
    except ValidationError as exc:
        raise ManifestParseError(f"Manifest validation failed: {exc}") from exc

    _validate_manifest(parsed.required_build, parsed.shards)
    return parsed


def _validate_manifest(
    required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
) -> None:
    num_dbs = required_build.num_dbs
    if len(shards) != num_dbs:
        raise ManifestParseError(
            f"Manifest shard count mismatch: expected {num_dbs}, got {len(shards)}"
        )

    ids = sorted(shard.db_id for shard in shards)
    expected = list(range(num_dbs))
    if ids != expected:
        raise ManifestParseError(
            f"Manifest shard coverage mismatch; expected {expected}, got {ids}"
        )

    if not required_build.sharding or not required_build.sharding.strategy:
        raise ManifestParseError("Manifest required.sharding.strategy is missing")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_current_artifact(
    *,
    manifest_ref: str,
    manifest_content_type: str,
    run_id: str,
) -> ManifestArtifact:
    from datetime import UTC, datetime

    pointer = CurrentPointer(
        manifest_ref=manifest_ref,
        manifest_content_type=manifest_content_type,
        run_id=run_id,
        updated_at=datetime.now(UTC).isoformat(),
    )
    payload = json.dumps(
        pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ManifestArtifact(payload=payload, content_type="application/json")
