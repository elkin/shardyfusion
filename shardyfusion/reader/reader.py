"""Service-side sharded reader — non-thread-safe variant."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Literal

from shardyfusion._rate_limiter import RateLimiter
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import (
    DbAdapterError,
    ManifestParseError,
    ReaderStateError,
    ShardyfusionError,
)
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)
from shardyfusion.manifest import ParsedManifest, RequiredShardMeta
from shardyfusion.manifest_store import ManifestStore
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.type_defs import (
    KeyInput,
    S3ConnectionOptions,
    ShardReader,
    ShardReaderFactory,
)

from ._base import _BaseShardedReader
from ._state import (
    ShardReaderHandle,
    _close_simple_state,
    _read_group_simple,
    _SimpleReaderState,
)
from ._types import (
    ReaderHealth,
    ShardDetail,
    SnapshotInfo,
    _NullShardReader,
    _shard_details_from_router,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ShardedReader — non-thread-safe, no locks, no refcounting
# ---------------------------------------------------------------------------


class ShardedReader(_BaseShardedReader):
    """Non-thread-safe sharded reader for single-threaded services.

    Provides the same ``get`` / ``multi_get`` / ``refresh`` API as
    ``ConcurrentShardedReader`` but without locks or reference counting.
    """

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: ManifestStore | None = None,
        current_pointer_key: str = "_CURRENT",
        reader_factory: ShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        max_workers: int | None = None,
        max_fallback_attempts: int = 3,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        super().__init__(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=manifest_store,
            current_pointer_key=current_pointer_key,
            reader_factory=reader_factory,
            slate_env_file=slate_env_file,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
            max_workers=max_workers,
            max_fallback_attempts=max_fallback_attempts,
            metrics_collector=metrics_collector,
            rate_limiter=rate_limiter,
        )
        self._state = self._load_initial_state()

        log_event(
            "reader_initialized",
            logger=_logger,
            s3_prefix=self.s3_prefix,
            num_shards=len(self._state.readers),
            manifest_ref=self._state.manifest_ref,
        )
        self._emit(MetricEvent.READER_INITIALIZED, {})

    @property
    def key_encoding(self) -> KeyEncoding:
        return KeyEncoding.from_value(self._state.router.key_encoding)

    def snapshot_info(self) -> SnapshotInfo:
        state = self._state
        rb = state.router.required_build
        return SnapshotInfo(
            run_id=rb.run_id,
            num_dbs=rb.num_dbs,
            sharding=rb.sharding.strategy,
            created_at=rb.created_at,
            manifest_ref=state.manifest_ref,
            key_encoding=rb.key_encoding,
            row_count=sum(s.row_count for s in state.router.shards),
        )

    def shard_details(self) -> list[ShardDetail]:
        """Return per-shard metadata from the current manifest."""
        return _shard_details_from_router(self._state.router)

    def health(self, *, staleness_threshold: timedelta | None = None) -> ReaderHealth:
        """Return a diagnostic snapshot of reader state."""
        if self._closed:
            return ReaderHealth(
                status="unhealthy",
                manifest_ref="",
                manifest_age=timedelta(0),
                num_shards=0,
                is_closed=True,
            )
        age = time.monotonic() - self._init_time
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if (
            staleness_threshold is not None
            and age > staleness_threshold.total_seconds()
        ):
            status = "degraded"
        return ReaderHealth(
            status=status,
            manifest_ref=self._state.manifest_ref,
            manifest_age=timedelta(seconds=round(age, 2)),
            num_shards=len(self._state.readers),
            is_closed=False,
        )

    def route_key(
        self,
        key: KeyInput,
        *,
        routing_context: dict[str, object] | None = None,
    ) -> int:
        """Return the shard db_id a key would route to."""
        return self._state.router.route(key, routing_context=routing_context)

    def shard_for_key(
        self,
        key: KeyInput,
        *,
        routing_context: dict[str, object] | None = None,
    ) -> RequiredShardMeta:
        """Return shard metadata for the shard a key routes to."""
        db_id = self._state.router.route(key, routing_context=routing_context)
        return self._state.router.shards[db_id]

    def shards_for_keys(
        self, keys: Sequence[KeyInput]
    ) -> dict[KeyInput, RequiredShardMeta]:
        """Return a mapping of keys to their shard metadata."""
        router = self._state.router
        return {key: router.shards[router.route_one(key)] for key in keys}

    def reader_for_key(self, key: KeyInput) -> ShardReaderHandle:
        """Return a borrowed read handle for the shard a key routes to."""
        if self._closed:
            raise ReaderStateError("Reader is closed")
        state = self._state
        state.borrow_count += 1
        db_id = state.router.route_one(key)
        return ShardReaderHandle(
            state.readers[db_id], release_fn=lambda: self._release_simple_state(state)
        )

    def readers_for_keys(
        self, keys: Sequence[KeyInput]
    ) -> dict[KeyInput, ShardReaderHandle]:
        """Return a mapping of keys to borrowed read handles."""
        if self._closed:
            raise ReaderStateError("Reader is closed")
        state = self._state
        unique_keys = dict.fromkeys(keys)
        state.borrow_count += len(unique_keys)
        return {
            key: ShardReaderHandle(
                state.readers[state.router.route_one(key)],
                release_fn=lambda: self._release_simple_state(state),
            )
            for key in unique_keys
        }

    def get(
        self,
        key: KeyInput,
        *,
        routing_context: dict[str, object] | None = None,
    ) -> bytes | None:
        if self._closed:
            raise ReaderStateError("Reader is closed")

        if self._rate_limiter is not None:
            self._rate_limiter.acquire(1)

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        state = self._state
        db_id = state.router.route(key, routing_context=routing_context)
        key_bytes = state.router.encode_lookup_key(key)
        result = state.readers[db_id].get(key_bytes)

        if mc is not None:
            mc.emit(
                MetricEvent.READER_GET,
                {
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                    "found": result is not None,
                },
            )

        return result

    def multi_get(
        self,
        keys: Sequence[KeyInput],
        *,
        routing_context: dict[str, object] | None = None,
    ) -> dict[KeyInput, bytes | None]:
        if self._closed:
            raise ReaderStateError("Reader is closed")

        if self._rate_limiter is not None:
            self._rate_limiter.acquire(len(keys))

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        key_list = list(keys)
        state = self._state

        grouped = state.router.group_keys(key_list, routing_context=routing_context)
        results: dict[KeyInput, bytes | None] = {}

        if self._executor is not None and len(grouped) > 1:
            futures = {
                db_id: self._executor.submit(
                    _read_group_simple, state.router, state.readers[db_id], shard_keys
                )
                for db_id, shard_keys in grouped.items()
            }
            for db_id, future in futures.items():
                try:
                    results.update(future.result())
                except ShardyfusionError:
                    raise
                except Exception as exc:
                    raise DbAdapterError(
                        f"Read failed for shard db_id={db_id}"
                    ) from exc
        else:
            for db_id, shard_keys in grouped.items():
                results.update(
                    _read_group_simple(state.router, state.readers[db_id], shard_keys)
                )

        ordered = {key: results.get(key) for key in key_list}

        if mc is not None:
            mc.emit(
                MetricEvent.READER_MULTI_GET,
                {
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                    "num_keys": len(key_list),
                },
            )

        return ordered

    def refresh(self) -> bool:
        if self._closed:
            raise ReaderStateError("Reader is closed")

        current = self._load_current()
        current_ref = current.ref

        if current_ref == self._state.manifest_ref:
            log_event(
                "reader_refresh_unchanged",
                logger=_logger,
                manifest_ref=current_ref,
            )
            self._emit(MetricEvent.READER_REFRESHED, {"changed": False})
            return False

        try:
            manifest = self._manifest_store.load_manifest(current_ref)
        except ManifestParseError:
            log_failure(
                "reader_refresh_malformed_manifest",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                manifest_ref=current_ref,
            )
            self._emit(MetricEvent.READER_REFRESHED, {"changed": False})
            return False

        new_state = self._build_simple_state(current_ref, manifest)
        old_state = self._state
        self._state = new_state
        old_state.retired = True
        if old_state.borrow_count == 0:
            _close_simple_state(old_state)

        log_event(
            "reader_refreshed",
            logger=_logger,
            old_manifest_ref=old_state.manifest_ref,
            new_manifest_ref=current_ref,
        )
        self._emit(MetricEvent.READER_REFRESHED, {"changed": True})
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        state = self._state
        num_readers = len(state.readers)
        state.retired = True
        if state.borrow_count == 0:
            _close_simple_state(state)
        self._shutdown_executor()

        log_event(
            "reader_closed",
            logger=_logger,
            s3_prefix=self.s3_prefix,
            num_handles_closed=num_readers,
        )
        self._emit(MetricEvent.READER_CLOSED, {"num_handles": num_readers})

    def _release_simple_state(self, state: _SimpleReaderState) -> None:
        state.borrow_count -= 1
        if state.retired and state.borrow_count == 0:
            _close_simple_state(state)

    def _load_initial_state(self) -> _SimpleReaderState:
        current = self._load_current()
        try:
            manifest = self._manifest_store.load_manifest(current.ref)
            return self._build_simple_state(current.ref, manifest)
        except ManifestParseError:
            if self._max_fallback_attempts == 0:
                raise
            return self._fallback_to_previous(failed_ref=current.ref)

    def _fallback_to_previous(self, *, failed_ref: str) -> _SimpleReaderState:
        """Walk backward through manifest history to find a valid manifest."""
        log_failure(
            "reader_cold_start_fallback",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            failed_ref=failed_ref,
            max_attempts=self._max_fallback_attempts,
        )
        refs = self._manifest_store.list_manifests(
            limit=self._max_fallback_attempts + 1
        )
        for ref in refs:
            if ref.ref == failed_ref:
                continue
            try:
                manifest = self._manifest_store.load_manifest(ref.ref)
                log_event(
                    "reader_fallback_succeeded",
                    logger=_logger,
                    fallback_ref=ref.ref,
                    skipped_ref=failed_ref,
                )
                return self._build_simple_state(ref.ref, manifest)
            except ManifestParseError:
                log_failure(
                    "reader_fallback_skip_malformed",
                    severity=FailureSeverity.ERROR,
                    logger=_logger,
                    manifest_ref=ref.ref,
                )
                continue
        raise ReaderStateError("No valid manifest found after fallback")

    def _build_simple_state(
        self, manifest_ref: str, manifest: ParsedManifest
    ) -> _SimpleReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        readers: dict[int, ShardReader] = {}
        try:
            for shard in router.shards:
                if shard.db_url is None:
                    readers[shard.db_id] = _NullShardReader()
                else:
                    readers[shard.db_id] = self._open_one_reader(shard)
        except Exception:
            for reader in readers.values():
                try:
                    reader.close()
                except Exception:
                    pass
            raise
        return _SimpleReaderState(
            manifest_ref=manifest_ref, router=router, readers=readers
        )
