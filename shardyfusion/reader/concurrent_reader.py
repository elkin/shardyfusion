"""Thread-safe sharded reader with lock or pool concurrency modes."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

from shardyfusion._rate_limiter import RateLimiter
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import (
    ManifestParseError,
    ReaderStateError,
    ShardyfusionError,
    SlateDbApiError,
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
    _LOCK_REFRESH,
    _LOCK_STATE,
    ShardReaderHandle,
    _close_handle,
    _close_state,
    _OrderedLock,
    _read_group,
    _read_one,
    _ReaderPool,
    _ReaderState,
    _ShardHandle,
)
from ._types import (
    ReaderHealth,
    ShardDetail,
    SnapshotInfo,
    _NullShardReader,
    _shard_details_from_router,
)

_logger = get_logger(__name__)


class ConcurrentShardedReader(_BaseShardedReader):
    """Thread-safe sharded reader with lock or pool concurrency modes."""

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: ManifestStore | None = None,
        current_name: str = "_CURRENT",
        reader_factory: ShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        thread_safety: Literal["lock", "pool"] = "lock",
        pool_checkout_timeout: float = 30.0,
        max_workers: int | None = None,
        max_fallback_attempts: int = 3,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        if thread_safety not in {"lock", "pool"}:
            raise ValueError("thread_safety must be 'lock' or 'pool'")

        if pool_checkout_timeout <= 0:
            raise ValueError(
                f"pool_checkout_timeout must be > 0, got {pool_checkout_timeout}"
            )

        super().__init__(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=manifest_store,
            current_name=current_name,
            reader_factory=reader_factory,
            slate_env_file=slate_env_file,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
            max_workers=max_workers,
            max_fallback_attempts=max_fallback_attempts,
            metrics_collector=metrics_collector,
            rate_limiter=rate_limiter,
        )

        self.thread_safety = thread_safety
        self._pool_checkout_timeout = pool_checkout_timeout
        # Lock ordering: _refresh_lock before _state_lock.  _refresh_lock
        # serialises the expensive I/O path (manifest load + reader open) so
        # only one thread builds new state at a time.  _state_lock protects
        # brief state swaps and refcount updates (no I/O held).
        self._refresh_lock = _OrderedLock(_LOCK_REFRESH, "_refresh_lock")
        self._state_lock = _OrderedLock(_LOCK_STATE, "_state_lock")
        self._state = self._load_initial_state()

        log_event(
            "reader_initialized",
            logger=_logger,
            s3_prefix=self.s3_prefix,
            num_shards=len(self._state.handles),
            manifest_ref=self._state.manifest_ref,
        )
        self._emit(MetricEvent.READER_INITIALIZED, {})

    @property
    def key_encoding(self) -> KeyEncoding:
        with self._use_state() as state:
            return KeyEncoding.from_value(state.router.key_encoding)

    def snapshot_info(self) -> SnapshotInfo:
        with self._use_state() as state:
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
        with self._use_state() as state:
            return _shard_details_from_router(state.router)

    def health(self, *, staleness_threshold_s: float | None = None) -> ReaderHealth:
        """Return a diagnostic snapshot of reader state."""
        with self._state_lock:
            if self._closed:
                return ReaderHealth(
                    status="unhealthy",
                    manifest_ref="",
                    manifest_age_seconds=0.0,
                    num_shards=0,
                    is_closed=True,
                )
            state = self._state
        age = time.monotonic() - self._init_time
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if staleness_threshold_s is not None and age > staleness_threshold_s:
            status = "degraded"
        return ReaderHealth(
            status=status,
            manifest_ref=state.manifest_ref,
            manifest_age_seconds=round(age, 2),
            num_shards=len(state.handles),
            is_closed=False,
        )

    def route_key(
        self,
        key: KeyInput,
        *,
        routing_context: dict[str, object] | None = None,
    ) -> int:
        """Return the shard db_id a key would route to."""
        with self._use_state() as state:
            return state.router.route(key, routing_context=routing_context)

    def shard_for_key(self, key: KeyInput) -> RequiredShardMeta:
        """Return shard metadata for the shard a key routes to."""
        with self._use_state() as state:
            db_id = state.router.route_one(key)
            return state.router.shards[db_id]

    def shards_for_keys(
        self, keys: Sequence[KeyInput]
    ) -> dict[KeyInput, RequiredShardMeta]:
        """Return a mapping of keys to their shard metadata."""
        with self._use_state() as state:
            router = state.router
            return {key: router.shards[router.route_one(key)] for key in keys}

    def reader_for_key(self, key: KeyInput) -> ShardReaderHandle:
        """Return a borrowed read handle for the shard a key routes to.

        The returned handle holds a refcount on the current reader state.
        Callers **must** call ``close()`` on the returned handle when done.
        """
        state = self._acquire_state()
        db_id = state.router.route_one(key)
        handle = state.handles[db_id]
        return ShardReaderHandle(handle, release_fn=lambda: self._release_state(state))

    def readers_for_keys(
        self, keys: Sequence[KeyInput]
    ) -> dict[KeyInput, ShardReaderHandle]:
        """Return a mapping of keys to borrowed read handles.

        Each returned handle holds one refcount increment.  Callers **must**
        call ``close()`` on every returned handle when done.
        """
        unique_keys = dict.fromkeys(keys)
        if not unique_keys:
            return {}
        with self._state_lock:
            if self._closed:
                raise ReaderStateError("Reader is closed")
            state = self._state
            state.refcount += len(unique_keys)
        result: dict[KeyInput, ShardReaderHandle] = {}
        for key in unique_keys:
            db_id = state.router.route_one(key)
            handle = state.handles[db_id]
            result[key] = ShardReaderHandle(
                handle, release_fn=lambda: self._release_state(state)
            )
        return result

    def get(
        self,
        key: KeyInput,
        *,
        routing_context: dict[str, object] | None = None,
    ) -> bytes | None:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire(1)

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        with self._use_state() as state:
            db_id = state.router.route(key, routing_context=routing_context)
            key_bytes = state.router.encode_lookup_key(key)
            handle = state.handles[db_id]
            result = _read_one(handle, key_bytes)

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
        if self._rate_limiter is not None:
            self._rate_limiter.acquire(len(keys))

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        key_list = list(keys)
        with self._use_state() as state:
            grouped = state.router.group_keys(key_list, routing_context=routing_context)
            results: dict[KeyInput, bytes | None] = {}

            if self._executor is not None and len(grouped) > 1:
                futures = {
                    db_id: self._executor.submit(
                        _read_group,
                        state.router,
                        state.handles[db_id],
                        shard_keys,
                    )
                    for db_id, shard_keys in grouped.items()
                }
                for db_id, future in futures.items():
                    try:
                        results.update(future.result())
                    except ShardyfusionError:
                        raise
                    except Exception as exc:
                        raise SlateDbApiError(
                            f"Read failed for shard db_id={db_id}"
                        ) from exc
            else:
                for db_id, shard_keys in grouped.items():
                    results.update(
                        _read_group(state.router, state.handles[db_id], shard_keys)
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
        with self._refresh_lock:
            return self._do_refresh()

    def _do_refresh(self) -> bool:
        assert self._refresh_lock.locked(), "_do_refresh requires _refresh_lock"
        current = self._manifest_store.load_current()
        if current is None:
            log_failure(
                "reader_refresh_current_not_found",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                s3_prefix=self.s3_prefix,
            )
            raise ReaderStateError("CURRENT pointer not found during refresh")

        current_ref = current.ref
        with self._state_lock:
            if self._closed:
                raise ReaderStateError("Reader is closed")
            old_ref_snapshot = self._state.manifest_ref
            if current_ref == old_ref_snapshot:
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
        new_state = self._build_state(current_ref, manifest)

        old_state: _ReaderState
        should_close_old = False
        with self._state_lock:
            if self._closed:
                _close_state(new_state)
                raise ReaderStateError("Reader is closed")
            if self._state.manifest_ref != old_ref_snapshot:
                # Another refresh already advanced past us; discard our work.
                _close_state(new_state)
                log_event(
                    "reader_refresh_superseded",
                    logger=_logger,
                    current_ref=current_ref,
                    active_ref=self._state.manifest_ref,
                )
                self._emit(MetricEvent.READER_REFRESHED, {"changed": False})
                return False
            old_state = self._state
            self._state = new_state
            old_state.retired = True
            should_close_old = old_state.refcount == 0

        if should_close_old:
            _close_state(old_state)

        log_event(
            "reader_refreshed",
            logger=_logger,
            old_manifest_ref=old_state.manifest_ref,
            new_manifest_ref=current_ref,
        )
        self._emit(MetricEvent.READER_REFRESHED, {"changed": True})
        return True

    def close(self) -> None:
        state_to_close: _ReaderState | None = None
        num_handles = 0
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            num_handles = len(self._state.handles)
            self._state.retired = True
            if self._state.refcount == 0:
                state_to_close = self._state

        if state_to_close is not None:
            _close_state(state_to_close)

        self._shutdown_executor()

        log_event(
            "reader_closed",
            logger=_logger,
            s3_prefix=self.s3_prefix,
            num_handles_closed=num_handles,
        )
        self._emit(MetricEvent.READER_CLOSED, {"num_handles": num_handles})

    def _load_initial_state(self) -> _ReaderState:
        current = self._load_current()
        try:
            manifest = self._manifest_store.load_manifest(current.ref)
            return self._build_state(current.ref, manifest)
        except ManifestParseError:
            if self._max_fallback_attempts == 0:
                raise
            return self._fallback_to_previous_concurrent(failed_ref=current.ref)

    def _fallback_to_previous_concurrent(self, *, failed_ref: str) -> _ReaderState:
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
                return self._build_state(ref.ref, manifest)
            except ManifestParseError:
                log_failure(
                    "reader_fallback_skip_malformed",
                    severity=FailureSeverity.ERROR,
                    logger=_logger,
                    manifest_ref=ref.ref,
                )
                continue
        raise ReaderStateError("No valid manifest found after fallback")

    def _build_state(self, manifest_ref: str, manifest: ParsedManifest) -> _ReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        handles: dict[int, _ShardHandle] = {}

        try:
            for shard in router.shards:
                if shard.db_url is None:
                    null_reader: ShardReader = _NullShardReader()
                    if self.thread_safety == "lock":
                        handles[shard.db_id] = _ShardHandle(
                            mode="lock",
                            reader=null_reader,
                            lock=threading.Lock(),
                        )
                    else:
                        handles[shard.db_id] = _ShardHandle(
                            mode="pool",
                            pool=_ReaderPool(
                                [null_reader],
                                checkout_timeout=self._pool_checkout_timeout,
                            ),
                        )
                elif self.thread_safety == "lock":
                    reader = self._open_one_reader(shard)
                    handles[shard.db_id] = _ShardHandle(
                        mode="lock",
                        reader=reader,
                        lock=threading.Lock(),
                    )
                else:
                    pool_size = self.max_workers or 4
                    readers: list[ShardReader] = []
                    try:
                        for _ in range(pool_size):
                            readers.append(self._open_one_reader(shard))
                    except Exception:
                        for r in readers:
                            try:
                                r.close()
                            except Exception:
                                pass
                        raise
                    handles[shard.db_id] = _ShardHandle(
                        mode="pool",
                        pool=_ReaderPool(
                            readers, checkout_timeout=self._pool_checkout_timeout
                        ),
                    )
        except Exception:
            for handle in handles.values():
                _close_handle(handle)
            raise

        return _ReaderState(manifest_ref=manifest_ref, router=router, handles=handles)

    @contextmanager
    def _use_state(self) -> Iterator[_ReaderState]:
        state = self._acquire_state()
        try:
            yield state
        finally:
            self._release_state(state)

    def _acquire_state(self) -> _ReaderState:
        with self._state_lock:
            if self._closed:
                raise ReaderStateError("Reader is closed")
            state = self._state
            state.refcount += 1
            log_event(
                "reader_state_acquired",
                level=logging.DEBUG,
                logger=_logger,
                refcount=state.refcount,
            )
            return state

    def _release_state(self, state: _ReaderState) -> None:
        should_close = False
        with self._state_lock:
            state.refcount -= 1
            log_event(
                "reader_state_released",
                level=logging.DEBUG,
                logger=_logger,
                refcount=state.refcount,
            )
            should_close = state.retired and state.refcount == 0

        if should_close:
            try:
                _close_state(state)
            except Exception:
                # Deferred cleanup of retired state — per-handle failures are
                # already logged inside _close_state.  Do not propagate to the
                # caller (who is finishing an unrelated read operation).
                pass
