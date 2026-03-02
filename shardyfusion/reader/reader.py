"""Service-side sharded SlateDB reader helpers."""

import logging
import os
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
from typing import Any

from shardyfusion.errors import (
    ReaderStateError,
    SlateDbApiError,
    SlatedbSparkShardedError,
)
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)
from shardyfusion.manifest import ParsedManifest
from shardyfusion.manifest_readers import (
    DefaultS3ManifestReader,
    ManifestReader,
)
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.type_defs import KeyInput, ShardReader, ShardReaderFactory

_logger = get_logger(__name__)


@dataclass(slots=True)
class SlateDbReaderFactory:
    """Picklable reader factory that captures SlateDB-specific config."""

    env_file: str | None = None

    def __call__(
        self, *, db_url: str, local_dir: str, checkpoint_id: str | None
    ) -> ShardReader:
        try:
            from slatedb import SlateDBReader
        except ImportError as exc:  # pragma: no cover - runtime dependent
            raise SlateDbApiError(
                "slatedb package is required for SlateShardedReader"
            ) from exc

        return SlateDBReader(
            local_dir,
            url=db_url,
            env_file=self.env_file,
            checkpoint_id=checkpoint_id,
        )


@dataclass(slots=True)
class _ReaderState:
    manifest_ref: str
    router: SnapshotRouter
    handles: dict[int, "_ShardHandle"]
    refcount: int = 0
    retired: bool = False


@dataclass(slots=True)
class _ShardHandle:
    mode: str
    reader: ShardReader | None = None
    lock: "threading.Lock | None" = None
    pool: "_ReaderPool | None" = None


class _ReaderPool:
    def __init__(self, readers: list[ShardReader]) -> None:
        if not readers:
            raise ValueError("Reader pool requires at least one reader")
        self._readers = readers
        self._indexes: Queue[int] = Queue()
        for idx in range(len(readers)):
            self._indexes.put(idx)

    @contextmanager
    def checkout(self) -> Iterator[ShardReader]:
        idx = self._indexes.get()
        try:
            yield self._readers[idx]
        finally:
            self._indexes.put(idx)

    def close(self) -> None:
        for reader in self._readers:
            try:
                reader.close()
            except Exception as exc:
                log_failure(
                    "reader_pool_member_close_failed",
                    severity=FailureSeverity.ERROR,
                    logger=_logger,
                    error=exc,
                )


class SlateShardedReader:
    """Load latest snapshot manifest and perform routed key lookups."""

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_reader: ManifestReader | None = None,
        current_name: str = "_CURRENT",
        reader_factory: ShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        thread_safety: str = "lock",
        max_workers: int | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        if thread_safety not in {"lock", "pool"}:
            raise ValueError("thread_safety must be 'lock' or 'pool'")

        self.s3_prefix = s3_prefix
        self.local_root = local_root
        self.thread_safety = thread_safety
        self.max_workers = max_workers
        self._metrics = metrics_collector

        # Resolve reader factory: explicit > legacy env_file > default
        if reader_factory is not None:
            self._reader_factory: ShardReaderFactory = reader_factory
        else:
            self._reader_factory = SlateDbReaderFactory(env_file=slate_env_file)

        if manifest_reader is not None:
            self._manifest_reader = manifest_reader
        else:
            self._manifest_reader = DefaultS3ManifestReader(
                s3_prefix,
                current_name=current_name,
                metrics_collector=metrics_collector,
            )

        self._state_lock = threading.Lock()
        self._closed = False
        self._state = self._load_initial_state()

        log_event(
            "reader_initialized",
            logger=_logger,
            s3_prefix=self.s3_prefix,
            num_shards=len(self._state.handles),
            manifest_ref=self._state.manifest_ref,
        )
        if self._metrics is not None:
            self._metrics.emit(MetricEvent.READER_INITIALIZED, {})

    @property
    def key_encoding(self) -> KeyEncoding:
        """The key encoding used by the loaded manifest."""
        return KeyEncoding.from_value(self._state.router.key_encoding)

    def snapshot_info(self) -> dict[str, Any]:
        """Return a dict of manifest metadata for the current snapshot."""
        state = self._state
        rb = state.router.required_build
        return {
            "run_id": rb.run_id,
            "num_dbs": rb.num_dbs,
            "sharding": rb.sharding.strategy.value,
            "created_at": rb.created_at,
            "manifest_ref": state.manifest_ref,
        }

    def get(self, key: KeyInput) -> bytes | None:
        """Get one key from the currently loaded snapshot."""

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        with self._use_state() as state:
            db_id = state.router.route_one(key)
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

    def multi_get(self, keys: Sequence[KeyInput]) -> dict[KeyInput, bytes | None]:
        """Get multiple keys with per-shard grouping and optional shard parallelism."""

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        key_list = list(keys)
        with self._use_state() as state:
            grouped = state.router.group_keys(key_list)
            results: dict[KeyInput, bytes | None] = {}

            if self.max_workers and self.max_workers > 1 and len(grouped) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        db_id: executor.submit(
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
                        except SlatedbSparkShardedError:
                            raise  # domain exceptions propagate unchanged
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
        """Reload CURRENT and manifest, atomically swapping readers when ref changes."""

        current = self._manifest_reader.load_current()
        if current is None:
            log_failure(
                "reader_refresh_current_not_found",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                s3_prefix=self.s3_prefix,
            )
            raise ReaderStateError("CURRENT pointer not found during refresh")

        current_ref = current.manifest_ref
        with self._state_lock:
            if self._closed:
                raise ReaderStateError("Reader is closed")
            if current_ref == self._state.manifest_ref:
                log_event(
                    "reader_refresh_unchanged",
                    logger=_logger,
                    manifest_ref=current_ref,
                )
                if self._metrics is not None:
                    self._metrics.emit(MetricEvent.READER_REFRESHED, {"changed": False})
                return False

        manifest = self._manifest_reader.load_manifest(
            current_ref,
            current.manifest_content_type,
        )
        new_state = self._build_state(current_ref, manifest)

        old_state: _ReaderState
        should_close_old = False
        with self._state_lock:
            if self._closed:
                _close_state(new_state)
                raise ReaderStateError("Reader is closed")
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
        if self._metrics is not None:
            self._metrics.emit(MetricEvent.READER_REFRESHED, {"changed": True})

        return True

    def close(self) -> None:
        """Close all active readers and prevent further operations."""

        state_to_close: _ReaderState | None = None
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._state.retired = True
            if self._state.refcount == 0:
                state_to_close = self._state

        if state_to_close is not None:
            _close_state(state_to_close)

        log_event(
            "reader_closed",
            logger=_logger,
            s3_prefix=self.s3_prefix,
            num_handles_closed=len(self._state.handles),
        )
        if self._metrics is not None:
            self._metrics.emit(
                MetricEvent.READER_CLOSED,
                {
                    "num_handles": len(self._state.handles),
                },
            )

    def __enter__(self) -> "SlateShardedReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _load_initial_state(self) -> _ReaderState:
        current = self._manifest_reader.load_current()
        if current is None:
            log_failure(
                "reader_current_not_found",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                s3_prefix=self.s3_prefix,
            )
            raise ReaderStateError("CURRENT pointer not found")

        manifest = self._manifest_reader.load_manifest(
            current.manifest_ref,
            current.manifest_content_type,
        )
        return self._build_state(current.manifest_ref, manifest)

    def _build_state(self, manifest_ref: str, manifest: ParsedManifest) -> _ReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        handles: dict[int, _ShardHandle] = {}

        try:
            for shard in manifest.shards:
                local_path = os.path.join(self.local_root, f"shard={shard.db_id:05d}")
                os.makedirs(local_path, exist_ok=True)

                if self.thread_safety == "lock":
                    reader = self._reader_factory(
                        db_url=shard.db_url,
                        local_dir=local_path,
                        checkpoint_id=shard.checkpoint_id,
                    )
                    handles[shard.db_id] = _ShardHandle(
                        mode="lock",
                        reader=reader,
                        lock=threading.Lock(),
                    )
                else:
                    pool_size = self.max_workers or 4
                    readers = [
                        self._reader_factory(
                            db_url=shard.db_url,
                            local_dir=local_path,
                            checkpoint_id=shard.checkpoint_id,
                        )
                        for _ in range(pool_size)
                    ]
                    handles[shard.db_id] = _ShardHandle(
                        mode="pool",
                        pool=_ReaderPool(readers),
                    )
        except Exception:
            for handle in handles.values():
                _close_handle(handle)
            raise

        return _ReaderState(manifest_ref=manifest_ref, router=router, handles=handles)

    @contextmanager
    def _use_state(self) -> Iterator[_ReaderState]:
        """Acquire and release a refcounted snapshot of the current state."""
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
            _close_state(state)


def _read_group(
    router: SnapshotRouter,
    handle: _ShardHandle,
    keys: list[KeyInput],
) -> dict[KeyInput, bytes | None]:
    results: dict[KeyInput, bytes | None] = {}
    for key in keys:
        key_bytes = router.encode_lookup_key(key)
        results[key] = _read_one(handle, key_bytes)
    return results


def _read_one(handle: _ShardHandle, key: bytes) -> bytes | None:
    if handle.mode == "lock":
        assert handle.lock is not None
        assert handle.reader is not None
        with handle.lock:
            return handle.reader.get(key)

    assert handle.pool is not None
    with handle.pool.checkout() as reader:
        return reader.get(key)


def _open_slatedb_reader(
    *,
    local_path: str,
    db_url: str,
    checkpoint_id: str | None,
    env_file: str | None,
) -> ShardReader:
    """Legacy helper kept for backward-compatible monkeypatching in tests."""
    return SlateDbReaderFactory(env_file=env_file)(
        db_url=db_url,
        local_dir=local_path,
        checkpoint_id=checkpoint_id,
    )


def _close_state(state: _ReaderState) -> None:
    errors: list[tuple[int, BaseException]] = []
    for db_id, handle in state.handles.items():
        try:
            _close_handle(handle)
        except Exception as exc:
            errors.append((db_id, exc))
            log_failure(
                "reader_handle_close_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                db_id=db_id,
                manifest_ref=state.manifest_ref,
            )
    if errors:
        log_failure(
            "reader_state_close_partial_failure",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            error=errors[0][1],
            failed_db_ids=[db_id for db_id, _ in errors],
            total_handles=len(state.handles),
        )


def _close_handle(handle: _ShardHandle) -> None:
    if handle.mode == "lock" and handle.reader is not None:
        handle.reader.close()
    if handle.mode == "pool" and handle.pool is not None:
        handle.pool.close()
