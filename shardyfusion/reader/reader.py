"""Service-side sharded SlateDB reader helpers."""

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal, Self

from shardyfusion.errors import (
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
from shardyfusion.manifest_store import ManifestStore, S3ManifestStore
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.type_defs import KeyInput, ShardReader, ShardReaderFactory

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lock ordering validator
# ---------------------------------------------------------------------------

# Lock ordering levels (lower number = must be acquired first):
#   _refresh_lock  = 0  — serialises refresh I/O
#   _state_lock    = 1  — guards state pointer + refcount
_LOCK_REFRESH = 0
_LOCK_STATE = 1


class _OrderedLock:
    """Lock wrapper that validates acquisition ordering in debug builds.

    Each thread tracks its currently-held lock levels via thread-local
    storage.  Acquiring a lock whose level is ``<=`` the maximum already
    held by the current thread raises ``AssertionError``, catching
    reverse-order acquisition at development/test time.

    Under ``python -O`` the tracking is compiled away and this behaves
    as a plain ``threading.Lock``.
    """

    _tls = threading.local()
    __slots__ = ("_lock", "_level", "_name")

    def __init__(self, level: int, name: str) -> None:
        self._lock = threading.Lock()
        self._level = level
        self._name = name

    def __enter__(self) -> "_OrderedLock":
        if __debug__:
            stack = getattr(self._tls, "stack", None)
            if stack is None:
                stack = []
                self._tls.stack = stack  # type: ignore[attr-defined]
            if stack:
                top_level, top_name = stack[-1]
                if top_level >= self._level:
                    raise AssertionError(
                        f"Lock ordering violation: acquiring {self._name} "
                        f"(level={self._level}) while holding {top_name} "
                        f"(level={top_level})"
                    )
            stack.append((self._level, self._name))
        self._lock.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self._lock.release()
        if __debug__:
            stack: list[tuple[int, str]] = getattr(self._tls, "stack", [])
            if stack and stack[-1][0] == self._level:
                stack.pop()

    def locked(self) -> bool:
        """Return ``True`` if the lock is currently held by any thread."""
        return self._lock.locked()


@dataclass(slots=True, frozen=True)
class SnapshotInfo:
    """Read-only snapshot of manifest metadata."""

    run_id: str
    num_dbs: int
    sharding: str
    created_at: str
    manifest_ref: str
    key_encoding: str = "u64be"
    row_count: int = 0


@dataclass(slots=True, frozen=True)
class ShardDetail:
    """Per-shard metadata exposed by ``shard_details()``."""

    db_id: int
    row_count: int
    min_key: int | float | str | None
    max_key: int | float | str | None
    db_url: str


@dataclass(slots=True)
class SlateDbReaderFactory:
    """Picklable reader factory that captures SlateDB-specific config."""

    env_file: str | None = None

    def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> ShardReader:
        try:
            from slatedb import SlateDBReader
        except ImportError as exc:  # pragma: no cover - runtime dependent
            raise SlateDbApiError(
                "slatedb package is required for reading shards"
            ) from exc

        return SlateDBReader(
            str(local_dir),
            url=db_url,
            env_file=self.env_file,
            checkpoint_id=checkpoint_id,
        )


# ---------------------------------------------------------------------------
# Shared reader state types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SimpleReaderState:
    """Lightweight reader state with borrow tracking — used by ShardedReader."""

    manifest_ref: str
    router: SnapshotRouter
    readers: dict[int, ShardReader]
    borrow_count: int = 0
    retired: bool = False


@dataclass(slots=True)
class _ReaderState:
    """Thread-safe reader state with refcounting — used by ConcurrentShardedReader."""

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
    def __init__(
        self, readers: list[ShardReader], *, checkout_timeout: float = 30.0
    ) -> None:
        if not readers:
            raise ValueError("Reader pool requires at least one reader")
        self._readers = readers
        self._checkout_timeout = checkout_timeout
        self._indexes: Queue[int] = Queue()
        for idx in range(len(readers)):
            self._indexes.put(idx)

    @contextmanager
    def checkout(self) -> Iterator[ShardReader]:
        try:
            idx = self._indexes.get(timeout=self._checkout_timeout)
        except Empty:
            raise SlateDbApiError(
                f"Shard reader pool checkout timed out after {self._checkout_timeout}s"
            ) from None
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


class ShardReaderHandle:
    """Borrowed reference to a single shard's database handle.

    Wraps either a raw ``ShardReader`` (from ``ShardedReader``) or a
    ``_ShardHandle`` (from ``ConcurrentShardedReader``).  Calling
    ``close()`` releases the borrow (decrements refcount / borrow count)
    but does **not** close the underlying shard database.

    Supports the context manager protocol::

        with reader.reader_for_key(42) as handle:
            raw = handle.get(key_bytes)
    """

    def __init__(
        self,
        handle: "_ShardHandle | ShardReader",
        release_fn: "Callable[[], None] | None" = None,
    ) -> None:
        self._handle = handle
        self._release_fn = release_fn
        self._released = False

    def get(self, key: bytes) -> bytes | None:
        """Read a single key from the underlying shard."""
        handle = self._handle
        if isinstance(handle, _ShardHandle):
            return _read_one(handle, key)
        return handle.get(key)

    def multi_get(self, keys: Sequence[bytes]) -> dict[bytes, bytes | None]:
        """Read multiple keys from the underlying shard."""
        return {key: self.get(key) for key in keys}

    def close(self) -> None:
        """Release the borrow.  Safe to call multiple times."""
        if self._released:
            return
        self._released = True
        if self._release_fn is not None:
            self._release_fn()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Base reader class
# ---------------------------------------------------------------------------


class _BaseShardedReader:
    """Shared constructor and utilities for ShardedReader and ConcurrentShardedReader."""

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: ManifestStore | None = None,
        current_name: str = "_CURRENT",
        reader_factory: ShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        max_workers: int | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self.s3_prefix = s3_prefix
        self.local_root = local_root
        self.max_workers = max_workers
        self._metrics = metrics_collector

        if reader_factory is not None:
            self._reader_factory: ShardReaderFactory = reader_factory
        else:
            self._reader_factory = SlateDbReaderFactory(env_file=slate_env_file)

        if manifest_store is not None:
            self._manifest_store = manifest_store
        else:
            self._manifest_store = S3ManifestStore(
                s3_prefix,
                current_name=current_name,
                metrics_collector=metrics_collector,
            )

        self._closed = False

        # Shared executor for multi_get parallelism (created once, reused)
        if max_workers is not None and max_workers > 1:
            self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
                max_workers=max_workers
            )
        else:
            self._executor = None

    def _load_current(self) -> Any:
        """Load CURRENT pointer, raising ReaderStateError if missing."""
        current = self._manifest_store.load_current()
        if current is None:
            log_failure(
                "reader_current_not_found",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                s3_prefix=self.s3_prefix,
            )
            raise ReaderStateError("CURRENT pointer not found")
        return current

    def _open_one_reader(self, shard: Any) -> ShardReader:
        """Create local dir and open a single shard reader."""
        local_path = Path(self.local_root) / f"shard={shard.db_id:05d}"
        local_path.mkdir(parents=True, exist_ok=True)

        return self._reader_factory(
            db_url=shard.db_url,
            local_dir=local_path,
            checkpoint_id=shard.checkpoint_id,
        )

    def _emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        if self._metrics is not None:
            self._metrics.emit(event, payload)

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        raise NotImplementedError


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
        current_name: str = "_CURRENT",
        reader_factory: ShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        max_workers: int | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        super().__init__(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=manifest_store,
            current_name=current_name,
            reader_factory=reader_factory,
            slate_env_file=slate_env_file,
            max_workers=max_workers,
            metrics_collector=metrics_collector,
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
            sharding=rb.sharding.strategy.value,
            created_at=rb.created_at,
            manifest_ref=state.manifest_ref,
            key_encoding=rb.key_encoding.value,
            row_count=sum(s.row_count for s in state.router.shards),
        )

    def shard_details(self) -> list[ShardDetail]:
        """Return per-shard metadata from the current manifest."""
        return _shard_details_from_router(self._state.router)

    def route_key(self, key: KeyInput) -> int:
        """Return the shard db_id a key would route to."""
        return self._state.router.route_one(key)

    def shard_for_key(self, key: KeyInput) -> RequiredShardMeta:
        """Return shard metadata for the shard a key routes to."""
        db_id = self._state.router.route_one(key)
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

    def get(self, key: KeyInput) -> bytes | None:
        if self._closed:
            raise ReaderStateError("Reader is closed")

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        state = self._state
        db_id = state.router.route_one(key)
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

    def multi_get(self, keys: Sequence[KeyInput]) -> dict[KeyInput, bytes | None]:
        if self._closed:
            raise ReaderStateError("Reader is closed")

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        key_list = list(keys)
        state = self._state
        grouped = state.router.group_keys(key_list)
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
                    raise SlateDbApiError(
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
        current_ref = current.manifest_ref

        if current_ref == self._state.manifest_ref:
            log_event(
                "reader_refresh_unchanged",
                logger=_logger,
                manifest_ref=current_ref,
            )
            self._emit(MetricEvent.READER_REFRESHED, {"changed": False})
            return False

        manifest = self._manifest_store.load_manifest(current_ref)
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
        manifest = self._manifest_store.load_manifest(current.manifest_ref)
        return self._build_simple_state(current.manifest_ref, manifest)

    def _build_simple_state(
        self, manifest_ref: str, manifest: ParsedManifest
    ) -> _SimpleReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        readers: dict[int, ShardReader] = {}
        try:
            for shard in manifest.shards:
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


# ---------------------------------------------------------------------------
# ConcurrentShardedReader — thread-safe with lock/pool modes
# ---------------------------------------------------------------------------


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
        thread_safety: Literal["lock", "pool"] = "lock",
        pool_checkout_timeout: float = 30.0,
        max_workers: int | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        if thread_safety not in {"lock", "pool"}:
            raise ValueError("thread_safety must be 'lock' or 'pool'")

        super().__init__(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=manifest_store,
            current_name=current_name,
            reader_factory=reader_factory,
            slate_env_file=slate_env_file,
            max_workers=max_workers,
            metrics_collector=metrics_collector,
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
                sharding=rb.sharding.strategy.value,
                created_at=rb.created_at,
                manifest_ref=state.manifest_ref,
                key_encoding=rb.key_encoding.value,
                row_count=sum(s.row_count for s in state.router.shards),
            )

    def shard_details(self) -> list[ShardDetail]:
        """Return per-shard metadata from the current manifest."""
        with self._use_state() as state:
            return _shard_details_from_router(state.router)

    def route_key(self, key: KeyInput) -> int:
        """Return the shard db_id a key would route to."""
        with self._use_state() as state:
            return state.router.route_one(key)

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

    def get(self, key: KeyInput) -> bytes | None:
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
        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        key_list = list(keys)
        with self._use_state() as state:
            grouped = state.router.group_keys(key_list)
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

        current_ref = current.manifest_ref
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

        manifest = self._manifest_store.load_manifest(current_ref)
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
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
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
            num_handles_closed=len(self._state.handles),
        )
        self._emit(MetricEvent.READER_CLOSED, {"num_handles": len(self._state.handles)})

    def _load_initial_state(self) -> _ReaderState:
        current = self._load_current()
        manifest = self._manifest_store.load_manifest(current.manifest_ref)
        return self._build_state(current.manifest_ref, manifest)

    def _build_state(self, manifest_ref: str, manifest: ParsedManifest) -> _ReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        handles: dict[int, _ShardHandle] = {}

        try:
            for shard in manifest.shards:
                if self.thread_safety == "lock":
                    reader = self._open_one_reader(shard)
                    handles[shard.db_id] = _ShardHandle(
                        mode="lock",
                        reader=reader,
                        lock=threading.Lock(),
                    )
                else:
                    pool_size = self.max_workers or 4
                    readers = [self._open_one_reader(shard) for _ in range(pool_size)]
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


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def _shard_details_from_router(router: SnapshotRouter) -> list[ShardDetail]:
    """Extract per-shard metadata from a router's shard list."""
    return [
        ShardDetail(
            db_id=s.db_id,
            row_count=s.row_count,
            min_key=s.min_key,
            max_key=s.max_key,
            db_url=s.db_url,
        )
        for s in router.shards
    ]


def _read_group_simple(
    router: SnapshotRouter,
    reader: ShardReader,
    keys: list[KeyInput],
) -> dict[KeyInput, bytes | None]:
    """Read a group of keys from a single shard reader (no locking)."""
    return {key: reader.get(router.encode_lookup_key(key)) for key in keys}


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
        if handle.lock is None or handle.reader is None:
            raise RuntimeError("lock-mode handle missing lock or reader")
        with handle.lock:
            return handle.reader.get(key)

    if handle.pool is None:
        raise RuntimeError("pool-mode handle missing reader pool")
    with handle.pool.checkout() as reader:
        return reader.get(key)


def _close_simple_state(state: _SimpleReaderState) -> None:
    errors: list[tuple[int, BaseException]] = []
    for db_id, reader in state.readers.items():
        try:
            reader.close()
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
        raise SlateDbApiError(
            f"Failed to close {len(errors)} shard reader(s): "
            f"db_ids={[db_id for db_id, _ in errors]}"
        ) from errors[0][1]


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
        raise SlateDbApiError(
            f"Failed to close {len(errors)} shard handle(s): "
            f"db_ids={[db_id for db_id, _ in errors]}"
        ) from errors[0][1]


def _close_handle(handle: _ShardHandle) -> None:
    if handle.mode == "lock" and handle.reader is not None:
        handle.reader.close()
    if handle.mode == "pool" and handle.pool is not None:
        handle.pool.close()
