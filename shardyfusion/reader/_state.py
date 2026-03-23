"""Lock ordering, state types, handles, and helper functions for readers."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Self

from shardyfusion.errors import (
    DbAdapterError,
    PoolExhaustedError,
)
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_failure,
)
from shardyfusion.routing import SnapshotRouter
from shardyfusion.type_defs import (
    KeyInput,
    ShardReader,
)

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

    def __enter__(self) -> _OrderedLock:
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
    handles: dict[int, _ShardHandle]
    refcount: int = 0
    retired: bool = False


@dataclass(slots=True)
class _ShardHandle:
    mode: str
    reader: ShardReader | None = None
    lock: threading.Lock | None = None
    pool: _ReaderPool | None = None


class _ReaderPool:
    def __init__(
        self, readers: list[ShardReader], *, checkout_timeout: float = 30.0
    ) -> None:
        if not readers:
            raise ValueError("Reader pool requires at least one reader")
        if checkout_timeout <= 0:
            raise ValueError(f"checkout_timeout must be > 0, got {checkout_timeout}")
        self._readers: tuple[ShardReader, ...] = tuple(readers)
        self._checkout_timeout = checkout_timeout
        self._indexes: Queue[int] = Queue(maxsize=len(readers))
        for idx in range(len(readers)):
            self._indexes.put(idx)

    @contextmanager
    def checkout(self) -> Iterator[ShardReader]:
        try:
            idx = self._indexes.get(timeout=self._checkout_timeout)
        except Empty:
            raise PoolExhaustedError(
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
        handle: _ShardHandle | ShardReader,
        release_fn: Callable[[], None] | None = None,
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

    def __del__(self) -> None:
        if not self._released:
            log_failure(
                "shard_reader_handle_not_closed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
            )
            self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


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
        raise DbAdapterError(
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
        raise DbAdapterError(
            f"Failed to close {len(errors)} shard handle(s): "
            f"db_ids={[db_id for db_id, _ in errors]}"
        ) from errors[0][1]


def _close_handle(handle: _ShardHandle) -> None:
    if handle.mode == "lock" and handle.reader is not None:
        handle.reader.close()
    if handle.mode == "pool" and handle.pool is not None:
        handle.pool.close()
