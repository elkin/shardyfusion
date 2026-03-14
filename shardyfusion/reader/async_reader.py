"""Async sharded reader for asyncio-based services (FastAPI, aiohttp, etc.)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from shardyfusion._rate_limiter import RateLimiter
from shardyfusion.async_manifest_store import (
    AsyncManifestStore,
    AsyncS3ManifestStore,
)
from shardyfusion.errors import ReaderStateError, SlateDbApiError
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)
from shardyfusion.manifest import ParsedManifest, RequiredShardMeta
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.reader.reader import (
    ReaderHealth,
    ShardDetail,
    SnapshotInfo,
    _shard_details_from_router,
)
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.type_defs import (
    AsyncShardReader,
    AsyncShardReaderFactory,
    KeyInput,
    S3ClientConfig,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _SlateDbAsyncShardReader:
    """Adapts SlateDB's ``get_async``/``close_async`` to ``AsyncShardReader``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def get(self, key: bytes) -> bytes | None:
        return await self._inner.get_async(key)

    async def close(self) -> None:
        await self._inner.close_async()


@dataclass(slots=True)
class _AsyncReaderState:
    manifest_ref: str
    router: SnapshotRouter
    readers: dict[int, AsyncShardReader]
    borrow_count: int = 0
    retired: bool = False


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncSlateDbReaderFactory:
    """Default async factory using ``SlateDBReader.open_async()``."""

    env_file: str | None = None

    async def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> AsyncShardReader:
        try:
            from slatedb import SlateDBReader
        except ImportError as exc:  # pragma: no cover - runtime dependent
            raise SlateDbApiError(
                "slatedb package is required for reading shards"
            ) from exc

        kwargs: dict[str, Any] = {"url": db_url, "checkpoint_id": checkpoint_id}
        if self.env_file is not None:
            kwargs["env_file"] = self.env_file

        inner = await SlateDBReader.open_async(str(local_dir), **kwargs)
        return _SlateDbAsyncShardReader(inner)


class AsyncShardReaderHandle:
    """Borrowed async reference to a single shard's reader.

    Supports ``async with`` and explicit ``close()``.  Closing releases the
    borrow (decrements refcount) but does **not** close the underlying shard.
    """

    def __init__(
        self,
        reader: AsyncShardReader,
        release_fn: Any | None = None,
    ) -> None:
        self._reader = reader
        self._release_fn = release_fn
        self._released = False

    async def get(self, key: bytes) -> bytes | None:
        return await self._reader.get(key)

    async def multi_get(self, keys: Sequence[bytes]) -> dict[bytes, bytes | None]:
        results: dict[bytes, bytes | None] = {}
        for key in keys:
            results[key] = await self._reader.get(key)
        return results

    def close(self) -> None:
        """Release the borrow.  Safe to call multiple times."""
        if self._released:
            return
        self._released = True
        if self._release_fn is not None:
            self._release_fn()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# AsyncShardedReader
# ---------------------------------------------------------------------------


class AsyncShardedReader:
    """Async sharded reader for single-threaded asyncio services.

    Uses ``asyncio.Lock`` for refresh serialization and native async
    SlateDB calls on the read path.
    """

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: AsyncManifestStore | None = None,
        current_name: str = "_CURRENT",
        reader_factory: AsyncShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        s3_client_config: S3ClientConfig | None = None,
        max_concurrency: int | None = None,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.s3_prefix = s3_prefix
        self.local_root = local_root
        self._metrics = metrics_collector
        self._max_concurrency = max_concurrency
        self._rate_limiter = rate_limiter
        self._refresh_lock = asyncio.Lock()
        self._closed = False
        self._init_time = time.monotonic()

        # State is set by open(), not __init__.
        self._state: _AsyncReaderState | None = None

        # Resolve manifest store
        if manifest_store is not None:
            self._manifest_store = manifest_store
        else:
            self._manifest_store = AsyncS3ManifestStore(
                s3_prefix,
                current_name=current_name,
                s3_client_config=s3_client_config,
                metrics_collector=metrics_collector,
            )

        # Resolve reader factory
        if reader_factory is not None:
            self._reader_factory = reader_factory
        else:
            self._reader_factory = AsyncSlateDbReaderFactory(env_file=slate_env_file)

    @classmethod
    async def open(
        cls,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: AsyncManifestStore | None = None,
        current_name: str = "_CURRENT",
        reader_factory: AsyncShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        s3_client_config: S3ClientConfig | None = None,
        max_concurrency: int | None = None,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> AsyncShardedReader:
        """Create and initialize an async sharded reader."""
        instance = cls(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=manifest_store,
            current_name=current_name,
            reader_factory=reader_factory,
            slate_env_file=slate_env_file,
            s3_client_config=s3_client_config,
            max_concurrency=max_concurrency,
            metrics_collector=metrics_collector,
            rate_limiter=rate_limiter,
        )
        state = await instance._load_initial_state()
        instance._state = state

        log_event(
            "reader_initialized",
            logger=_logger,
            s3_prefix=s3_prefix,
            num_shards=len(state.readers),
            manifest_ref=state.manifest_ref,
        )
        instance._emit(MetricEvent.READER_INITIALIZED, {})
        return instance

    # -- Properties / metadata (no I/O, synchronous) ----------------------

    def _require_state(self) -> _AsyncReaderState:
        if self._closed:
            raise ReaderStateError("Reader is closed")
        if self._state is None:
            raise ReaderStateError("Reader not yet initialized (use open())")
        return self._state

    @property
    def key_encoding(self) -> KeyEncoding:
        return KeyEncoding.from_value(self._require_state().router.key_encoding)

    def snapshot_info(self) -> SnapshotInfo:
        state = self._require_state()
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
        return _shard_details_from_router(self._require_state().router)

    def health(self, *, staleness_threshold_s: float | None = None) -> ReaderHealth:
        """Return a diagnostic snapshot of reader state."""
        if self._closed:
            return ReaderHealth(
                status="unhealthy",
                manifest_ref="",
                manifest_age_seconds=0.0,
                num_shards=0,
                is_closed=True,
                circuit_breaker_state=None,
            )
        if self._state is None:
            return ReaderHealth(
                status="unhealthy",
                manifest_ref="",
                manifest_age_seconds=0.0,
                num_shards=0,
                is_closed=False,
                circuit_breaker_state=None,
            )
        age = time.monotonic() - self._init_time
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if staleness_threshold_s is not None and age > staleness_threshold_s:
            status = "degraded"
        return ReaderHealth(
            status=status,
            manifest_ref=self._state.manifest_ref,
            manifest_age_seconds=round(age, 2),
            num_shards=len(self._state.readers),
            is_closed=False,
            circuit_breaker_state=None,
        )

    def route_key(self, key: KeyInput) -> int:
        return self._require_state().router.route_one(key)

    def shard_for_key(self, key: KeyInput) -> RequiredShardMeta:
        state = self._require_state()
        db_id = state.router.route_one(key)
        return state.router.shards[db_id]

    def shards_for_keys(
        self, keys: Sequence[KeyInput]
    ) -> dict[KeyInput, RequiredShardMeta]:
        state = self._require_state()
        router = state.router
        return {key: router.shards[router.route_one(key)] for key in keys}

    # -- Borrow handles (synchronous, no I/O) ----------------------------

    def reader_for_key(self, key: KeyInput) -> AsyncShardReaderHandle:
        state = self._require_state()
        state.borrow_count += 1
        db_id = state.router.route_one(key)
        return AsyncShardReaderHandle(
            state.readers[db_id],
            release_fn=lambda: self._release_state(state),
        )

    def readers_for_keys(
        self, keys: Sequence[KeyInput]
    ) -> dict[KeyInput, AsyncShardReaderHandle]:
        state = self._require_state()
        unique_keys = dict.fromkeys(keys)
        state.borrow_count += len(unique_keys)
        return {
            key: AsyncShardReaderHandle(
                state.readers[state.router.route_one(key)],
                release_fn=lambda: self._release_state(state),
            )
            for key in unique_keys
        }

    # -- Async read operations -------------------------------------------

    async def get(self, key: KeyInput) -> bytes | None:
        state = self._require_state()

        if self._rate_limiter is not None:
            await _async_acquire(self._rate_limiter, 1)

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        db_id = state.router.route_one(key)
        key_bytes = state.router.encode_lookup_key(key)
        result = await state.readers[db_id].get(key_bytes)

        if mc is not None:
            mc.emit(
                MetricEvent.READER_GET,
                {
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                    "found": result is not None,
                },
            )
        return result

    async def multi_get(self, keys: Sequence[KeyInput]) -> dict[KeyInput, bytes | None]:
        state = self._require_state()

        if self._rate_limiter is not None:
            await _async_acquire(self._rate_limiter, len(keys))

        mc = self._metrics
        t0 = time.perf_counter() if mc is not None else 0.0

        key_list = list(keys)
        grouped = state.router.group_keys(key_list)
        results: dict[KeyInput, bytes | None] = {}

        semaphore = (
            asyncio.Semaphore(self._max_concurrency)
            if self._max_concurrency is not None
            else None
        )

        async def _read_group(
            db_id: int, shard_keys: list[KeyInput]
        ) -> dict[KeyInput, bytes | None]:
            reader = state.readers[db_id]
            group_results: dict[KeyInput, bytes | None] = {}
            for key in shard_keys:
                key_bytes = state.router.encode_lookup_key(key)
                if semaphore is not None:
                    async with semaphore:
                        group_results[key] = await reader.get(key_bytes)
                else:
                    group_results[key] = await reader.get(key_bytes)
            return group_results

        async with asyncio.TaskGroup() as tg:
            tasks = {
                db_id: tg.create_task(_read_group(db_id, shard_keys))
                for db_id, shard_keys in grouped.items()
            }

        for task in tasks.values():
            results.update(task.result())

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

    # -- Lifecycle -------------------------------------------------------

    async def refresh(self) -> bool:
        self._require_state()

        async with self._refresh_lock:
            return await self._do_refresh()

    async def _do_refresh(self) -> bool:
        state = self._require_state()
        current = await self._manifest_store.load_current()
        if current is None:
            log_failure(
                "reader_refresh_current_not_found",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                s3_prefix=self.s3_prefix,
            )
            raise ReaderStateError("CURRENT pointer not found during refresh")

        current_ref = current.manifest_ref

        if current_ref == state.manifest_ref:
            log_event(
                "reader_refresh_unchanged",
                logger=_logger,
                manifest_ref=current_ref,
            )
            self._emit(MetricEvent.READER_REFRESHED, {"changed": False})
            return False

        manifest = await self._manifest_store.load_manifest(current_ref)
        new_state = await self._build_state(current_ref, manifest)
        old_state = state
        self._state = new_state
        old_state.retired = True
        if old_state.borrow_count == 0:
            await _close_async_state(old_state)

        log_event(
            "reader_refreshed",
            logger=_logger,
            old_manifest_ref=old_state.manifest_ref,
            new_manifest_ref=current_ref,
        )
        self._emit(MetricEvent.READER_REFRESHED, {"changed": True})
        return True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        state = self._state
        if state is not None:
            num_readers = len(state.readers)
            state.retired = True
            if state.borrow_count == 0:
                await _close_async_state(state)

            log_event(
                "reader_closed",
                logger=_logger,
                s3_prefix=self.s3_prefix,
                num_handles_closed=num_readers,
            )
            self._emit(MetricEvent.READER_CLOSED, {"num_handles": num_readers})

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # -- Private helpers -------------------------------------------------

    async def _load_initial_state(self) -> _AsyncReaderState:
        current = await self._manifest_store.load_current()
        if current is None:
            log_failure(
                "reader_current_not_found",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                s3_prefix=self.s3_prefix,
            )
            raise ReaderStateError("CURRENT pointer not found")
        manifest = await self._manifest_store.load_manifest(current.manifest_ref)
        return await self._build_state(current.manifest_ref, manifest)

    async def _build_state(
        self, manifest_ref: str, manifest: ParsedManifest
    ) -> _AsyncReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        readers: dict[int, AsyncShardReader] = {}
        try:
            for shard in manifest.shards:
                local_path = Path(self.local_root) / f"shard={shard.db_id:05d}"
                local_path.mkdir(parents=True, exist_ok=True)
                readers[shard.db_id] = await self._reader_factory(
                    db_url=shard.db_url,
                    local_dir=local_path,
                    checkpoint_id=shard.checkpoint_id,
                )
        except Exception:
            for reader in readers.values():
                try:
                    await reader.close()
                except Exception:
                    pass
            raise
        return _AsyncReaderState(
            manifest_ref=manifest_ref, router=router, readers=readers
        )

    def _release_state(self, state: _AsyncReaderState) -> None:
        state.borrow_count -= 1
        if state.retired and state.borrow_count == 0:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_close_async_state(state))
            except RuntimeError:
                pass

    def _emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        if self._metrics is not None:
            self._metrics.emit(event, payload)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def _async_acquire(limiter: RateLimiter, tokens: int) -> None:
    """Acquire tokens from a rate limiter, using async sleep if available.

    If the limiter has an ``acquire_async`` method (like ``TokenBucket``),
    use it directly.  Otherwise, fall back to a ``try_acquire`` + ``asyncio.sleep``
    loop — still non-blocking for the event loop.
    """
    acquire_async = getattr(limiter, "acquire_async", None)
    if acquire_async is not None:
        await acquire_async(tokens)
        return

    # Fallback: try_acquire is guaranteed non-blocking (pure arithmetic).
    while True:
        result = limiter.try_acquire(tokens)
        if result:
            return
        await asyncio.sleep(result.deficit)


async def _close_async_state(state: _AsyncReaderState) -> None:
    """Close all shard readers in an async state, logging errors."""
    errors: list[tuple[int, BaseException]] = []
    for db_id, reader in state.readers.items():
        try:
            await reader.close()
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
