"""Async sharded reader for asyncio-based services (FastAPI, aiohttp, etc.)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Self

from shardyfusion._rate_limiter import RateLimiter
from shardyfusion._slatedb_symbols import get_slatedb_reader_class
from shardyfusion.async_manifest_store import (
    AsyncManifestStore,
    AsyncS3ManifestStore,
)
from shardyfusion.credentials import (
    CredentialProvider,
    resolve_env_file,
)
from shardyfusion.errors import DbAdapterError, ManifestParseError, ReaderStateError
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)
from shardyfusion.manifest import ParsedManifest, RequiredShardMeta
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.reader._types import (
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
    Manifest,
    S3ConnectionOptions,
)

_logger = get_logger(__name__)


def _is_unknown_categorical_token_error(exc: Exception) -> bool:
    from shardyfusion.cel import UnknownRoutingTokenError

    return isinstance(exc, UnknownRoutingTokenError)


# ---------------------------------------------------------------------------
# Null async shard reader for empty shards (db_url=None)
# ---------------------------------------------------------------------------


class _NullAsyncShardReader:
    """No-op async reader for empty shards — always returns ``None``."""

    async def get(self, key: bytes) -> bytes | None:
        return None

    async def close(self) -> None:
        pass


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


class _AsyncReaderState:
    """Internal state for AsyncShardedReader with async cleanup support."""

    __slots__ = (
        "manifest_ref",
        "router",
        "readers",
        "borrow_count",
        "retired",
        "_closed",
    )

    def __init__(
        self,
        manifest_ref: str,
        router: SnapshotRouter,
        readers: dict[int, AsyncShardReader],
        borrow_count: int = 0,
        retired: bool = False,
    ) -> None:
        self.manifest_ref = manifest_ref
        self.router = router
        self.readers = readers
        self.borrow_count = borrow_count
        self.retired = retired
        self._closed = False

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            # Cannot await aclose here, but we can log a warning
            # if we had a logger, but it might be gone.
            # ShardReaderHandle.__del__ and AsyncShardedReader.__del__
            # are the primary cleanup paths that try to schedule aclose().
            pass

    async def aclose(self) -> None:
        """Close all shard readers, logging per-reader errors."""
        if self._closed:
            return
        self._closed = True
        errors: list[tuple[int, BaseException]] = []
        for db_id, reader in self.readers.items():
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
                    manifest_ref=self.manifest_ref,
                )
        if errors:
            raise DbAdapterError(
                f"Failed to close {len(errors)} shard reader(s): "
                f"db_ids={[db_id for db_id, _ in errors]}"
            ) from errors[0][1]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncSlateDbReaderFactory:
    """Default async factory using ``SlateDBReader.open_async()``."""

    env_file: str | None = None
    credential_provider: CredentialProvider | None = None

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncShardReader:
        try:
            reader_cls = get_slatedb_reader_class()
        except DbAdapterError as exc:  # pragma: no cover - runtime dependent
            raise DbAdapterError(
                "slatedb package is required for reading shards"
            ) from exc

        with resolve_env_file(self.env_file, self.credential_provider) as env_path:
            kwargs: dict[str, Any] = {"url": db_url, "checkpoint_id": checkpoint_id}
            if env_path is not None:
                kwargs["env_file"] = env_path
            inner = await reader_cls.open_async(str(local_dir), **kwargs)
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

    def __del__(self) -> None:
        if not self._released:
            log_failure(
                "async_shard_reader_handle_not_closed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
            )
            self.close()

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
        current_pointer_key: str = "_CURRENT",
        reader_factory: AsyncShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        max_concurrency: int | None = None,
        max_fallback_attempts: int = 3,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.s3_prefix = s3_prefix
        self.local_root = local_root
        self._metrics = metrics_collector
        self._max_concurrency = max_concurrency
        self._rate_limiter = rate_limiter
        self._max_fallback_attempts = max_fallback_attempts
        self._refresh_lock = asyncio.Lock()
        self._closed = False
        self._init_time = time.monotonic()

        # State is set by open(), not __init__.
        self._state: _AsyncReaderState | None = None

        # Resolve manifest store
        if manifest_store is not None:
            self._manifest_store = manifest_store
        else:
            credentials = credential_provider.resolve() if credential_provider else None
            from ..storage import AsyncObstoreBackend, create_s3_store, parse_s3_url

            bucket, _ = parse_s3_url(s3_prefix)
            store = create_s3_store(
                bucket=bucket,
                credentials=credentials,
                connection_options=s3_connection_options,
            )
            backend = AsyncObstoreBackend(store)
            self._manifest_store = AsyncS3ManifestStore(
                backend,
                s3_prefix,
                current_pointer_key=current_pointer_key,
            )

        # Resolve reader factory
        if reader_factory is not None:
            self._reader_factory = reader_factory
        else:
            self._reader_factory = AsyncSlateDbReaderFactory(
                env_file=slate_env_file,
                credential_provider=credential_provider,
            )

    @classmethod
    async def open(
        cls,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: AsyncManifestStore | None = None,
        current_pointer_key: str = "_CURRENT",
        reader_factory: AsyncShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        max_concurrency: int | None = None,
        max_fallback_attempts: int = 3,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> AsyncShardedReader:
        """Create and initialize an async sharded reader."""
        instance = cls(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=manifest_store,
            current_pointer_key=current_pointer_key,
            reader_factory=reader_factory,
            slate_env_file=slate_env_file,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
            max_concurrency=max_concurrency,
            max_fallback_attempts=max_fallback_attempts,
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
            hash_algorithm=rb.sharding.hash_algorithm,
            row_count=sum(s.row_count for s in state.router.shards),
        )

    def shard_details(self) -> list[ShardDetail]:
        return _shard_details_from_router(self._require_state().router)

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
        if self._state is None:
            return ReaderHealth(
                status="unhealthy",
                manifest_ref="",
                manifest_age=timedelta(0),
                num_shards=0,
                is_closed=False,
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
        return self._require_state().router.route(key, routing_context=routing_context)

    def shard_for_key(
        self,
        key: KeyInput,
        *,
        routing_context: dict[str, object] | None = None,
    ) -> RequiredShardMeta:
        state = self._require_state()
        db_id = state.router.route(key, routing_context=routing_context)
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

    async def get(
        self,
        key: KeyInput,
        *,
        routing_context: dict[str, object] | None = None,
    ) -> bytes | None:
        state = self._require_state()
        state.borrow_count += 1

        try:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire_async(1)

            mc = self._metrics
            t0 = time.perf_counter() if mc is not None else 0.0

            try:
                db_id = state.router.route(key, routing_context=routing_context)
            except ValueError as exc:
                if _is_unknown_categorical_token_error(exc):
                    result = None
                    if mc is not None:
                        mc.emit(
                            MetricEvent.READER_GET,
                            {
                                "duration_ms": int((time.perf_counter() - t0) * 1000),
                                "found": False,
                            },
                        )
                    return result
                raise
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
        finally:
            self._release_state(state)

    async def multi_get(
        self,
        keys: Sequence[KeyInput],
        *,
        routing_context: dict[str, object] | None = None,
    ) -> dict[KeyInput, bytes | None]:
        state = self._require_state()
        state.borrow_count += 1

        try:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire_async(len(keys))

            mc = self._metrics
            t0 = time.perf_counter() if mc is not None else 0.0

            key_list = list(keys)

            grouped, missing = state.router.group_keys_allow_missing(
                key_list,
                routing_context=routing_context,
            )
            results: dict[KeyInput, bytes | None] = {key: None for key in missing}

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
        finally:
            self._release_state(state)

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

        current_ref = current.ref

        if current_ref == state.manifest_ref:
            log_event(
                "reader_refresh_unchanged",
                logger=_logger,
                manifest_ref=current_ref,
            )
            self._emit(MetricEvent.READER_REFRESHED, {"changed": False})
            return False

        try:
            manifest = await self._manifest_store.load_manifest(current_ref)
        except ManifestParseError:
            log_failure(
                "reader_refresh_malformed_manifest",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                manifest_ref=current_ref,
            )
            self._emit(MetricEvent.READER_REFRESHED, {"changed": False})
            return False

        new_state = await self._build_state(current_ref, manifest)
        old_state = state
        self._state = new_state
        old_state.retired = True
        if old_state.borrow_count == 0:
            await old_state.aclose()

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
                await state.aclose()

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

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            # Not closed explicitly. Mark state as retired and try to schedule
            # cleanup if an event loop is running.
            self._closed = True
            state = self._state
            if state is not None:
                state.retired = True
                if state.borrow_count == 0:
                    try:
                        loop = asyncio.get_running_loop()
                        if not loop.is_closed():
                            loop.create_task(state.aclose())
                    except Exception:
                        pass

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
        try:
            manifest = await self._manifest_store.load_manifest(current.ref)
            return await self._build_state(current.ref, manifest)
        except ManifestParseError:
            if self._max_fallback_attempts == 0:
                raise
            return await self._fallback_to_previous(failed_ref=current.ref)

    async def _fallback_to_previous(self, *, failed_ref: str) -> _AsyncReaderState:
        """Walk backward through manifest history to find a valid manifest."""
        log_failure(
            "reader_cold_start_fallback",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            failed_ref=failed_ref,
            max_attempts=self._max_fallback_attempts,
        )
        refs = await self._manifest_store.list_manifests(
            limit=self._max_fallback_attempts + 1
        )
        for ref in refs:
            if ref.ref == failed_ref:
                continue
            try:
                manifest = await self._manifest_store.load_manifest(ref.ref)
                log_event(
                    "reader_fallback_succeeded",
                    logger=_logger,
                    fallback_ref=ref.ref,
                    skipped_ref=failed_ref,
                )
                return await self._build_state(ref.ref, manifest)
            except ManifestParseError:
                log_failure(
                    "reader_fallback_skip_malformed",
                    severity=FailureSeverity.ERROR,
                    logger=_logger,
                    manifest_ref=ref.ref,
                )
                continue
        raise ReaderStateError("No valid manifest found after fallback")

    async def _build_state(
        self, manifest_ref: str, manifest: ParsedManifest
    ) -> _AsyncReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        readers: dict[int, AsyncShardReader] = {}
        try:
            for shard in router.shards:
                if shard.db_url is None:
                    readers[shard.db_id] = _NullAsyncShardReader()
                else:
                    local_path = Path(self.local_root) / f"shard={shard.db_id:05d}"
                    local_path.mkdir(parents=True, exist_ok=True)
                    readers[shard.db_id] = await self._reader_factory(
                        db_url=shard.db_url,
                        local_dir=local_path,
                        checkpoint_id=shard.checkpoint_id,
                        manifest=manifest,
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
            except RuntimeError:
                # No running event loop — cannot schedule cleanup.
                # This can happen when a handle is released from a
                # non-async context (e.g. __del__ or a sync callback).
                return
            try:
                loop.create_task(state.aclose())
            except RuntimeError as exc:
                log_failure(
                    "async_reader_deferred_cleanup_failed",
                    severity=FailureSeverity.ERROR,
                    logger=_logger,
                    error=exc,
                    manifest_ref=state.manifest_ref,
                )

    def _emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        if self._metrics is not None:
            self._metrics.emit(event, payload)
