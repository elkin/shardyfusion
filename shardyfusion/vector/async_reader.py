"""Async sharded vector reader — asyncio-based read-side router."""

from __future__ import annotations

import asyncio
import io
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._rate_limiter import RateLimiter
from ..async_manifest_store import AsyncManifestStore, AsyncS3ManifestStore
from ..credentials import CredentialProvider
from ..errors import ReaderStateError
from ..logging import get_logger, log_event
from ..manifest import ManifestRef, ParsedManifest, RequiredShardMeta
from ..metrics._events import MetricEvent
from ..metrics._protocol import MetricsCollector
from ..storage import ObstoreBackend, create_s3_store, parse_s3_url
from ..type_defs import S3ConnectionOptions
from ._merge import merge_results
from .config import VectorIndexConfig
from .reader import VectorReaderHealth
from .sharding import route_vector_to_shards
from .types import (
    AsyncVectorShardReader,
    AsyncVectorShardReaderFactory,
    DistanceMetric,
    SearchResult,
    VectorSearchResponse,
    VectorShardDetail,
    VectorShardingStrategy,
    VectorSnapshotInfo,
)

_logger = get_logger(__name__)


@dataclass(slots=True)
class _CachedAsyncShardReader:
    """Cache entry that defers close until in-flight searches release it."""

    reader: AsyncVectorShardReader
    borrow_count: int = 0
    retired: bool = False


class AsyncShardedVectorReader:
    """Async read-side router that loads a vector manifest and fans out searches."""

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        reader_factory: AsyncVectorShardReaderFactory | None = None,
        manifest_store: AsyncManifestStore | None = None,
        max_concurrency: int | None = None,
        max_fallback_attempts: int = 3,
        preload_shards: bool = False,
        max_cached_shards: int | None = None,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
    ) -> None:
        self._s3_prefix = s3_prefix
        self._local_root = Path(local_root)
        self._max_concurrency = max_concurrency
        self._max_fallback_attempts = max_fallback_attempts
        self._preload_shards = preload_shards
        self._max_cached_shards = max_cached_shards
        self._mc = metrics_collector
        self._rate_limiter = rate_limiter
        self._credential_provider = credential_provider
        self._s3_connection_options = s3_connection_options
        self._closed = False
        self._refresh_lock = asyncio.Lock()

        if manifest_store is not None:
            self._store = manifest_store
        else:
            from ..storage import AsyncObstoreBackend

            credentials = credential_provider.resolve() if credential_provider else None
            bucket, _ = parse_s3_url(s3_prefix)
            store = create_s3_store(
                bucket=bucket,
                credentials=credentials,
                connection_options=s3_connection_options,
            )
            backend = AsyncObstoreBackend(store)
            self._store = AsyncS3ManifestStore(
                backend,
                s3_prefix,
            )

        if reader_factory is not None:
            self._reader_factory = reader_factory
        else:
            from .adapters.lancedb_adapter import AsyncLanceDbReaderFactory

            self._reader_factory = AsyncLanceDbReaderFactory(
                s3_connection_options=s3_connection_options,
                credential_provider=credential_provider,
            )

        # Sync backend used within threadpool to load centroids/hyperplanes.
        # This occurs only during initialization/refresh.
        credentials = credential_provider.resolve() if credential_provider else None
        bucket, _ = parse_s3_url(s3_prefix)
        store = create_s3_store(
            bucket=bucket,
            credentials=credentials,
            connection_options=s3_connection_options,
        )
        self._backend = ObstoreBackend(store)

        # Shard reader cache (lazy loading)
        self._shard_readers: OrderedDict[int, _CachedAsyncShardReader] = OrderedDict()
        self._shard_locks: dict[int, asyncio.Lock] = {}
        self._cache_generation = 0
        self._cache_lock = asyncio.Lock()  # protects _shard_readers structure

        # State loaded from manifest
        self._manifest_ref: ManifestRef | None = None
        self._manifest: ParsedManifest | None = None
        self._index_config: VectorIndexConfig | None = None
        self._sharding_strategy: VectorShardingStrategy | None = None
        self._num_dbs: int = 0
        self._num_probes: int = 1
        self._metric: DistanceMetric = DistanceMetric.COSINE
        self._centroids: np.ndarray | None = None
        self._hyperplanes: np.ndarray | None = None
        self._cel_expr: str | None = None
        self._cel_columns: dict[str, str] | None = None
        self._routing_values: list[int | str | bytes] | None = None
        self._shard_meta: dict[int, RequiredShardMeta] = {}

    @classmethod
    async def open(
        cls,
        *,
        s3_prefix: str,
        local_root: str,
        **kwargs: Any,
    ) -> AsyncShardedVectorReader:
        """Create and initialize an async sharded vector reader."""
        instance = cls(s3_prefix=s3_prefix, local_root=local_root, **kwargs)
        await instance._load_initial_manifest()

        if instance._mc is not None:
            instance._mc.emit(
                MetricEvent.VECTOR_READER_INITIALIZED,
                {"s3_prefix": s3_prefix, "num_shards": instance._num_dbs},
            )
        return instance

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        shard_ids: list[int] | None = None,
        num_probes: int | None = None,
        routing_context: dict[str, Any] | None = None,
    ) -> VectorSearchResponse:
        """Search across shards and merge results."""
        self._check_open()
        started = time.perf_counter()

        probes = num_probes if num_probes is not None else self._num_probes
        target_shards = route_vector_to_shards(
            query,
            strategy=self._sharding_strategy or VectorShardingStrategy.EXPLICIT,
            num_dbs=self._num_dbs,
            num_probes=probes,
            metric=self._metric,
            centroids=self._centroids,
            hyperplanes=self._hyperplanes,
            shard_ids=shard_ids,
            routing_context=routing_context,
            cel_expr=self._cel_expr,
            cel_columns=self._cel_columns,
            routing_values=self._routing_values,
        )

        target_shards = [s for s in target_shards if s in self._shard_meta]

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire_async()

        # Fan out to shards
        per_shard_results: list[list[SearchResult]] = []
        if len(target_shards) > 0:
            semaphore = (
                asyncio.Semaphore(self._max_concurrency)
                if self._max_concurrency is not None
                else None
            )

            async def _search_shard_bounded(sid: int) -> list[SearchResult]:
                if semaphore is not None:
                    async with semaphore:
                        return await self._search_shard(sid, query, top_k)
                return await self._search_shard(sid, query, top_k)

            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(_search_shard_bounded(sid)) for sid in target_shards
                ]

            for task in tasks:
                per_shard_results.append(task.result())

        merged = merge_results(per_shard_results, top_k, self._metric)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if self._mc is not None:
            self._mc.emit(
                MetricEvent.VECTOR_SEARCH,
                {
                    "top_k": top_k,
                    "num_shards_queried": len(target_shards),
                    "num_results": len(merged),
                    "elapsed_ms": elapsed_ms,
                },
            )

        return VectorSearchResponse(
            results=merged,
            num_shards_queried=len(target_shards),
            latency_ms=elapsed_ms,
        )

    async def batch_search(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[VectorSearchResponse]:
        """Search multiple queries sequentially. (Consider asyncio.gather for parallel)"""
        # Doing it sequentially matches the sync version semantics, but since it's async we
        # could dispatch concurrently if we wanted. For parity, we yield back to the loop.
        results = []
        for q in queries:
            results.append(await self.search(q, top_k, **kwargs))
        return results

    # ------------------------------------------------------------------
    # Shard inspection
    # ------------------------------------------------------------------

    def route_vector(
        self,
        query: np.ndarray,
        *,
        num_probes: int | None = None,
        routing_context: dict[str, Any] | None = None,
    ) -> list[int]:
        self._check_open()
        probes = num_probes if num_probes is not None else self._num_probes
        return route_vector_to_shards(
            query,
            strategy=self._sharding_strategy or VectorShardingStrategy.EXPLICIT,
            num_dbs=self._num_dbs,
            num_probes=probes,
            metric=self._metric,
            centroids=self._centroids,
            hyperplanes=self._hyperplanes,
            routing_context=routing_context,
            cel_expr=self._cel_expr,
            cel_columns=self._cel_columns,
            routing_values=self._routing_values,
        )

    def shard_for_id(self, shard_id: int) -> VectorShardDetail:
        self._check_open()
        meta = self._shard_meta.get(shard_id)
        if meta is None:
            return VectorShardDetail(
                db_id=shard_id, db_url=None, vector_count=0, checkpoint_id=None
            )
        return VectorShardDetail(
            db_id=meta.db_id,
            db_url=meta.db_url,
            vector_count=meta.row_count,
            checkpoint_id=meta.checkpoint_id,
        )

    def shard_details(self) -> list[VectorShardDetail]:
        self._check_open()
        details: list[VectorShardDetail] = []
        for db_id in range(self._num_dbs):
            details.append(self.shard_for_id(db_id))
        return details

    def snapshot_info(self) -> VectorSnapshotInfo:
        self._check_open()
        total = sum(m.row_count for m in self._shard_meta.values())
        return VectorSnapshotInfo(
            run_id=self._manifest_ref.run_id if self._manifest_ref else "",
            num_dbs=self._num_dbs,
            dim=self._index_config.dim if self._index_config else 0,
            metric=self._metric,
            sharding=self._sharding_strategy or VectorShardingStrategy.EXPLICIT,
            manifest_ref=self._manifest_ref.ref if self._manifest_ref else "",
            total_vectors=total,
        )

    def health(
        self,
        *,
        staleness_threshold: timedelta | None = None,
    ) -> VectorReaderHealth:
        if self._closed:
            return VectorReaderHealth(
                status="unhealthy",
                manifest_ref=None,
                manifest_age_seconds=None,
                num_shards=0,
                is_closed=True,
            )

        ref = self._manifest_ref
        age: float | None = None
        if ref is not None:
            age = (datetime.now(UTC) - ref.published_at).total_seconds()

        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if staleness_threshold is not None and age is not None:
            if age > staleness_threshold.total_seconds():
                status = "degraded"

        return VectorReaderHealth(
            status=status,
            manifest_ref=ref.ref if ref else None,
            manifest_age_seconds=age,
            num_shards=self._num_dbs,
            is_closed=False,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def refresh(self) -> bool:
        self._check_open()
        async with self._refresh_lock:
            try:
                new_ref = await self._store.load_current()
            except Exception:
                return False

            if new_ref is None:
                return False
            if self._manifest_ref is not None and new_ref.ref == self._manifest_ref.ref:
                return False

            try:
                manifest = await self._store.load_manifest(new_ref.ref)
            except Exception:
                return False

            try:
                await self._apply_manifest(new_ref, manifest)
            except Exception:
                log_event(
                    "async_vector_reader_refresh_apply_failed",
                    logger=_logger,
                    manifest_ref=new_ref.ref,
                )
                return False

            async with self._cache_lock:
                old_readers = list(self._shard_readers.values())
                self._cache_generation += 1
                self._shard_readers = OrderedDict()
                self._shard_locks = {}

            for cache_entry in old_readers:
                self._retire_reader(cache_entry)

            if self._mc is not None:
                self._mc.emit(
                    MetricEvent.VECTOR_READER_REFRESHED,
                    {"manifest_ref": new_ref.ref},
                )
            return True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._cache_lock:
            readers_to_close = list(self._shard_readers.values())
            self._cache_generation += 1
            self._shard_readers.clear()
            self._shard_locks = {}
        for cache_entry in readers_to_close:
            self._retire_reader(cache_entry)

        if self._mc is not None:
            self._mc.emit(MetricEvent.VECTOR_READER_CLOSED, {})

    async def __aenter__(self) -> AsyncShardedVectorReader:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self._closed = True
            try:
                loop = asyncio.get_running_loop()
                if not loop.is_closed():
                    for cache_entry in self._shard_readers.values():
                        cache_entry.retired = True
                        if cache_entry.borrow_count == 0:
                            loop.create_task(cache_entry.reader.close())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ReaderStateError("Reader is closed")

    async def _load_initial_manifest(self) -> None:
        from ..errors import ManifestParseError

        ref = await self._store.load_current()
        if ref is None:
            raise ReaderStateError("No CURRENT pointer found")

        try:
            manifest = await self._store.load_manifest(ref.ref)
        except ManifestParseError:
            if self._max_fallback_attempts <= 0:
                raise
            all_refs = await self._store.list_manifests()
            manifest = None
            attempts_left = self._max_fallback_attempts
            for fallback_ref in all_refs:
                if fallback_ref.ref == ref.ref:
                    continue
                if attempts_left <= 0:
                    break
                attempts_left -= 1
                try:
                    manifest = await self._store.load_manifest(fallback_ref.ref)
                    ref = fallback_ref
                    break
                except ManifestParseError:
                    continue
            if manifest is None:
                raise

        await self._apply_manifest(ref, manifest)

        if self._preload_shards:
            # We don't parallelize preloading to avoid S3 / memory spikes during startup
            for db_id in self._shard_meta:
                cache_entry = await self._get_or_load_reader(db_id)
                if cache_entry is not None:
                    self._release_reader(cache_entry)

    async def _apply_manifest(
        self,
        ref: ManifestRef,
        manifest: ParsedManifest,
    ) -> None:
        vector_meta = manifest.custom.get("vector", {})
        if not isinstance(vector_meta, dict) or not vector_meta:
            raise ReaderStateError("Manifest does not contain vector metadata.")

        dim = int(vector_meta.get("dim", 0))
        metric = DistanceMetric(vector_meta.get("metric", "cosine"))
        strategy = VectorShardingStrategy(
            vector_meta.get("sharding_strategy", "explicit")
        )
        num_probes = int(vector_meta.get("num_probes", 1))

        cel_expr = vector_meta.get("cel_expr")
        cel_columns = vector_meta.get("cel_columns")
        raw_routing_values: list[Any] | None = vector_meta.get("routing_values")
        routing_values: list[int | str | bytes] | None = None
        if raw_routing_values is not None:
            decoded: list[int | str | bytes] = []
            for v in raw_routing_values:
                if isinstance(v, dict) and "__bytes_hex__" in v:
                    decoded.append(bytes.fromhex(v["__bytes_hex__"]))
                elif isinstance(v, (int, str, bytes)):
                    decoded.append(v)
                else:
                    decoded.append(v)  # type: ignore[arg-type]
            routing_values = decoded

        centroids: np.ndarray | None = None
        centroids_ref = vector_meta.get("centroids_ref")
        if centroids_ref:
            try:
                data = await asyncio.to_thread(self._backend.get, centroids_ref)
                centroids = np.load(io.BytesIO(data))
            except Exception:
                log_event(
                    "centroids_load_failed", logger=_logger, centroids_ref=centroids_ref
                )

        hyperplanes: np.ndarray | None = None
        hyperplanes_ref = vector_meta.get("hyperplanes_ref")
        if hyperplanes_ref:
            try:
                data = await asyncio.to_thread(self._backend.get, hyperplanes_ref)
                hyperplanes = np.load(io.BytesIO(data))
            except Exception:
                log_event(
                    "hyperplanes_load_failed",
                    logger=_logger,
                    hyperplanes_ref=hyperplanes_ref,
                )

        self._manifest_ref = ref
        self._manifest = manifest
        self._num_dbs = manifest.required_build.num_dbs
        self._shard_meta = {s.db_id: s for s in manifest.shards}
        self._metric = metric
        self._index_config = VectorIndexConfig(
            dim=dim,
            metric=metric,
            index_type=vector_meta.get("index_type", "hnsw"),
            quantization=vector_meta.get("quantization"),
        )
        self._sharding_strategy = strategy
        self._num_probes = num_probes
        self._centroids = centroids
        self._hyperplanes = hyperplanes
        self._cel_expr = cel_expr
        self._cel_columns = cel_columns
        self._routing_values = routing_values

    async def _search_shard(
        self,
        shard_id: int,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        cache_entry = await self._get_or_load_reader(shard_id)
        if cache_entry is None:
            return []

        try:
            started = time.perf_counter()
            results = await cache_entry.reader.search(query, top_k)
            elapsed_ms = (time.perf_counter() - started) * 1000

            if self._mc is not None:
                self._mc.emit(
                    MetricEvent.VECTOR_SHARD_SEARCH,
                    {
                        "shard_id": shard_id,
                        "top_k": top_k,
                        "num_results": len(results),
                        "elapsed_ms": elapsed_ms,
                    },
                )

            return results
        finally:
            self._release_reader(cache_entry)

    async def _get_or_load_reader(
        self, shard_id: int
    ) -> _CachedAsyncShardReader | None:
        async with self._cache_lock:
            cache_entry = self._shard_readers.get(shard_id)
            if cache_entry is not None:
                cache_entry.borrow_count += 1
                self._shard_readers.move_to_end(shard_id)
                return cache_entry

            meta = self._shard_meta.get(shard_id)
            if meta is None or meta.db_url is None:
                return None

            if shard_id not in self._shard_locks:
                self._shard_locks[shard_id] = asyncio.Lock()
            lock = self._shard_locks[shard_id]
            generation = self._cache_generation
            db_url = meta.db_url
            index_config = self._index_config or VectorIndexConfig(dim=0)

        async with lock:
            # Re-check cache under shard lock
            async with self._cache_lock:
                cache_entry = self._shard_readers.get(shard_id)
                if cache_entry is not None:
                    cache_entry.borrow_count += 1
                    self._shard_readers.move_to_end(shard_id)
                    return cache_entry

            local_dir = self._local_root / f"shard_{shard_id:05d}"
            assert self._manifest is not None
            reader = await self._reader_factory(
                db_url=db_url,
                local_dir=local_dir,
                index_config=index_config,
                manifest=self._manifest,
            )
            new_entry = _CachedAsyncShardReader(reader=reader, borrow_count=1)
            stale_reader_to_close: AsyncVectorShardReader | None = None
            reader_to_retire: _CachedAsyncShardReader | None = None

            async with self._cache_lock:
                cache_entry = self._shard_readers.get(shard_id)
                if cache_entry is not None:
                    cache_entry.borrow_count += 1
                    self._shard_readers.move_to_end(shard_id)
                    stale_reader_to_close = reader
                elif generation != self._cache_generation:
                    new_entry.retired = True
                    cache_entry = new_entry
                else:
                    self._shard_readers[shard_id] = new_entry
                    self._shard_readers.move_to_end(shard_id)
                    cache_entry = new_entry

                    if (
                        self._max_cached_shards is not None
                        and self._max_cached_shards > 0
                        and len(self._shard_readers) > self._max_cached_shards
                    ):
                        evict_id, evict_entry = self._shard_readers.popitem(last=False)
                        if evict_id == shard_id:
                            self._shard_readers[shard_id] = evict_entry
                            self._shard_readers.move_to_end(shard_id)
                        else:
                            self._shard_locks.pop(evict_id, None)
                            reader_to_retire = evict_entry

            if stale_reader_to_close is not None:
                try:
                    await stale_reader_to_close.close()
                except Exception:
                    pass
            if reader_to_retire is not None:
                self._retire_reader(reader_to_retire)
            return cache_entry

    def _release_reader(self, cache_entry: _CachedAsyncShardReader) -> None:
        async def _do_release() -> None:
            reader_to_close: AsyncVectorShardReader | None = None
            async with self._cache_lock:
                cache_entry.borrow_count -= 1
                if cache_entry.retired and cache_entry.borrow_count == 0:
                    reader_to_close = cache_entry.reader
            if reader_to_close is not None:
                try:
                    await reader_to_close.close()
                except Exception:
                    pass

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_release())
        except RuntimeError:
            pass

    def _retire_reader(self, cache_entry: _CachedAsyncShardReader) -> None:
        async def _do_retire() -> None:
            reader_to_close: AsyncVectorShardReader | None = None
            async with self._cache_lock:
                cache_entry.retired = True
                if cache_entry.borrow_count == 0:
                    reader_to_close = cache_entry.reader
            if reader_to_close is not None:
                try:
                    await reader_to_close.close()
                except Exception:
                    pass

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_retire())
        except RuntimeError:
            pass
