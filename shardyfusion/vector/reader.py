"""Sharded vector reader — sync, thread-pool fan-out."""

from __future__ import annotations

import io
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._rate_limiter import RateLimiter
from ..credentials import CredentialProvider
from ..errors import ConfigValidationError, ReaderStateError
from ..logging import get_logger, log_event
from ..manifest import ManifestRef, ParsedManifest, RequiredShardMeta
from ..manifest_store import ManifestStore, S3ManifestStore
from ..metrics._events import MetricEvent
from ..metrics._protocol import MetricsCollector
from ..storage import get_bytes
from ..type_defs import S3ConnectionOptions
from ._merge import merge_results
from .config import VectorIndexConfig
from .sharding import route_vector_to_shards
from .types import (
    DistanceMetric,
    SearchResult,
    VectorSearchResponse,
    VectorShardDetail,
    VectorShardingStrategy,
    VectorShardReader,
    VectorShardReaderFactory,
    VectorSnapshotInfo,
)

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VectorReaderHealth:
    """Diagnostic snapshot of vector reader state."""

    status: Literal["healthy", "degraded", "unhealthy"]
    manifest_ref: str | None
    manifest_age_seconds: float | None
    num_shards: int
    is_closed: bool


@dataclass(slots=True)
class _CachedShardReader:
    """Cache entry that defers close until in-flight searches release it."""

    reader: VectorShardReader
    borrow_count: int = 0
    retired: bool = False


class ShardedVectorReader:
    """Read-side router that loads a vector manifest and fans out searches."""

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        reader_factory: VectorShardReaderFactory | None = None,
        manifest_store: ManifestStore | None = None,
        max_workers: int | None = None,
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
        self._max_workers = max_workers
        self._max_fallback_attempts = max_fallback_attempts
        self._preload_shards = preload_shards
        self._max_cached_shards = max_cached_shards
        self._mc = metrics_collector
        self._rate_limiter = rate_limiter
        self._credential_provider = credential_provider
        self._s3_connection_options = s3_connection_options
        self._closed = False
        self._refresh_lock = threading.Lock()

        self._store = manifest_store or S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
            metrics_collector=metrics_collector,
        )

        # S3 client for loading centroids/hyperplanes
        credentials = credential_provider.resolve() if credential_provider else None
        from ..storage import create_s3_client

        self._s3_client = create_s3_client(credentials, s3_connection_options)

        # Shard reader cache (lazy loading)
        self._shard_readers: OrderedDict[int, _CachedShardReader] = OrderedDict()
        self._shard_locks: dict[int, threading.Lock] = {}
        self._cache_generation = 0
        self._cache_lock = (
            threading.Lock()
        )  # protects shard_locks creation + LRU eviction

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

        # Factory selection: either user-supplied, or auto-dispatched based
        # on the manifest backend after it's loaded.  Populated below.
        self._user_reader_factory: VectorShardReaderFactory | None = reader_factory
        self._reader_factory: VectorShardReaderFactory  # set during init

        self._load_initial_manifest()

        if self._mc is not None:
            self._mc.emit(
                MetricEvent.VECTOR_READER_INITIALIZED,
                {"s3_prefix": s3_prefix, "num_shards": self._num_dbs},
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        ef: int = 50,
        shard_ids: list[int] | None = None,
        num_probes: int | None = None,
        routing_context: dict[str, Any] | None = None,
    ) -> VectorSearchResponse:
        """Search across shards and merge results.

        For CEL sharding, provide ``routing_context`` to route the query to
        a specific shard via the CEL expression.  For CLUSTER/LSH, the query
        vector itself determines the target shards.
        """
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

        # Filter to shards that actually have data
        target_shards = [s for s in target_shards if s in self._shard_meta]

        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        # Fan out to shards
        per_shard_results: list[list[SearchResult]] = []
        if self._max_workers and len(target_shards) > 1:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {
                    pool.submit(self._search_shard, sid, query, top_k, ef): sid
                    for sid in target_shards
                }
                for future in futures:
                    per_shard_results.append(future.result())
        else:
            for sid in target_shards:
                per_shard_results.append(self._search_shard(sid, query, top_k, ef))

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

    def batch_search(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[VectorSearchResponse]:
        """Search multiple queries sequentially."""
        return [self.search(q, top_k, **kwargs) for q in queries]

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
        """Return shard IDs that would be queried for this vector."""
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
        """Return metadata for a specific shard."""
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
        """Return metadata for all shards."""
        self._check_open()
        details: list[VectorShardDetail] = []
        for db_id in range(self._num_dbs):
            details.append(self.shard_for_id(db_id))
        return details

    def snapshot_info(self) -> VectorSnapshotInfo:
        """Return snapshot-level metadata."""
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
        """Return diagnostic snapshot of reader state."""
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

    def refresh(self) -> bool:
        """Reload manifest from S3. Returns True if manifest changed."""
        self._check_open()
        with self._refresh_lock:
            try:
                new_ref = self._store.load_current()
            except Exception:
                return False

            if new_ref is None:
                return False
            if self._manifest_ref is not None and new_ref.ref == self._manifest_ref.ref:
                return False

            try:
                manifest = self._store.load_manifest(new_ref.ref)
            except Exception:
                return False

            # Validate the new manifest before discarding current state.
            # If _apply_manifest raises (e.g. unknown metric), we keep the
            # old readers and return False rather than leaving a half-updated
            # reader.
            try:
                self._apply_manifest(new_ref, manifest)
            except Exception:
                log_event(
                    "vector_reader_refresh_apply_failed",
                    logger=_logger,
                    manifest_ref=new_ref.ref,
                )
                return False

            with self._cache_lock:
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

    def close(self) -> None:
        """Close all shard readers and release resources."""
        if self._closed:
            return
        self._closed = True
        with self._cache_lock:
            readers_to_close = list(self._shard_readers.values())
            self._cache_generation += 1
            self._shard_readers.clear()
            self._shard_locks = {}
        for cache_entry in readers_to_close:
            self._retire_reader(cache_entry)

        if self._mc is not None:
            self._mc.emit(MetricEvent.VECTOR_READER_CLOSED, {})

    def __enter__(self) -> ShardedVectorReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ReaderStateError("Reader is closed")

    def _load_initial_manifest(self) -> None:
        """Load manifest on construction, with fallback."""
        from ..errors import ManifestParseError

        ref = self._store.load_current()
        if ref is None:
            raise ReaderStateError("No CURRENT pointer found")

        # Try current, then fall back to previous manifests
        try:
            manifest = self._store.load_manifest(ref.ref)
        except ManifestParseError:
            if self._max_fallback_attempts <= 0:
                raise
            all_refs = self._store.list_manifests()
            manifest = None
            # list_manifests() returns newest-first; skip the current
            # (already-failed) ref and try progressively older ones.
            attempts_left = self._max_fallback_attempts
            for fallback_ref in all_refs:
                if fallback_ref.ref == ref.ref:
                    continue
                if attempts_left <= 0:
                    break
                attempts_left -= 1
                try:
                    manifest = self._store.load_manifest(fallback_ref.ref)
                    ref = fallback_ref
                    break
                except ManifestParseError:
                    continue
            if manifest is None:
                raise

        self._apply_manifest(ref, manifest)

        if self._preload_shards:
            for db_id in self._shard_meta:
                cache_entry = self._get_or_load_reader(db_id)
                if cache_entry is not None:
                    self._release_reader(cache_entry)

    def _apply_manifest(
        self,
        ref: ManifestRef,
        manifest: ParsedManifest,
    ) -> None:
        """Apply a parsed manifest to reader state."""
        self._manifest_ref = ref
        self._manifest = manifest
        self._num_dbs = manifest.required_build.num_dbs

        # Build shard metadata map
        self._shard_meta = {s.db_id: s for s in manifest.shards}

        # Parse vector metadata from custom fields
        vector_meta = manifest.custom.get("vector", {})
        if vector_meta:
            dim = vector_meta.get("dim", 0)
            metric_str = vector_meta.get("metric", "cosine")
            metric = DistanceMetric(metric_str)
            self._metric = metric
            self._index_config = VectorIndexConfig(
                dim=dim,
                metric=metric,
                index_type=vector_meta.get("index_type", "hnsw"),
                quantization=vector_meta.get("quantization"),
            )
            strategy_str = vector_meta.get("sharding_strategy", "explicit")
            self._sharding_strategy = VectorShardingStrategy(strategy_str)
            self._num_probes = vector_meta.get("num_probes", 1)

            # Load CEL metadata if present
            self._cel_expr = vector_meta.get("cel_expr")
            self._cel_columns = vector_meta.get("cel_columns")
            raw_rv: list[Any] | None = vector_meta.get("routing_values")
            if raw_rv is not None:
                decoded: list[int | str | bytes] = []
                for v in raw_rv:
                    if isinstance(v, dict) and "__bytes_hex__" in v:
                        decoded.append(bytes.fromhex(v["__bytes_hex__"]))
                    elif isinstance(v, (int, str, bytes)):
                        decoded.append(v)
                    else:
                        decoded.append(v)  # type: ignore[arg-type]
                self._routing_values = decoded
            else:
                self._routing_values = None

            # Load centroids/hyperplanes if referenced
            centroids_ref = vector_meta.get("centroids_ref")
            if centroids_ref:
                try:
                    data = get_bytes(centroids_ref, s3_client=self._s3_client)
                    self._centroids = np.load(io.BytesIO(data))
                except Exception:
                    self._centroids = None  # clear stale data on failure
                    log_event(
                        "centroids_load_failed",
                        logger=_logger,
                        centroids_ref=centroids_ref,
                    )

            hyperplanes_ref = vector_meta.get("hyperplanes_ref")
            if hyperplanes_ref:
                try:
                    data = get_bytes(hyperplanes_ref, s3_client=self._s3_client)
                    self._hyperplanes = np.load(io.BytesIO(data))
                except Exception:
                    self._hyperplanes = None  # clear stale data on failure
                    log_event(
                        "hyperplanes_load_failed",
                        logger=_logger,
                        hyperplanes_ref=hyperplanes_ref,
                    )

        # Select the reader factory based on the manifest backend, unless
        # the user supplied one explicitly.
        backend = vector_meta.get("backend", "usearch") if vector_meta else "usearch"
        self._reader_factory = self._select_reader_factory(backend)

    def _select_reader_factory(self, backend: str) -> VectorShardReaderFactory:
        """Pick a reader factory for ``backend`` (or honor user override)."""
        if self._user_reader_factory is not None:
            return self._user_reader_factory

        if backend == "sqlite-vec":
            try:
                from .adapters.sqlite_vec_adapter import (
                    SqliteVecVectorReaderFactory,
                )
            except ImportError as exc:  # pragma: no cover - defensive
                raise ConfigValidationError(
                    "Manifest declares backend 'sqlite-vec' but the "
                    "'vector-sqlite' extra is not installed. "
                    "Install with: pip install 'shardyfusion[vector-sqlite]'"
                ) from exc

            return SqliteVecVectorReaderFactory(
                credential_provider=self._credential_provider,
                s3_connection_options=self._s3_connection_options,
            )

        if backend == "usearch":
            from .adapters.usearch_adapter import USearchReaderFactory

            return USearchReaderFactory(s3_client=self._s3_client)

        raise ConfigValidationError(
            f"Unknown vector backend in manifest: {backend!r}. "
            "Supported backends: 'usearch', 'sqlite-vec'."
        )

    def _search_shard(
        self,
        shard_id: int,
        query: np.ndarray,
        top_k: int,
        ef: int,
    ) -> list[SearchResult]:
        """Search a single shard, loading it lazily if needed."""
        cache_entry = self._get_or_load_reader(shard_id)
        if cache_entry is None:
            return []

        try:
            started = time.perf_counter()
            results = cache_entry.reader.search(query, top_k, ef=ef)
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

    def _get_or_load_reader(self, shard_id: int) -> _CachedShardReader | None:
        """Get or lazily load a shard reader."""
        with self._cache_lock:
            cache_entry = self._shard_readers.get(shard_id)
            if cache_entry is not None:
                cache_entry.borrow_count += 1
                self._shard_readers.move_to_end(shard_id)
                return cache_entry

            meta = self._shard_meta.get(shard_id)
            if meta is None or meta.db_url is None:
                return None

            if shard_id not in self._shard_locks:
                self._shard_locks[shard_id] = threading.Lock()
            lock = self._shard_locks[shard_id]
            generation = self._cache_generation
            db_url = meta.db_url
            index_config = self._index_config or VectorIndexConfig(dim=0)

        with lock:
            with self._cache_lock:
                cache_entry = self._shard_readers.get(shard_id)
                if cache_entry is not None:
                    cache_entry.borrow_count += 1
                    self._shard_readers.move_to_end(shard_id)
                    return cache_entry

            local_dir = self._local_root / f"shard_{shard_id:05d}"
            reader = self._reader_factory(
                db_url=db_url,
                local_dir=local_dir,
                index_config=index_config,
            )
            new_entry = _CachedShardReader(reader=reader, borrow_count=1)
            stale_reader_to_close: VectorShardReader | None = None
            reader_to_retire: _CachedShardReader | None = None

            with self._cache_lock:
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
                    stale_reader_to_close.close()
                except Exception:
                    pass
            if reader_to_retire is not None:
                self._retire_reader(reader_to_retire)
            return cache_entry

    def _release_reader(self, cache_entry: _CachedShardReader) -> None:
        reader_to_close: VectorShardReader | None = None
        with self._cache_lock:
            cache_entry.borrow_count -= 1
            if cache_entry.retired and cache_entry.borrow_count == 0:
                reader_to_close = cache_entry.reader
        if reader_to_close is not None:
            try:
                reader_to_close.close()
            except Exception:
                pass

    def _retire_reader(self, cache_entry: _CachedShardReader) -> None:
        reader_to_close: VectorShardReader | None = None
        with self._cache_lock:
            cache_entry.retired = True
            if cache_entry.borrow_count == 0:
                reader_to_close = cache_entry.reader
        if reader_to_close is not None:
            try:
                reader_to_close.close()
            except Exception:
                pass
