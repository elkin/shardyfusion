"""Unified KV + vector reader — point lookups and vector search on one snapshot.

Wraps a ``ShardedReader`` and adds ``search()`` / ``batch_search()`` methods
that query the vector index embedded in each shard (via sqlite-vec or
LanceDB sidecar).  Uses the same manifest, routing, and shard lifecycle as
the underlying KV reader.

Requires that the snapshot was built with ``vector_spec`` set on
``WriteConfig`` (unified KV+vector mode, CEL sharding only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from shardyfusion._rate_limiter import RateLimiter
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import ConfigValidationError, ReaderStateError
from shardyfusion.logging import get_logger, log_event
from shardyfusion.manifest import ParsedManifest
from shardyfusion.manifest_store import ManifestStore
from shardyfusion.metrics import MetricsCollector
from shardyfusion.type_defs import (
    S3ConnectionOptions,
    ShardReaderFactory,
)
from shardyfusion.vector._merge import merge_results
from shardyfusion.vector.types import DistanceMetric, SearchResult, VectorSearchResponse

from ._state import _SimpleReaderState
from ._types import _NullShardReader
from .reader import ShardedReader

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UnifiedVectorMeta:
    """Vector metadata parsed from the manifest custom fields."""

    dim: int
    metric: DistanceMetric
    index_type: str
    quantization: str | None
    index_params: dict[str, Any]
    backend: str  # "sqlite-vec" or "lancedb"
    kv_backend: str  # "slatedb", "sqlite", or "sqlite-vec"


def _parse_vector_custom(custom: dict[str, Any]) -> UnifiedVectorMeta:
    """Parse vector metadata from manifest custom fields."""
    vector = custom.get("vector")
    if not isinstance(vector, dict):
        raise ConfigValidationError(
            "Manifest does not contain vector metadata in custom fields. "
            "Was this snapshot built with vector_spec?"
        )
    return UnifiedVectorMeta(
        dim=int(vector["dim"]),
        metric=_distance_metric_from_str(str(vector["metric"])),
        index_type=str(vector.get("index_type", "hnsw")),
        quantization=vector.get("quantization"),
        index_params=vector.get("index_params", {}),
        backend=str(vector.get("backend", "lancedb")),
        kv_backend=str(vector.get("kv_backend", "slatedb")),
    )


def _auto_reader_factory(
    meta: UnifiedVectorMeta,
    *,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> ShardReaderFactory:
    """Auto-select the reader factory based on the manifest backend field.

    For SQLite-based backends (``sqlite-vec`` or ``sqlite`` KV), this selects
    the *adaptive* factory so each snapshot's access mode (download vs range)
    is decided per-shard size distribution.  Users that need a fixed mode
    must construct the concrete factory themselves and pass it via the
    reader's ``reader_factory=`` parameter.
    """
    if meta.backend == "sqlite-vec":
        from shardyfusion.sqlite_vec_adapter import AdaptiveSqliteVecReaderFactory

        return AdaptiveSqliteVecReaderFactory(
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
    elif meta.backend == "lancedb":
        # lancedb: compose KV reader + vector reader
        from shardyfusion.composite_adapter import CompositeReaderFactory
        from shardyfusion.config import VectorSpec, vector_metric_to_str

        vs = VectorSpec(
            dim=meta.dim,
            metric=vector_metric_to_str(meta.metric),
            index_type=meta.index_type,
            index_params=meta.index_params,
            quantization=meta.quantization,
        )

        # Select KV reader factory based on manifest kv_backend field
        kv_backend = getattr(meta, "kv_backend", None) or "slatedb"
        kv_factory: ShardReaderFactory
        if kv_backend == "sqlite":
            from shardyfusion.sqlite_adapter import AdaptiveSqliteReaderFactory

            kv_factory = AdaptiveSqliteReaderFactory(
                credential_provider=credential_provider,
                s3_connection_options=s3_connection_options,
            )
        else:
            # Default: SlateDB
            from shardyfusion.reader._types import SlateDbReaderFactory

            kv_factory = SlateDbReaderFactory(
                credential_provider=credential_provider,
            )

        try:
            from shardyfusion.storage import create_s3_client
            from shardyfusion.vector.adapters.lancedb_adapter import (
                LanceDbReaderFactory,
            )

            credentials = credential_provider.resolve() if credential_provider else None
            s3_client = create_s3_client(credentials, s3_connection_options)
            vector_factory = LanceDbReaderFactory(
                s3_client=s3_client,
                s3_connection_options=s3_connection_options,
                credential_provider=credential_provider,
            )
        except ImportError as exc:
            raise ConfigValidationError(
                "Unified KV+vector reader with lancedb backend requires "
                "the 'vector' extra. Install with: pip install shardyfusion[vector]"
            ) from exc

        return CompositeReaderFactory(
            kv_factory=kv_factory,
            vector_factory=vector_factory,
            vector_spec=vs,
        )
    else:
        raise ConfigValidationError(f"Unknown vector backend '{meta.backend}'.")


class UnifiedShardedReader(ShardedReader):
    """Sharded reader supporting both KV lookups and vector search.

    Inherits all ``ShardedReader`` methods (``get``, ``multi_get``,
    ``refresh``, ``close``, etc.) and adds ``search()`` and
    ``batch_search()`` for vector similarity queries.

    The shard reader instances must support both ``get(key)`` and
    ``search(query, top_k)`` — as provided by ``CompositeShardReader``
    or ``SqliteVecShardReader``.
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
        # _manifest_custom is set during _load_initial_state via
        # the overridden _build_simple_state.  On the first call
        # (from super().__init__), the parent builds state from the
        # manifest, which triggers _build_simple_state → captures custom.
        self._manifest_custom: dict[str, Any] = {}
        # Store for deferred auto-dispatch; None means "auto from manifest"
        self._user_reader_factory = reader_factory
        self._credential_provider_for_auto = credential_provider
        self._s3_conn_opts_for_auto = s3_connection_options
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
        self._vector_meta = _parse_vector_custom(self._manifest_custom)

    def _build_simple_state(
        self, manifest_ref: str, manifest: ParsedManifest
    ) -> _SimpleReaderState:
        """Build reader state only after vector metadata validates cleanly."""
        vector_meta = _parse_vector_custom(manifest.custom)
        previous_factory = self._reader_factory

        # Auto-dispatch reader factory from manifest backend on first load and
        # on every refresh, but only keep the new factory if state construction
        # succeeds. This avoids committing a factory/metadata change before the
        # parent reader swaps to the new manifest state.
        if self._user_reader_factory is None:
            self._reader_factory = _auto_reader_factory(
                vector_meta,
                credential_provider=self._credential_provider_for_auto,
                s3_connection_options=self._s3_conn_opts_for_auto,
            )

        try:
            state = super()._build_simple_state(manifest_ref, manifest)
        except Exception:
            self._reader_factory = previous_factory
            raise

        self._manifest_custom = manifest.custom
        self._vector_meta = vector_meta
        return state

    @property
    def vector_meta(self) -> UnifiedVectorMeta:
        """Return vector metadata from the manifest."""
        return self._vector_meta

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        routing_context: dict[str, object] | None = None,
        shard_ids: list[int] | None = None,
    ) -> VectorSearchResponse:
        """Search for nearest neighbors across shards.

        Args:
            query: Query vector of shape ``(dim,)``.
            top_k: Number of results to return.
            routing_context: CEL routing context to scope the search
                to a specific shard.  When provided, only that shard
                is queried.
            shard_ids: Explicit list of shard IDs to query. Overrides
                ``routing_context`` if both are provided.

        Returns:
            ``VectorSearchResponse`` with merged top-k results.
        """
        if self._closed:
            raise ReaderStateError("Reader is closed")

        t0 = time.perf_counter()
        state = self._state

        if shard_ids is not None:
            target_shards = shard_ids
        elif routing_context is not None:
            db_id = state.router.route_with_context(routing_context)
            target_shards = [db_id]
        else:
            # Query all non-empty shards
            target_shards = [
                i for i, s in enumerate(state.router.shards) if s.db_url is not None
            ]

        per_shard_results: list[list[SearchResult]] = []

        if self._executor is not None and len(target_shards) > 1:
            futures = {
                db_id: self._executor.submit(
                    _search_shard, state.readers[db_id], query, top_k
                )
                for db_id in target_shards
            }
            for _db_id, future in futures.items():
                per_shard_results.append(future.result())
        else:
            for db_id in target_shards:
                per_shard_results.append(
                    _search_shard(state.readers[db_id], query, top_k)
                )

        metric = self._vector_meta.metric
        if not isinstance(metric, DistanceMetric):
            metric = _distance_metric_from_str(str(metric))

        merged = merge_results(per_shard_results, top_k, metric)

        latency_ms = (time.perf_counter() - t0) * 1000
        log_event(
            "unified_reader_search",
            logger=_logger,
            top_k=top_k,
            num_shards_queried=len(target_shards),
            num_results=len(merged),
            latency_ms=round(latency_ms, 2),
        )

        return VectorSearchResponse(
            results=merged,
            num_shards_queried=len(target_shards),
            latency_ms=latency_ms,
        )

    def batch_search(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        *,
        routing_context: dict[str, object] | None = None,
        shard_ids: list[int] | None = None,
    ) -> list[VectorSearchResponse]:
        """Search multiple queries. Each query is searched independently."""
        return [
            self.search(
                queries[i],
                top_k,
                routing_context=routing_context,
                shard_ids=shard_ids,
            )
            for i in range(len(queries))
        ]

    def refresh(self) -> bool:
        """Refresh manifest, validating vector metadata before state swap."""
        return super().refresh()


def _search_shard(
    reader: Any,
    query: np.ndarray,
    top_k: int,
) -> list[SearchResult]:
    """Search a single shard reader."""
    if not hasattr(reader, "search"):
        if isinstance(reader, _NullShardReader):
            return []
        raise ReaderStateError(
            "Shard reader does not support vector search for this unified snapshot"
        )
    return reader.search(query, top_k)


def _distance_metric_from_str(metric: str) -> DistanceMetric:
    """Convert metric string to ``DistanceMetric`` with clear validation."""
    try:
        return DistanceMetric(metric)
    except ValueError as exc:
        valid_metrics = ", ".join(m.value for m in DistanceMetric)
        raise ConfigValidationError(
            f"Invalid vector metric '{metric}' in manifest metadata. "
            f"Expected one of: {valid_metrics}."
        ) from exc
