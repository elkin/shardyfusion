"""Unified KV + vector reader — point lookups and vector search on one snapshot.

Wraps a ``ShardedReader`` and adds ``search()`` / ``batch_search()`` methods
that query the vector index embedded in each shard (via sqlite-vec or
USearch sidecar).  Uses the same manifest, routing, and shard lifecycle as
the underlying KV reader.

Requires that the snapshot was built with ``vector_spec`` set on
``WriteConfig`` (unified KV+vector mode, CEL sharding only).
"""

from __future__ import annotations

import heapq
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
from shardyfusion.vector.types import SearchResult, VectorSearchResponse

from ._state import _SimpleReaderState
from .reader import ShardedReader

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UnifiedVectorMeta:
    """Vector metadata parsed from the manifest custom fields."""

    dim: int
    metric: str
    index_type: str
    quantization: str | None
    index_params: dict[str, Any]
    backend: str  # "sqlite-vec" or "usearch-sidecar"


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
        metric=str(vector["metric"]),
        index_type=str(vector.get("index_type", "hnsw")),
        quantization=vector.get("quantization"),
        index_params=vector.get("index_params", {}),
        backend=str(vector.get("backend", "usearch-sidecar")),
    )


def _auto_reader_factory(
    meta: UnifiedVectorMeta,
    *,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> ShardReaderFactory:
    """Auto-select the reader factory based on the manifest backend field."""
    if meta.backend == "sqlite-vec":
        from shardyfusion.sqlite_vec_adapter import SqliteVecReaderFactory

        return SqliteVecReaderFactory(
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
    else:
        # usearch-sidecar: compose KV reader + vector reader
        from shardyfusion.composite_adapter import CompositeReaderFactory
        from shardyfusion.config import VectorSpec

        vs = VectorSpec(
            dim=meta.dim,
            metric=meta.metric,
            index_type=meta.index_type,
            index_params=meta.index_params,
            quantization=meta.quantization,
        )

        # Use SlateDB reader for KV side, USearch for vector side
        from shardyfusion.reader._types import SlateDbReaderFactory

        kv_factory = SlateDbReaderFactory()

        try:
            from shardyfusion.vector.adapters.usearch_adapter import (
                USearchReaderFactory,
            )

            vector_factory = USearchReaderFactory()
        except ImportError as exc:
            raise ConfigValidationError(
                "Unified KV+vector reader with usearch-sidecar backend requires "
                "the 'vector' extra. Install with: pip install shardyfusion[vector]"
            ) from exc

        return CompositeReaderFactory(
            kv_factory=kv_factory,
            vector_factory=vector_factory,
            vector_spec=vs,
        )


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
        # Store for deferred auto-dispatch
        self._auto_reader_factory = reader_factory is None
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
        """Override to capture custom fields and auto-select reader factory."""
        self._manifest_custom = manifest.custom
        # Auto-dispatch reader factory from manifest backend on first load
        if self._auto_reader_factory and manifest.custom.get("vector"):
            meta = _parse_vector_custom(manifest.custom)
            self._reader_factory = _auto_reader_factory(
                meta,
                credential_provider=self._credential_provider_for_auto,
                s3_connection_options=self._s3_conn_opts_for_auto,
            )
            self._auto_reader_factory = False
        return super()._build_simple_state(manifest_ref, manifest)

    @property
    def vector_meta(self) -> UnifiedVectorMeta:
        """Return vector metadata from the manifest."""
        return self._vector_meta

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        ef: int = 50,
        routing_context: dict[str, object] | None = None,
        shard_ids: list[int] | None = None,
    ) -> VectorSearchResponse:
        """Search for nearest neighbors across shards.

        Args:
            query: Query vector of shape ``(dim,)``.
            top_k: Number of results to return.
            ef: Expansion factor for HNSW search.
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

        all_results: list[SearchResult] = []

        if self._executor is not None and len(target_shards) > 1:
            futures = {
                db_id: self._executor.submit(
                    _search_shard, state.readers[db_id], query, top_k, ef
                )
                for db_id in target_shards
            }
            for _db_id, future in futures.items():
                all_results.extend(future.result())
        else:
            for db_id in target_shards:
                all_results.extend(
                    _search_shard(state.readers[db_id], query, top_k, ef)
                )

        # Merge: take top-k by lowest score (distance)
        merged = _merge_top_k(all_results, top_k, self._vector_meta.metric)

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
        ef: int = 50,
        routing_context: dict[str, object] | None = None,
        shard_ids: list[int] | None = None,
    ) -> list[VectorSearchResponse]:
        """Search multiple queries. Each query is searched independently."""
        return [
            self.search(
                queries[i],
                top_k,
                ef=ef,
                routing_context=routing_context,
                shard_ids=shard_ids,
            )
            for i in range(len(queries))
        ]

    def refresh(self) -> bool:
        """Refresh manifest and update vector metadata if changed."""
        changed = super().refresh()
        if changed:
            self._vector_meta = _parse_vector_custom(self._manifest_custom)
        return changed


def _search_shard(
    reader: Any,
    query: np.ndarray,
    top_k: int,
    ef: int,
) -> list[SearchResult]:
    """Search a single shard reader."""
    return reader.search(query, top_k, ef=ef)


def _merge_top_k(
    results: list[SearchResult],
    top_k: int,
    metric: str,
) -> list[SearchResult]:
    """Merge results from multiple shards, returning top-k.

    For distance-based metrics (cosine, l2), lower is better.
    For dot_product, higher is better.
    """
    if not results:
        return []

    if metric == "dot_product":
        # Higher score = better match
        return heapq.nlargest(top_k, results, key=lambda r: r.score)
    else:
        # Lower score = better match (cosine distance, L2 distance)
        return heapq.nsmallest(top_k, results, key=lambda r: r.score)
