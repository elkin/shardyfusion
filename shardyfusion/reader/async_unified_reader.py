"""Async unified KV + vector reader — point lookups and vector search on one snapshot."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from shardyfusion._rate_limiter import RateLimiter
from shardyfusion.async_manifest_store import AsyncManifestStore
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import ConfigValidationError, ReaderStateError
from shardyfusion.logging import get_logger, log_event
from shardyfusion.manifest import ParsedManifest
from shardyfusion.metrics import MetricsCollector
from shardyfusion.reader._types import _NullShardReader
from shardyfusion.reader.async_reader import (
    AsyncShardedReader,
    _AsyncReaderState,
    _NullAsyncShardReader,
)
from shardyfusion.reader.unified_reader import (
    UnifiedVectorMeta,
    _distance_metric_from_str,
    _parse_vector_custom,
)
from shardyfusion.type_defs import (
    AsyncShardReaderFactory,
    S3ConnectionOptions,
)
from shardyfusion.vector._merge import merge_results
from shardyfusion.vector.types import DistanceMetric, SearchResult, VectorSearchResponse

_logger = get_logger(__name__)


def _auto_async_reader_factory(
    meta: UnifiedVectorMeta,
    *,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> AsyncShardReaderFactory:
    """Auto-select the async reader factory based on the manifest backend field.

    For SQLite-based backends (``sqlite-vec`` or ``sqlite`` KV), this selects
    the *adaptive* factory so each snapshot's access mode (download vs range)
    is decided per-shard size distribution.  Users that need a fixed mode
    must construct the concrete factory themselves and pass it via the
    reader's ``reader_factory=`` parameter.
    """
    if meta.backend == "sqlite-vec":
        from shardyfusion.sqlite_vec_adapter import (
            AsyncAdaptiveSqliteVecReaderFactory,
        )

        return AsyncAdaptiveSqliteVecReaderFactory(
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
    elif meta.backend == "lancedb":
        # lancedb: compose async KV reader + async vector reader
        from shardyfusion.composite_adapter import AsyncCompositeReaderFactory
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
        kv_factory: AsyncShardReaderFactory
        if kv_backend == "sqlite":
            from shardyfusion.sqlite_adapter import AsyncAdaptiveSqliteReaderFactory

            kv_factory = AsyncAdaptiveSqliteReaderFactory(
                credential_provider=credential_provider,
                s3_connection_options=s3_connection_options,
            )
        else:
            # Default: SlateDB
            from shardyfusion.reader.async_reader import AsyncSlateDbReaderFactory

            kv_factory = AsyncSlateDbReaderFactory(
                credential_provider=credential_provider,
            )

        try:
            from shardyfusion.vector.adapters.lancedb_adapter import (
                AsyncLanceDbReaderFactory,
            )

            vector_factory = AsyncLanceDbReaderFactory(
                s3_connection_options=s3_connection_options,
                credential_provider=credential_provider,
            )
        except ImportError as exc:
            raise ConfigValidationError(
                "Unified KV+vector async reader with lancedb backend requires "
                "the 'vector' extra. Install with: pip install shardyfusion[vector]"
            ) from exc

        return AsyncCompositeReaderFactory(
            kv_factory=kv_factory,
            vector_factory=vector_factory,
            vector_spec=vs,
        )
    else:
        raise ConfigValidationError(f"Unknown vector backend '{meta.backend}'.")


class AsyncUnifiedShardedReader(AsyncShardedReader):
    """Async sharded reader supporting both KV lookups and vector search.

    Inherits all ``AsyncShardedReader`` methods (``get``, ``multi_get``,
    ``refresh``, ``close``, etc.) and adds ``search()`` and
    ``batch_search()`` for vector similarity queries.
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
        self._manifest_custom: dict[str, Any] = {}
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
            max_concurrency=max_concurrency,
            max_fallback_attempts=max_fallback_attempts,
            metrics_collector=metrics_collector,
            rate_limiter=rate_limiter,
        )
        self._vector_meta = (
            _parse_vector_custom(self._manifest_custom)
            if self._manifest_custom
            else None
        )

    async def _build_state(
        self, manifest_ref: str, manifest: ParsedManifest
    ) -> _AsyncReaderState:
        vector_meta = _parse_vector_custom(manifest.custom)
        previous_factory = self._reader_factory

        new_factory = previous_factory
        if self._user_reader_factory is None:
            new_factory = _auto_async_reader_factory(
                vector_meta,
                credential_provider=self._credential_provider_for_auto,
                s3_connection_options=self._s3_conn_opts_for_auto,
            )

        self._reader_factory = new_factory
        committed = False
        try:
            state = await super()._build_state(manifest_ref, manifest)
            committed = True
        finally:
            if not committed:
                self._reader_factory = previous_factory

        self._manifest_custom = manifest.custom
        self._vector_meta = vector_meta
        return state

    @property
    def vector_meta(self) -> UnifiedVectorMeta:
        """Return vector metadata from the manifest."""
        if self._vector_meta is None:
            raise ReaderStateError("Reader not yet initialized")
        return self._vector_meta

    async def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        routing_context: dict[str, object] | None = None,
        shard_ids: list[int] | None = None,
    ) -> VectorSearchResponse:
        """Search for nearest neighbors across shards asynchronously."""
        state = self._require_state()

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire_async(1)

        t0 = time.perf_counter()

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

        semaphore = (
            asyncio.Semaphore(self._max_concurrency)
            if self._max_concurrency is not None
            else None
        )

        async def _search_shard_bounded(db_id: int) -> list[SearchResult]:
            reader = state.readers[db_id]
            if not hasattr(reader, "search"):
                if isinstance(reader, _NullAsyncShardReader) or isinstance(
                    reader, _NullShardReader
                ):
                    return []
                raise ReaderStateError(
                    "Shard reader does not support vector search for this unified snapshot"
                )

            from typing import cast

            if semaphore is not None:
                async with semaphore:
                    return await cast(Any, reader).search(query, top_k)
            else:
                return await cast(Any, reader).search(query, top_k)

        if len(target_shards) > 1:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(_search_shard_bounded(db_id))
                    for db_id in target_shards
                ]
            for task in tasks:
                per_shard_results.append(task.result())
        elif len(target_shards) == 1:
            per_shard_results.append(await _search_shard_bounded(target_shards[0]))

        metric = self.vector_meta.metric
        if not isinstance(metric, DistanceMetric):
            metric = _distance_metric_from_str(str(metric))

        merged = merge_results(per_shard_results, top_k, metric)

        latency_ms = (time.perf_counter() - t0) * 1000
        log_event(
            "async_unified_reader_search",
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

    async def batch_search(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        *,
        routing_context: dict[str, object] | None = None,
        shard_ids: list[int] | None = None,
    ) -> list[VectorSearchResponse]:
        """Search multiple queries. Each query is searched independently."""
        results = []
        for q in queries:
            results.append(
                await self.search(
                    q,
                    top_k,
                    routing_context=routing_context,
                    shard_ids=shard_ids,
                )
            )
        return results
