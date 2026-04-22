# Public API reference

This page enumerates the **public** symbols (those in `shardyfusion.__all__`) and the most useful module-level imports. For full signatures, read the source — paths are linked.

## Top-level package

`shardyfusion/__init__.py` exports:

### Configuration

- `WriteConfig` — writer config (num_dbs, s3_prefix, sharding, adapter_factory, vector_spec, shard_retry, key_encoding, ...).
- `ShardingSpec` — sharding strategy + parameters (`strategy`, `routing_values`, `cel_expr`, `cel_columns`, `max_keys_per_shard`, `infer_routing_values_from_data`).
- `KeyEncoding` — `U64BE`, `U32BE`, `UTF8`, `RAW`.
- `VectorSpec` — vector dimension, metric, index type/params, optional quantization. Set on `WriteConfig.vector_spec` for unified KV+vector flows. The backend (lancedb/sqlite-vec) is determined by the adapter factory, not by `VectorSpec`.

### Sharding / routing

- `SnapshotRouter` — public methods: `from_build_meta`, `get_shard`, `route`, `group_keys`, `group_keys_allow_missing`, `route_with_context`, `is_lazy`.

### Manifests / runs

- `ManifestRef` — `(ref: str, run_id: str, published_at: datetime)`.
- `BuildResult`, `RequiredBuildMeta`, `RequiredShardMeta`, `BuildDurations`, `BuildStats`, `WriterInfo`, `ManifestArtifact`, `ManifestShardingSpec`, `ParsedManifest`, `CurrentPointer`, `SqliteManifestBuilder`, `SQLITE_MANIFEST_CONTENT_TYPE`.

### Adapters (top-level re-export)

- `SlateDbFactory` — only adapter factory in public `__all__`.
  - SQLite, sqlite-vec, composite factories must be imported from their modules:
    - `from shardyfusion.sqlite_adapter import SqliteFactory, SqliteReaderFactory, SqliteRangeReaderFactory, AsyncSqliteReaderFactory`
    - `from shardyfusion.sqlite_vec_adapter import SqliteVecFactory`
    - `from shardyfusion.composite_adapter import CompositeFactory`

### Readers

- `ShardedReader` — sync.
- `ConcurrentShardedReader` — sync with thread-pool fan-out and optional pooled reader copies per shard.
- `AsyncShardedReader` — async; construct via `await AsyncShardedReader.open(...)`.
- `ShardReaderHandle`, `AsyncShardReaderHandle` — borrowed per-shard handles returned by `reader_for_key` / `readers_for_keys`.
- `SnapshotInfo`, `ShardDetail`, `ReaderHealth` — value types returned by `snapshot_info`, `shard_details`, `health`.
- `SlateDbReaderFactory`, `AsyncSlateDbReaderFactory` — adapter factories for SlateDB.

Lazy unified KV+vector readers (loaded via top-level `__getattr__`):

- `UnifiedShardedReader` — sync. Resolves to `shardyfusion.reader.unified_reader.UnifiedShardedReader`.
- `AsyncUnifiedShardedReader` — async. Resolves to `shardyfusion.reader.async_unified_reader.AsyncUnifiedShardedReader`.

### Errors

- `ShardyfusionError` (base).
- `ConfigValidationError`, `ManifestBuildError`, `ManifestParseError`, `ManifestStoreError`, `PublishManifestError`, `PublishCurrentError`.
- `ShardCoverageError`, `ShardAssignmentError`, `ShardWriteError`.
- `PoolExhaustedError`, `ReaderStateError`, `S3TransientError`.
- `DbAdapterError`, `SlateDbApiError` (deprecated alias for `DbAdapterError`, still exported).
- **NOT in `__all__`**: `SqliteAdapterError`, `SqliteVecAdapterError`, `CompositeAdapterError`, `VectorIndexError`, `VectorSearchError`, `UnknownRoutingTokenError`. Import these from `shardyfusion.errors` (or, for `UnknownRoutingTokenError`, `shardyfusion.routing`) if needed.
- `UnknownRoutingTokenError` inherits from `ValueError`, **not** `ShardyfusionError`.

Note: there is **no** `ManifestNotFoundError`. A missing `_CURRENT` surfaces as `ReaderStateError("CURRENT pointer not found")` from the reader; the CLI converts it to `click.ClickException("No current manifest found")` for `cleanup`.

## Writers

Each framework has its own subpackage:

- `shardyfusion.writer.python` — `write_sharded`, `write_single_db`.
- `shardyfusion.writer.spark` — `write_sharded`, `write_single_db`, `DataFrameCacheContext`, `SparkConfOverrideContext`.
- `shardyfusion.writer.dask` — `write_sharded`, `write_single_db`, `DaskCacheContext`.
- `shardyfusion.writer.ray` — `write_sharded`, `write_single_db`, `RayCacheContext`.

`write_vector_sharded` exists on Spark/Dask/Ray writers but is **not re-exported** from their `__init__.py`. Import from the module file directly.

## Vector

`shardyfusion.vector` (also lazy):

- `VectorRecord(id, vector, payload=None, shard_id=None, routing_context=None)`.
- `VectorWriteConfig` — `num_dbs, s3_prefix, index_config, sharding, adapter_factory, batch_size`.
- `VectorIndexConfig(dim, metric, ...)`.
- `VectorShardingSpec` — strategies `CLUSTER`, `LSH`, `EXPLICIT`, `CEL`.
- `DistanceMetric` — `COSINE`, `L2`, `DOT_PRODUCT`.
- `write_vector_sharded(records, config) -> BuildResult`.
- `ShardedVectorReader` — `search(query, top_k, *, shard_ids=None, num_probes=None, routing_context=None)`. No filter parameter.

## Metrics

`shardyfusion.metrics`:

- `MetricsCollector` — Protocol.
- `MetricEvent` — `str, Enum` (`write.started`, `write.shard.completed`, `s3.retry`, `vector.reader.search`, ...).
- `PrometheusCollector(registry=None, prefix="shardyfusion_")`.
- `OtelCollector(meter_provider=None, meter_name="shardyfusion")`.

Pass `metrics_collector=None` to disable; there is no no-op recorder.

## CLI

Entry point: `shardy = "shardyfusion.cli.app:main"`. See [`reference/cli.md`](cli.md).

## Versioning

`shardyfusion/__init__.py` does not declare `__version__`. Read the version from `pyproject.toml` (`version = "0.1.0"`) or `importlib.metadata.version("shardyfusion")`.
