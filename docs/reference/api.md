# Public API reference

This page enumerates the **public** symbols (those in `shardyfusion.__all__`) and the most useful module-level imports. For full signatures, read the source — paths are linked.

## Top-level package

`shardyfusion/__init__.py` exports:

### Configuration

- `HashShardedWriteConfig` — writer config for HASH sharding (num_dbs, max_keys_per_shard, s3_prefix, adapter_factory, vector_spec, shard_retry, key_encoding, ...).
- `CelShardedWriteConfig` — writer config for CEL sharding (cel_expr, cel_columns, routing_values, infer_routing_values_from_data, s3_prefix, adapter_factory, vector_spec, shard_retry, key_encoding, ...).
- `BaseShardedWriteConfig` — shared base class for KV writer configs. Do not instantiate directly.
- `SingleDbWriteConfig` — writer config for single-database KV snapshots.
- `WriterStorageConfig`, `WriterOutputConfig`, `WriterManifestConfig`, `KeyValueWriteConfig`, `WriterRetryConfig`, `KvWriteRateLimitConfig`, `WriterObservabilityConfig`, `WriterLifecycleConfig` — nested config groups shared by KV writer configs.
- `PythonRecordInput`, `ColumnWriteInput`, `VectorColumnInput` — input mapping objects passed to public writer functions.
- `PythonWriteOptions`, `SparkWriteOptions`, `DaskWriteOptions`, `RayWriteOptions`, `SingleDbWriteOptions`, `SharedMemoryOptions`, `BufferingOptions` — per-call writer execution options.
- `HashShardingSpec` / `CelShardingSpec` — sharding strategy parameters for HASH and CEL respectively.
- `ShardingSpec` — base class for sharding specs.
- `KeyEncoding` — `U64BE`, `U32BE`, `UTF8`, `RAW`.
- `ShardHashAlgorithm` — currently `XXH3_64`.
- `VectorSpec` — vector dimension, metric, index type/params, optional quantization. Set on `HashShardedWriteConfig.vector_spec` / `CelShardedWriteConfig.vector_spec` for unified KV+vector flows. The backend (lancedb/sqlite-vec) is determined by the adapter factory, not by `VectorSpec`.

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
- `SlateDbSettings` — typed dataclass for slatedb 0.12 settings (`Settings.set(dotted_key, json_literal)`); includes a `raw_overrides: dict[str, str]` escape hatch for keys not yet promoted to typed fields. Re-exported at the top level. Legacy JSON-dict configs are still accepted but emit `DeprecationWarning`.

#### Adapter Protocol changes (slatedb 0.12)

The `DbAdapter` write-side Protocol exposes:

- `write_batch(pairs)`, `flush()`, `seal() -> None`, `db_bytes() -> int`, `close()`, context-manager.

`seal()` (formerly `checkpoint() -> str | None`) finalises the shard DB and uploads it; it returns `None`. Writers stamp the shard's `checkpoint_id` themselves via `shardyfusion._checkpoint_id.generate_checkpoint_id()` (an opaque `uuid4().hex`). See [adding an adapter](../contributing/adding-an-adapter.md) for the full template.

### Readers

- `ShardedReader` — sync.
- `ConcurrentShardedReader` — sync with thread-pool fan-out and optional pooled reader copies per shard.
- `AsyncShardedReader` — async; construct via `await AsyncShardedReader.open(...)`.
- `ShardReaderHandle`, `AsyncShardReaderHandle` — borrowed per-shard handles returned by `reader_for_key` / `readers_for_keys`.
- `SnapshotInfo`, `ShardDetail`, `ReaderHealth` — value types returned by `snapshot_info`, `shard_details`, `health`.
- `SlateDbReaderFactory`, `AsyncSlateDbReaderFactory` — adapter factories for SlateDB. The sync `SlateDbReaderFactory` accepts `iterator_chunk_size: int = 1024` to amortise the sync→async bridge cost on `scan_iter` (tune downward for low-latency partial scans, upward for bulk dumps); the async factory exposes the native uniffi iterator and does not need this knob. The `checkpoint_id` argument is accepted on both for Protocol symmetry but ignored — slatedb 0.12 has no read-side checkpoint API. SQLite/SQLiteVec/LanceDB factories continue to use `checkpoint_id` as a local-cache identity key.

Lazy unified KV+vector readers (loaded via top-level `__getattr__`):

- `UnifiedShardedReader` — sync. Resolves to `shardyfusion.reader.unified_reader.UnifiedShardedReader`.
- `AsyncUnifiedShardedReader` — async. Resolves to `shardyfusion.reader.async_unified_reader.AsyncUnifiedShardedReader`.

### Errors

- `ShardyfusionError` (base).
- `ConfigValidationError`, `ManifestBuildError`, `ManifestParseError`, `ManifestStoreError`, `PublishManifestError`, `PublishCurrentError`.
- `ShardCoverageError`, `ShardAssignmentError`, `ShardWriteError`.
- `PoolExhaustedError`, `ReaderStateError`, `S3TransientError`.
- `DbAdapterError`
- **NOT in `__all__`**: `SqliteAdapterError`, `SqliteVecAdapterError`, `CompositeAdapterError`, `VectorIndexError`, `VectorSearchError`, `UnknownRoutingTokenError`. Import these from `shardyfusion.errors` (or, for `UnknownRoutingTokenError`, `shardyfusion.routing`) if needed.
- `UnknownRoutingTokenError` inherits from `ValueError`, **not** `ShardyfusionError`.

Note: there is **no** `ManifestNotFoundError`. A missing `_CURRENT` surfaces as `ReaderStateError("CURRENT pointer not found")` from the reader; the CLI converts it to `click.ClickException("No current manifest found")` for `cleanup`.

## Writers

Each framework has its own subpackage. Sharded writers are split by strategy:

- `shardyfusion.writer.python` — `write_hash_sharded`, `write_cel_sharded`.
- `shardyfusion.writer.spark` — `write_hash_sharded`, `write_cel_sharded`, `write_single_db`, `DataFrameCacheContext`, `SparkConfOverrideContext`.
- `shardyfusion.writer.dask` — `write_hash_sharded`, `write_cel_sharded`, `write_single_db`, `DaskCacheContext`.
- `shardyfusion.writer.ray` — `write_hash_sharded`, `write_cel_sharded`, `write_single_db`, `RayCacheContext`.

KV writer signatures are intentionally small:

```python
write_hash_sharded(data, config, input, options=None)
write_cel_sharded(data, config, input, options=None)
write_single_db(data, config, input, options=None)
```

For Python iterables, `input` is `PythonRecordInput`. For Spark/Dask/Ray DataFrame-style writers, `input` is `ColumnWriteInput`. Framework-specific settings such as routing verification, caching, sorting, and Python multiprocessing live in the matching `*WriteOptions` object.

## Vector

`shardyfusion.vector` (also lazy):

- `VectorRecord(id, vector, payload=None, shard_id=None, routing_context=None)`.
- `VectorShardedWriteConfig` — `index_config, sharding, storage, output, manifest, adapter, rate_limits, observability, lifecycle`.
- `VectorIndexConfig(dim, metric, ...)`.
- `VectorShardingConfig` / `VectorShardingSpec` — strategies `CLUSTER`, `LSH`, `EXPLICIT`, `CEL`; carries vector `num_dbs`.
- `VectorRecordInput`, `VectorWriteOptions` — standalone vector input marker and per-call options.
- `DistanceMetric` — `COSINE`, `L2`, `DOT_PRODUCT`.
- `write_sharded(records, config, input=None, options=None) -> BuildResult`.
- `ShardedVectorReader` — `search(query, top_k, *, shard_ids=None, num_probes=None, routing_context=None)`. No filter parameter.

Distributed vector writers are imported from engine-specific vector modules:

```python
from shardyfusion.writer.spark.vector_writer import write_sharded
from shardyfusion.writer.dask.vector_writer import write_sharded
from shardyfusion.writer.ray.vector_writer import write_sharded
```

Their signature is `write_sharded(data, config, input, options=None)`, where `input` is `VectorColumnInput`.

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
