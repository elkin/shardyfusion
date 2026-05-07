# Configuration reference

This page enumerates the public configuration objects. Defaults and constraints come from source — links point at file:line.

## `HashShardedWriteConfig`

`shardyfusion/config.py:260`. Dataclass, `slots=True`. Primary config for **HASH** sharded snapshot writes.

Inherits all common fields from `BaseShardedWriteConfig` (see below) and adds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `num_dbs` | `int \| None` | `None` | Number of shards. Required (>0) unless `max_keys_per_shard` is set. |
| `max_keys_per_shard` | `int \| None` | `None` | Alternative to `num_dbs` — computes shard count as `ceil(total_rows / max_keys_per_shard)` at write time. |

## `CelShardedWriteConfig`

`shardyfusion/config.py:289`. Dataclass, `slots=True`. Primary config for **CEL** sharded snapshot writes.

Inherits all common fields from `BaseShardedWriteConfig` (see below) and adds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `cel_expr` | `str` | `""` | CEL expression that produces a shard ID or categorical token. Required. |
| `cel_columns` | `dict[str, str]` | `{}` | Mapping of CEL variable names to their types (e.g. `{"key": "int"}`). Required. |
| `routing_values` | `list[RoutingValue] \| None` | `None` | Optional categorical values for token-based routing. |
| `infer_routing_values_from_data` | `bool` | `False` | Discover routing values from input at write time (single-process only). |

## `BaseShardedWriteConfig` (base class)

`shardyfusion/config.py:162`. Dataclass, `slots=True`. **Base class** for `HashShardedWriteConfig` and `CelShardedWriteConfig`. Do not instantiate directly.

Common fields inherited by both concrete configs:

| Field | Type | Default |
|---|---|---|
| `storage` | `WriterStorageConfig` | `WriterStorageConfig()` |
| `output` | `WriterOutputConfig` | `WriterOutputConfig()` |
| `manifest` | `WriterManifestConfig` | `WriterManifestConfig()` |
| `kv` | `KeyValueWriteConfig` | `KeyValueWriteConfig()` |
| `retry` | `WriterRetryConfig` | `WriterRetryConfig()` |
| `rate_limits` | `KvWriteRateLimitConfig` | `KvWriteRateLimitConfig()` |
| `observability` | `WriterObservabilityConfig` | `WriterObservabilityConfig()` |
| `lifecycle` | `WriterLifecycleConfig` | `WriterLifecycleConfig()` |
| `vector` | `VectorSpec \| None` | `None` |

For migration convenience, concrete config constructors also accept previous flat keyword names such as `s3_prefix`, `key_encoding`, `batch_size`, `adapter_factory`, `metrics_collector`, `run_registry`, `shard_retry`, `credential_provider`, `s3_connection_options`, `vector_spec`, and rate-limit fields. New code should prefer nested group names when several related fields are set together.

### Common nested groups

| Config | Key fields |
|---|---|
| `WriterStorageConfig` | `s3_prefix`, `credential_provider`, `s3_connection_options` |
| `WriterOutputConfig` | `run_id`, `db_path_template`, `shard_prefix`, `run_registry_prefix`, `local_root` |
| `WriterManifestConfig` | `store`, `custom_manifest_fields`, `credential_provider`, `s3_connection_options` |
| `KeyValueWriteConfig` | `key_encoding`, `batch_size`, `adapter_factory` |
| `WriterRetryConfig` | `shard_retry` |
| `KvWriteRateLimitConfig` | `max_writes_per_second`, `max_write_bytes_per_second` |
| `WriterObservabilityConfig` | `metrics_collector` |
| `WriterLifecycleConfig` | `run_registry` |

Manifest layout (path, naming, store) is configured via the nested `manifest: WriterManifestConfig` field — there is no top-level `manifest_store` or `manifest_name` on writer configs.

## `HashShardingSpec`

`shardyfusion/sharding_types.py`. Strategy-specific parameters for HASH routing:

- `hash_algorithm` — `ShardHashAlgorithm`, currently `XXH3_64` with `seed=0`.
- `max_keys_per_shard` — soft cap (writer-side); incompatible with explicit `num_dbs`.

## `CelShardingSpec`

`shardyfusion/sharding_types.py`. Strategy-specific parameters for CEL routing:

- `cel_expr` — CEL expression returning routing token.
- `cel_columns` — input columns for CEL.
- `routing_values` — closed token set (categorical CEL).
- `infer_routing_values_from_data` — derive `routing_values` from input.

## `ShardingSpec`

Base class for `HashShardingSpec` and `CelShardingSpec`.

## `KeyEncoding`

`shardyfusion/type_defs.py`. Enum: `U64BE`, `U32BE`, `UTF8`, `RAW`. Default on `KeyValueWriteConfig` is `U64BE`.

## `ShardHashAlgorithm`

`shardyfusion/sharding_types.py`. Enum: currently `XXH3_64` only. The value is required in manifest sharding metadata so future readers can reject unsupported algorithms rather than silently misrouting.

## `VectorSpec`

`shardyfusion/config.py:93`. Dataclass.

| Field | Type | Default |
|---|---|---|
| `dim` | `int` | required |
| `vector_col` | `str \| None` | `None` |
| `metric` | `VectorMetric` (str) | `"cosine"` (`"cosine"`, `"l2"`, `"dot_product"`) |
| `index_type` | `str` | `"hnsw"` |
| `index_params` | `dict[str, object]` | `{}` |
| `quantization` | `str \| None` | `None` (`"fp16"`, `"i8"`, or `None`) |
| `sharding` | `VectorSpecSharding` | default factory |

`VectorSpec` does **not** carry a `backend` field. Backend selection happens via the adapter factory (`SqliteVecFactory` ⇒ sqlite-vec; `CompositeFactory(..., vector_factory=LanceDbFactory(), ...)` ⇒ lancedb). The manifest's `vector.backend` field is filled in from the chosen adapter and used by `UnifiedShardedReader` to dispatch.

## `VectorShardedWriteConfig`

`shardyfusion/vector/config.py:74`. Standalone and distributed vector writer config.

Key fields (see source for full list):

- `index_config: VectorIndexConfig` — required, `dim > 0`.
- `sharding: VectorShardingConfig` — carries `num_dbs` and strategy-specific settings.
- `storage: WriterStorageConfig` — `s3_prefix` and object-store connection settings.
- `output: WriterOutputConfig` — run and shard path settings.
- `manifest: WriterManifestConfig` — manifest-store settings.
- `adapter: VectorAdapterConfig` — vector adapter factory, reader factory, and batch size.
- `rate_limits: VectorWriteRateLimitConfig` — vector write ops/sec limit.
- `observability: WriterObservabilityConfig` — metrics collector.
- `lifecycle: WriterLifecycleConfig` — run registry.

Like the KV configs, `VectorShardedWriteConfig` accepts flat migration keywords such as `num_dbs`, `s3_prefix`, `adapter_factory`, `reader_factory`, `batch_size`, and `max_writes_per_second`, but grouped configs are preferred for new code.

## Writer input and options

Public writer functions take `(data, config, input, options=None)` except standalone vector writes, where `input` is optional:

- `PythonRecordInput(key_fn, value_fn, columns_fn=None, vector_fn=None)` for Python KV writes.
- `ColumnWriteInput(key_col, value_spec, vector=None, vector_fn=None)` for Spark/Dask/Ray KV writes.
- `VectorColumnInput(vector_col, id_col=None, payload_cols=None, shard_id_col=None, routing_context_cols=None)` for distributed vector writes.
  - `shard_id_col` here is the user **input** column carrying explicit shard IDs (EXPLICIT strategy only). It is distinct from `config.shard_id_col` on `VectorShardedWriteConfig`, which names the internal routing column added by the writer.
- `PythonWriteOptions`, `SparkWriteOptions`, `DaskWriteOptions`, `RayWriteOptions`, `SingleDbWriteOptions`, and `VectorWriteOptions` carry per-call execution behavior.

## `VectorShardingConfig` / `VectorShardingSpec`

Strategies: `CLUSTER` (k-means, default), `LSH`, `EXPLICIT` (uses `VectorRecord.shard_id`), `CEL` (uses `routing_context`).

## Manifest paths

- Manifest: `manifests/<timestamp>_run_id=<run_id>/manifest`
- Run record: `runs/<timestamp>_run_id=<run_id>_<uuidhex>/run.yaml`
- Pointer: `_CURRENT` (configurable as `current_pointer_key`)
- Timestamp format: `%Y-%m-%dT%H:%M:%S.%fZ`
- Supported manifest format versions: `{4}`.
- `CurrentPointer.format_version: int = 1` (separate from manifest version).

## Adapter factories

| Factory | Module | Notes |
|---|---|---|
| `SlateDbFactory()` | `shardyfusion.slatedb_adapter` | Default. In top-level `__all__`. |
| `LocalSlateDbFactory(s3_connection_options=None, credential_provider=None)` | `shardyfusion.local_slatedb_adapter` | Local-first: writes to ``file://``, uploads to S3 on close. In top-level `__all__`. |
| `SqliteFactory(page_size=4096, cache_size_pages=-2000, journal_mode="OFF", synchronous="OFF", temp_store="MEMORY", mmap_size=0)` | `shardyfusion.sqlite_adapter` | Not re-exported. |
| `SqliteVecFactory(vector_spec, ...)` | `shardyfusion.sqlite_vec_adapter` | Unified KV+vector single backend. |
| `LanceDbFactory()` | `shardyfusion.vector.adapters.lancedb_adapter` | Vector only. |
| `CompositeFactory(kv_factory, vector_factory, vector_spec)` | `shardyfusion.composite_adapter` | KV + vector composition. |

All adapter factories are kw-only-callable: `factory(*, db_url, local_dir)`.
