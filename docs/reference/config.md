# Configuration reference

This page enumerates the public configuration objects. Defaults and constraints come from source — links point at file:line.

## `HashWriteConfig`

`shardyfusion/config.py:260`. Dataclass, `slots=True`. Primary config for **HASH** sharded snapshot writes.

Inherits all common fields from `WriteConfig` (see below) and adds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `num_dbs` | `int \| None` | `None` | Number of shards. Required (>0) unless `max_keys_per_shard` is set. |
| `max_keys_per_shard` | `int \| None` | `None` | Alternative to `num_dbs` — computes shard count as `ceil(total_rows / max_keys_per_shard)` at write time. |

## `CelWriteConfig`

`shardyfusion/config.py:289`. Dataclass, `slots=True`. Primary config for **CEL** sharded snapshot writes.

Inherits all common fields from `WriteConfig` (see below) and adds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `cel_expr` | `str` | `""` | CEL expression that produces a shard ID or categorical token. Required. |
| `cel_columns` | `dict[str, str]` | `{}` | Mapping of CEL variable names to their types (e.g. `{"key": "int"}`). Required. |
| `routing_values` | `list[RoutingValue] \| None` | `None` | Optional categorical values for token-based routing. |
| `infer_routing_values_from_data` | `bool` | `False` | Discover routing values from input at write time (single-process only). |

## `WriteConfig` (base class)

`shardyfusion/config.py:162`. Dataclass, `slots=True`. **Base class** for `HashWriteConfig` and `CelWriteConfig`. Do not instantiate directly.

Common fields inherited by both concrete configs:

| Field | Type | Default |
|---|---|---|
| `s3_prefix` | `str` | `""` |
| `key_encoding` | `KeyEncoding` | `KeyEncoding.U64BE` |
| `batch_size` | `int` | `50_000` |
| `adapter_factory` | `DbAdapterFactory \| None` | `None` (treated as `SlateDbFactory()`) |
| `output` | `OutputOptions` | `OutputOptions()` |
| `manifest` | `ManifestOptions` | `ManifestOptions()` |
| `metrics_collector` | `MetricsCollector \| None` | `None` |
| `run_registry` | `RunRegistry \| None` | `None` |
| `shard_retry` | `RetryConfig \| None` | `None` |
| `credential_provider` | `CredentialProvider \| None` | `None` |
| `s3_connection_options` | `S3ConnectionOptions \| None` | `None` |
| `vector_spec` | `VectorSpec \| None` | `None` |

Manifest layout (path, naming, store) is configured via the nested `manifest: ManifestOptions` field — there is no top-level `manifest_store` or `manifest_name` on `WriteConfig`.

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

`shardyfusion/type_defs.py`. Enum: `U64BE`, `U32BE`, `UTF8`, `RAW`. Default on `WriteConfig` is `U64BE`.

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

## `VectorWriteConfig`

`shardyfusion/vector/config.py:74`. Standalone vector writer config.

Key fields (see source for full list):

- `num_dbs: int` — required.
- `s3_prefix: str` — required.
- `index_config: VectorIndexConfig` — required.
- `sharding: VectorShardingSpec` — default `cluster()`.
- `adapter_factory` — required (LanceDB or sqlite-vec).
- `batch_size: int = 10_000`.

## `VectorShardingSpec`

Strategies: `CLUSTER` (k-means, default), `LSH`, `EXPLICIT` (uses `VectorRecord.shard_id`), `CEL` (uses `routing_context`).

## Manifest paths

- Manifest: `manifests/<timestamp>_run_id=<run_id>/manifest`
- Run record: `runs/<timestamp>_run_id=<run_id>_<uuidhex>/run.yaml`
- Pointer: `_CURRENT` (configurable as `current_pointer_key`)
- Timestamp format: `%Y-%m-%dT%H:%M:%S.%fZ`
- Supported manifest format versions: `{1, 2, 3}`. v3 required for `routing_values`.
- `CurrentPointer.format_version: int = 1` (separate from manifest version).

## Adapter factories

| Factory | Module | Notes |
|---|---|---|
| `SlateDbFactory()` | `shardyfusion.slatedb_adapter` | Default. In top-level `__all__`. |
| `SqliteFactory(page_size=4096, cache_size_pages=-2000, journal_mode="OFF", synchronous="OFF", temp_store="MEMORY", mmap_size=0)` | `shardyfusion.sqlite_adapter` | Not re-exported. |
| `SqliteVecFactory(vector_spec, ...)` | `shardyfusion.sqlite_vec_adapter` | Unified KV+vector single backend. |
| `LanceDbFactory()` | `shardyfusion.vector.adapters.lancedb_adapter` | Vector only. |
| `CompositeFactory(kv_factory, vector_factory, vector_spec)` | `shardyfusion.composite_adapter` | KV + vector composition. |

All adapter factories are kw-only-callable: `factory(*, db_url, local_dir)`.
