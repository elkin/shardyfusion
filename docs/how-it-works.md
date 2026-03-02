# shardyfusion: How It Works

## Overview

`shardyfusion` builds a sharded snapshot from a Spark `DataFrame` into multiple independent SlateDB databases, writes metadata (manifest + CURRENT pointer), and provides service-side readers that route keys to the correct shard.

Core behavior:

- One logical snapshot write creates exactly `num_dbs` shard outputs.
- Writes are retry/speculation-safe with attempt-isolated paths.
- A deterministic winner is selected per shard (`db_id`) on the driver.
- Reader side loads CURRENT -> manifest -> opens per-shard readers -> routes lookups.

## Write Side (Snapshot Build)

Public entrypoint:

- `write_sharded(df, config, *, key_col, value_spec, sort_within_partitions=False, spark_conf_overrides=None, cache_input=False, storage_level=None, max_writes_per_second=None)`

### Write-side pipeline

1. Optional Spark conf overrides are applied during the call.
2. Optional input persistence (`persist`) is applied if `cache_input=True`.
3. `add_db_id_column` computes shard assignment using `ShardingSpec`:
   - `hash`: `pmod(xxhash64(cast(key_col as long)), num_dbs)`
   - `range`: explicit boundaries or boundaries from `approxQuantile`
   - `custom_expr`: Spark SQL expression or column builder
4. `prepare_partitioned_rdd` enforces `num_dbs` writer partitions:
   - data is converted to pair RDD `(db_id, row)`
   - `partitionBy(num_dbs, ...)`
   - then `mapPartitionsWithIndex` where partition index is shard `db_id`
5. Each partition writes one attempt-isolated shard location on S3-compatible storage.
6. Driver collects partition results (attempt metadata), picks deterministic winners per `db_id`.
7. Manifest artifact is built (default JSON builder unless custom builder is supplied).
8. Manifest is published, then CURRENT pointer is published.
9. `BuildResult` is returned with winners, artifact, refs, and typed stats.

### Retry/speculation safety

- Partition writer output URL includes `attempt=<NN>`.
- Multiple attempts can exist for the same `db_id`.
- Winner selection is deterministic:
  - lowest attempt number,
  - then lowest `task_attempt_id`,
  - then stable tie-break on URL.

## S3/Object Layout

Assume:

- `s3_prefix = s3://bucket/prefix`
- `run_id = 9f...`
- `manifest_name = manifest`
- `current_name = _CURRENT`

Write artifacts:

- Shard attempt data (temporary attempt-isolated paths):
  - `s3://bucket/prefix/_tmp/run_id=9f.../db=00000/attempt=00/...`
  - `s3://bucket/prefix/_tmp/run_id=9f.../db=00001/attempt=00/...`
  - ...
- Manifest object:
  - `s3://bucket/prefix/manifests/run_id=9f.../manifest`
- CURRENT pointer object (JSON):
  - `s3://bucket/prefix/_CURRENT`

Notes:

- Manifest stores winner shard URLs (`db_url`) and required routing/build metadata.
- Non-winning attempt paths are not automatically cleaned by the library.

## Configuration

Top-level writer config model: `WriteConfig`

Required fields:

- `num_dbs`
- `s3_prefix`

Direct fields:

- `key_encoding` (default `u64be`)
- `batch_size` (default 50,000)
- `adapter_factory` (default `SlateDbFactory()`)

Grouped options:

- `sharding: ShardingSpec`
  - `strategy`, `boundaries`, `approx_quantile_rel_error`, `custom_expr`, `custom_column_builder`
- `output: OutputOptions`
  - `run_id`
  - `db_path_template`
  - `tmp_prefix`
  - `local_root`
- `manifest: ManifestOptions`
  - `manifest_name`
  - `current_name`
  - `manifest_builder`
  - `publisher`
  - `custom_manifest_fields`
  - `s3_client_config`
    - `endpoint_url`
    - `region_name`
    - `access_key_id`
    - `secret_access_key`
    - `session_token`

Extra runtime controls on `write_sharded`:

- `key_col`: name of the key column
- `value_spec`: how to serialize row values
- `sort_within_partitions`: sort rows within each partition before writing
- `spark_conf_overrides`: temporary Spark settings for one call
- `cache_input`: persist input DataFrame for the call
- `storage_level`: Spark `StorageLevel` passed to `persist(...)`
- `max_writes_per_second`: rate-limit writes (see [Rate Limiting](writer.md#rate-limiting))

### Key type contracts

Enforced in `add_db_id_column`:

- `hash` strategy: `key_col` must be Spark `IntegerType` or `LongType`
- `range` strategy: `key_col` must be one of:
  - `IntegerType`, `LongType`, `FloatType`, `DoubleType`, `StringType`

## Reader Side

Primary class:

- `SlateShardedReader(...)`

### Reader initialization flow

1. Select manifest reader implementation:
   - default mode: uses `DefaultS3ManifestReader`
   - custom mode: caller may provide `manifest_reader`
2. Load CURRENT pointer.
3. Load manifest from CURRENT’s `manifest_ref`.
4. Build `SnapshotRouter`.
5. Open one SlateDB reader handle per shard (lock mode or pool mode).

### Lookup flow

- `get(key)`:
  - route key to `db_id`
  - encode key bytes using manifest `key_encoding`
  - read from selected shard reader
- `multi_get(keys)`:
  - group keys by shard
  - optional parallel shard fetch with `ThreadPoolExecutor`
  - return mapping for input keys

### Routing semantics

- `hash`:
  - writer: Spark `xxhash64(cast(key_col as long))`
  - reader: Python `xxhash` implementation aligned with Spark semantics for this contract
- `range`:
  - route by shard min/max intervals when present
  - fallback to manifest boundaries using right-biased boundary handling
- `custom_expr`:
  - Spark expression is not evaluated at read time
  - requires manifest routing hints (boundaries or shard ranges)

### Refresh and concurrency model

- `refresh()` loads CURRENT again.
- If `manifest_ref` unchanged: no-op.
- If changed:
  - open new shard readers/router
  - atomically swap active state
  - close retired readers when no in-flight operations are using them

## Mermaid Diagram: Write Side

```mermaid
flowchart TD
    A[DataFrame + WriteConfig] --> B[SparkConfOverrideContext]
    B --> C{cache_input?}
    C -->|yes| D[DataFrameCacheContext.persist]
    C -->|no| E[Use input DataFrame]
    D --> F[add_db_id_column]
    E --> F
    F --> G["prepare_partitioned_rdd<br/>partitionBy num_dbs"]
    G --> H[mapPartitionsWithIndex]
    H --> I["Write shard attempt<br/>_tmp/run_id/db/attempt"]
    I --> J[Collect attempt results]
    J --> K[Deterministic winner per db_id]
    K --> L[Build manifest artifact]
    L --> M[publish_manifest]
    M --> N[Build CURRENT JSON]
    N --> O[publish_current]
    O --> P[Return BuildResult]
```

## Mermaid Diagram: Reader Side

```mermaid
flowchart TD
    A[SlateShardedReader init] --> B{custom publisher?}
    B -->|no / default| C[DefaultS3ManifestReader]
    B -->|yes| D{manifest_reader provided?}
    D -->|no| E[Raise ValueError]
    D -->|yes| F[Use provided ManifestReader]
    C --> G[load_current]
    F --> G
    G --> H[load_manifest]
    H --> I[ParsedManifest]
    I --> J[Build SnapshotRouter]
    I --> K[Open one reader per shard]
    J --> L[get/multi_get route key to db_id]
    K --> L
    L --> M[Return values]
    M --> N[refresh]
    N --> O{manifest_ref changed?}
    O -->|no| P[No-op]
    O -->|yes| Q[Open new state and atomic swap]
    Q --> R[Close retired readers when refcount = 0]
```
