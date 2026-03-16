# shardyfusion: How It Works

## Overview

`shardyfusion` builds a sharded snapshot into multiple independent SlateDB databases, writes metadata (manifest + CURRENT pointer), and provides service-side readers that route keys to the correct shard. Four writer backends are supported: Spark, Dask, Ray, and pure Python.

Core behavior:

- One logical snapshot write creates exactly `num_dbs` shard outputs.
- Writes are retry/speculation-safe with attempt-isolated paths.
- A deterministic winner is selected per shard (`db_id`) on the driver.
- Reader side loads CURRENT -> manifest -> opens per-shard readers -> routes lookups.

## Write Side (Snapshot Build)

All four backends follow the same three-phase pipeline:

1. **Sharding** — assign each row a `_slatedb_db_id` column
2. **Write** — partition data by `db_id`, write each shard to S3
3. **Publish** — build manifest, publish manifest, publish `_CURRENT` pointer

The backends differ in how they distribute work, but share all core logic from
`_writer_core.py` (routing, winner selection, manifest building, publishing).

### Spark write pipeline

Entrypoint: `shardyfusion.writer.spark.write_sharded`

1. Optional Spark conf overrides are applied during the call.
2. Optional input persistence (`persist`) is applied if `cache_input=True`.
3. `add_db_id_column` computes shard assignment using `ShardingSpec`:
   - `hash`: `pmod(xxhash64(cast(key_col as long)), num_dbs)`
   - `range`: explicit boundaries or boundaries from `approxQuantile`
4. `prepare_partitioned_rdd` enforces `num_dbs` writer partitions:
   - data is converted to pair RDD `(db_id, row)`
   - `partitionBy(num_dbs, ...)`
   - then `mapPartitionsWithIndex` where partition index is shard `db_id`
5. Each partition writes one attempt-isolated shard location on S3-compatible storage.
6. Driver collects partition results (attempt metadata), picks deterministic winners per `db_id`.
7. Manifest artifact is built, published, then CURRENT pointer is published.
8. `BuildResult` is returned with winners, artifact, refs, and typed stats.

### Dask write pipeline

Entrypoint: `shardyfusion.writer.dask.write_sharded`

1. `add_db_id_column` computes shard assignment via Python `route_key()` per partition.
   Range boundaries are computed via Dask quantiles.
2. Dask DataFrame is shuffled by `_slatedb_db_id`, then `map_partitions` writes each shard.
3. Empty shards (partitions with no rows) are omitted from the manifest — no S3 I/O is performed.
4. Optional rate limiting and routing verification.
5. Same `_writer_core.py` functions for winner selection, manifest building, and publishing.

### Ray write pipeline

Entrypoint: `shardyfusion.writer.ray.write_sharded`

1. `add_db_id_column` computes shard assignment via `map_batches` with Arrow batch format
   (`batch_format="pyarrow"`, `zero_copy_batch=True`) to avoid Arrow→pandas→Arrow overhead.
   Range boundaries are computed via sampling.
2. Dataset is repartitioned by `_slatedb_db_id` using hash shuffle
   (`repartition(num_dbs, shuffle=True, keys=[DB_ID_COL])`).
   `DataContext.shuffle_strategy` is saved/restored in a `try/finally` block.
3. `map_batches` with `batch_format="pandas"` writes each shard.
   Empty shards (partitions with no rows) are omitted from the manifest — no S3 I/O is performed.
4. Optional rate limiting and routing verification.
5. Same `_writer_core.py` functions for winner selection, manifest building, and publishing.

### Python write pipeline

Entrypoint: `shardyfusion.writer.python.write_sharded`

1. Accepts `Iterable[T]` with `key_fn`/`value_fn` callables.
2. Routes each item via `route_key()`, writes to the appropriate shard adapter.
3. Supports single-process (all adapters open simultaneously) or multi-process
   (`parallel=True`, one worker per shard via `multiprocessing.spawn`).
4. Same `_writer_core.py` functions for winner selection, manifest building, and publishing.

### Retry/speculation safety

- Partition writer output URL includes `attempt=<NN>`.
- Multiple attempts can exist for the same `db_id`.
- Winner selection is deterministic:
  - lowest attempt number,
  - then lowest `task_attempt_id`,
  - then stable tie-break on URL.

## Manifest Lifecycle

Every write ends with a two-phase publish. This guarantees that the manifest payload exists before the current pointer references it — readers never dereference a dangling pointer.

```mermaid
sequenceDiagram
    participant W as Writer
    participant S3 as S3 / ManifestStore
    participant R as Reader

    W->>S3: 1. PUT manifest payload
    Note over S3: manifests/{timestamp}_run_id={id}/manifest
    W->>S3: 2. PUT _CURRENT pointer
    Note over S3: _CURRENT → manifest ref

    R->>S3: GET _CURRENT
    S3-->>R: ManifestRef (ref, run_id, published_at)
    R->>S3: GET manifest by ref
    S3-->>R: ParsedManifest (required, shards, custom)
```

### Manifest Format

The manifest is a JSON object with three top-level keys:

- **`required`** — library-owned build metadata (`RequiredBuildMeta`): run_id, num_dbs, s3_prefix, sharding strategy, key_encoding, etc.
- **`shards`** — list of winner shard metadata (`RequiredShardMeta`): db_id, db_url, row_count, min/max key, checkpoint_id
- **`custom`** — user-defined fields passed via `ManifestOptions.custom_manifest_fields`

### CURRENT Pointer Format

A small JSON object pointing to the current manifest:

```json
{
  "manifest_ref": "s3://bucket/prefix/manifests/2026-03-14T10:30:00.000000Z_run_id=abc123/manifest",
  "manifest_content_type": "application/json",
  "run_id": "abc123",
  "updated_at": "2026-03-14T10:30:01Z",
  "format_version": 1
}
```

### ManifestRef

`ManifestRef` is the backend-agnostic handle that abstracts away whether the manifest lives in S3 or a database. Returned by `load_current()` and `list_manifests()`, it carries `ref` (the backend-specific reference), `run_id`, and `published_at`.

See [Manifest Stores](manifest-stores.md) for details on S3, database, and in-memory backends.

## S3/Object Layout

Assume:

- `s3_prefix = s3://bucket/prefix`
- `run_id = 9f...`
- `manifest_name = manifest`
- `current_name = _CURRENT`

Write artifacts:

- Shard attempt data (attempt-isolated paths):
  - `s3://bucket/prefix/shards/run_id=9f.../db=00000/attempt=00/...`
  - `s3://bucket/prefix/shards/run_id=9f.../db=00001/attempt=00/...`
  - ...
- Manifest object (timestamp-prefixed for chronological listing):
  - `s3://bucket/prefix/manifests/2026-03-14T10:30:00.000000Z_run_id=9f.../manifest`
- CURRENT pointer object (JSON):
  - `s3://bucket/prefix/_CURRENT`

Notes:

- Manifest S3 keys are timestamp-prefixed (e.g., `2026-03-14T10:30:00.000000Z_run_id=abc123/manifest`) so that `list_manifests()` can use S3 `CommonPrefixes` for chronological ordering.
- Manifest stores winner shard URLs (`db_url`) and required routing/build metadata.
- Non-winning attempt paths are automatically cleaned up after publish via `cleanup_losers()` (best-effort).

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
  - `strategy`, `boundaries`, `approx_quantile_rel_error`
- `output: OutputOptions`
  - `run_id`
  - `db_path_template`
  - `shard_prefix`
  - `local_root`
- `manifest: ManifestOptions`
  - `manifest_builder`
  - `store`
  - `custom_manifest_fields`
  - `credential_provider`
  - `s3_connection_options`
- `credential_provider` (top-level, inherited by manifest store if not set per-store)
- `s3_connection_options` (top-level, inherited by manifest store if not set per-store)

Extra runtime controls on `write_sharded` (vary by backend):

- `key_col`: name of the key column (Spark, Dask, Ray)
- `value_spec`: how to serialize row values (Spark, Dask, Ray)
- `sort_within_partitions`: sort rows within each partition before writing (Spark, Dask, Ray)
- `max_writes_per_second`: rate-limit writes (all backends; see [Rate Limiting](writer.md#rate-limiting))
- `verify_routing`: runtime routing spot-check (Spark, Dask, Ray; default `True`)
- `spark_conf_overrides`: temporary Spark settings (Spark only)
- `cache_input` / `storage_level`: persist input DataFrame (Spark only)
- `key_fn` / `value_fn`: key/value extraction callables (Python only)
- `parallel`: multi-process mode (Python only)

### Sharding strategy support

| Strategy | Spark | Dask | Ray | Python |
|----------|-------|------|-----|--------|
| `HASH` | Yes | Yes | Yes | Yes |
| `RANGE` | Yes | Yes | Yes | Yes |

## Reader Side

Primary classes: `ShardedReader`, `ConcurrentShardedReader`, `AsyncShardedReader`. See [Reader Side](reader.md) for full usage docs.

### Reader initialization flow

1. Select manifest store implementation:
   - default mode: uses `S3ManifestStore`
   - custom mode: caller may provide `manifest_store`
2. Load CURRENT pointer.
3. Load manifest from CURRENT’s `manifest_ref`. If the manifest is malformed, [cold-start fallback](reader.md#cold-start-fallback) attempts previous manifests.
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
  - Spark writer: `xxhash64(cast(key_col as long))` (Spark SQL)
  - Dask/Ray/Python writers: `route_key()` from `_writer_core.py` (same xxhash64 algorithm)
  - reader: Python `xxhash` implementation aligned with Spark semantics for this contract
- `range`:
  - prefers explicit shard boundaries over min/max intervals when both are available
  - route by shard min/max intervals when present; empty shards (`None`/`None` bounds) are excluded from interval routing
  - fallback to manifest boundaries using right-biased boundary handling

### Refresh and concurrency model

- `refresh()` loads CURRENT again.
- If `manifest_ref` unchanged: no-op.
- If changed:
  - open new shard readers/router
  - atomically swap active state
  - close retired readers when no in-flight operations are using them

## Mermaid Diagram: Write Side (Generic)

```mermaid
flowchart TD
    A["Input data<br/>(DataFrame / Dataset / Iterable)"] --> B[add_db_id_column<br/>route_key per row]
    B --> C["Partition by db_id<br/>(RDD / shuffle / repartition)"]
    C --> D["Write each shard<br/>shards/run_id/db/attempt"]
    D --> E[Collect attempt results]
    E --> F[Deterministic winner per db_id]
    F --> G[Build manifest artifact]
    G --> H[publish_manifest]
    H --> I[Build CURRENT JSON]
    I --> J[publish_current]
    J --> K[Return BuildResult]
```

## Mermaid Diagram: Write Side (Spark Detail)

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
    H --> I["Write shard attempt<br/>shards/run_id/db/attempt"]
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
    A[ConcurrentShardedReader init] --> B{custom store?}
    B -->|no / default| C[S3ManifestStore]
    B -->|yes| D{manifest_store provided?}
    D -->|no| E[Raise ValueError]
    D -->|yes| F[Use provided ManifestStore]
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
