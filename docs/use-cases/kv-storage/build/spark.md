# Build a snapshot with the Spark writer

Use the **Spark writer** to build a sharded snapshot from a PySpark `DataFrame`. Requires Java 17+ and PySpark.

## When to use

- Records already live in a Spark `DataFrame` (lakehouse query, batch ETL output).
- Dataset is too large to stream through a single host.
- You have a Spark cluster (or local Spark) with Java 17+.

## When NOT to use

- No existing Spark pipeline — the [Python writer](python.md) is much simpler.
- No Java runtime available — use [Dask](dask.md) or [Ray](ray.md).

## Install

```bash
# SlateDB backend (default)
uv add 'shardyfusion[writer-spark-slatedb]'

# SQLite backend
uv add 'shardyfusion[writer-spark-sqlite]'
```

`pyspark>=3.3` comes with the extra. The CI matrix exercises Spark 3.5 and Spark 4.

## Minimal example

### HASH (default)

```python
from pyspark.sql import SparkSession
from shardyfusion import HashWriteConfig
from shardyfusion.writer.spark import write_sharded_by_hash
from shardyfusion.serde import ValueSpec

spark = SparkSession.builder.appName("sf-build").getOrCreate()
df = spark.read.parquet("s3://lake/users/")

config = HashWriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users",
)

result = write_sharded_by_hash(
    df,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)
print(result.manifest_ref.ref)
```

### CEL routing

```python
from shardyfusion import CelWriteConfig
from shardyfusion.writer.spark import write_sharded_by_cel
from shardyfusion.serde import ValueSpec

config = CelWriteConfig(
    cel_expr='key % 16u',
    cel_columns={"key": "int"},
    s3_prefix="s3://my-bucket/snapshots/users-cel",
)

result = write_sharded_by_cel(
    df,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)
```

### SQLite backend

Swap `adapter_factory` on either config:

```python
from shardyfusion.sqlite_adapter import SqliteFactory

config = HashWriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    adapter_factory=SqliteFactory(),
)
```

## Data flow

```mermaid
flowchart TD
    A[DataFrame + HashWriteConfig / CelWriteConfig] --> B["Apply Spark conf overrides<br/>(optional)"]
    B --> C{"cache_input?"}
    C -->|yes| D["Persist input DataFrame"]
    C -->|no| E[Use input DataFrame as-is]
    D --> F["Assign shard IDs<br/>(Arrow-native, hash or CEL)"]
    E --> F
    F --> G{"verify_routing?"}
    G -->|yes| H["Verify routing<br/>(sample rows on driver)"]
    G -->|no| I[Skip verification]
    H --> J["Partition by shard ID<br/>(num_dbs partitions)"]
    I --> J
    J --> K["Write one shard per task"]
    K --> L["Stream results to driver"]
    L --> M["Select winners<br/>(deterministic per shard)"]
    M --> N["Publish manifest + _CURRENT"]
    N --> O["Clean up non-winning attempts"]
    O --> P[Return BuildResult]
```

## Configuration

Spark-specific knobs on `write_sharded_by_hash` and `write_sharded_by_cel` (`shardyfusion/writer/spark/writer.py`):

| Param | Default | Purpose |
|---|---|---|
| `key_col` | required | DataFrame column used for routing. |
| `value_spec` | required | How to encode rows to bytes — `binary_col`, `json_cols`, or callable. |
| `sort_within_partitions` | `False` | Sort each partition by `key_col` before writing. |
| `spark_conf_overrides` | `None` | `dict[str, str]` applied for the duration of the build. |
| `cache_input` | `False` | `df.persist(...)` the input. |
| `storage_level` | `None` | StorageLevel for `cache_input`. |
| `verify_routing` | `True` | Re-verify writer-side routing matches reader-side router. |
| `max_writes_per_second` / `max_write_bytes_per_second` | `None` | Rate limits across all executors. |

`HashWriteConfig` and `CelWriteConfig` fields are the same as for the Python writer; SlateDB is the default `adapter_factory`.

## Backend-specific properties

### SlateDB

- Incremental writes through SlateDB adapter; `checkpoint()` flushes memtable.
- `sort_within_partitions=True` helps SlateDB compaction by writing keys in sorted order.

### SQLite

- Complete `.db` file per shard; one PUT per shard on upload.
- Whole-file rewrite on retry (unlike SlateDB incremental writes).

## Non-functional properties

- **Driver work**: routing planning, manifest assembly, S3 publish — all on the driver.
- **Executor work**: each task opens its adapter, writes batches of `config.batch_size`, calls `checkpoint()`.
- **No Python UDFs**: routing uses Arrow `mapInArrow` to avoid UDF overhead.
- **Rate limiting**: per-partition scope. Aggregate rate = `max_writes_per_second x num_dbs`.

## Speculative execution safety

Spark may launch duplicate tasks for slow partitions. This is safe because:

- S3 paths are attempt-isolated: `shards/run_id=.../db=XXXXX/attempt=00/` vs `attempt=01/`.
- Winner selection is deterministic: lowest attempt, then lowest `task_attempt_id`, then URL tiebreaker.
- Non-winning attempt paths are cleaned up after publishing.

## Guarantees

- Same as all writers: successful return => manifest + `_CURRENT` published.
- `verify_routing=True` (default) re-checks the writer-reader sharding contract before publishing.

## Weaknesses

- **Java 17+ required.**
- **Driver bottleneck for very wide manifests.** With thousands of shards, manifest assembly becomes serial driver work.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Bad config | `ConfigValidationError` | Fix config; nothing was written. |
| Routing mismatch | `ShardAssignmentError` (when `verify_routing=True`) | Bug in routing change; do not silence by disabling verification. |
| Spark task fails | Task retried by Spark; if `config.shard_retry` set, additional shard-level retry | Tune Spark retries + `config.shard_retry`. |
| All attempts of a shard fail | `ShardCoverageError` | Investigate executor logs. |
| Manifest publish fails | `PublishManifestError` | Transient — rerun. |
| `_CURRENT` publish fails | `PublishCurrentError` | Manifest exists; rerun publishes a new pointer. |

## See also

- [KV Storage Overview](../overview.md) — sharding, manifests, two-phase publish, safety
- [`architecture/writer-core.md`](../../../architecture/writer-core.md)
- [`architecture/sharding.md`](../../../architecture/sharding.md)
- [Python writer](python.md) — when you don't need Spark
- [Read -> Sync SlateDB](../read/sync/slatedb.md)
- [Read -> Sync SQLite](../read/sync/sqlite.md)
