# Writer Side

Four writer backends are available, all producing the same manifest format:

| Backend | Entrypoint | Input Type | Requires Java |
|---------|-----------|------------|---------------|
| Spark | `shardyfusion.writer.spark.write_sharded` | PySpark `DataFrame` | Yes |
| Dask | `shardyfusion.writer.dask.write_sharded` | Dask `DataFrame` | No |
| Ray | `shardyfusion.writer.ray.write_sharded` | Ray `Dataset` | No |
| Python | `shardyfusion.writer.python.write_sharded` | `Iterable[T]` | No |

Each backend also has a `write_single_db` variant for single-shard writes (Spark, Dask, Ray). Depending on the installed extra, shard output can use the default SlateDB backend or the SQLite backend.

**Deep dives:** For data flow diagrams and framework-specific behavior, see the per-writer docs:

- [Spark Writer](writers/spark.md) — PySpark DataFrame, `mapInArrow` sharding, speculative execution support
- [Dask Writer](writers/dask.md) — Dask DataFrame, lazy execution model, pandas-based sharding
- [Ray Writer](writers/ray.md) — Ray Data, Arrow-native sharding, lock-guarded shuffle
- [Python Writer](writers/python.md) — Pure-Python iterator, single-process or parallel modes

Key guarantees (all backends):

- one-writer-per-db partitioning contract (`num_dbs` partitions)
- retry/speculation safety via attempt-isolated shard output URLs
- deterministic winner selection per `db_id`
- one run-scoped writer record per invocation for deferred cleanup discovery

## Typical Usage

### Spark

```python
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.spark import write_sharded

config = WriteConfig(num_dbs=8, s3_prefix="s3://bucket/prefix")

result = write_sharded(
    df,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    spark_conf_overrides={"spark.speculation": "false"},
)
```

### Dask

```python
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.dask import write_sharded

config = WriteConfig(num_dbs=8, s3_prefix="s3://bucket/prefix")

result = write_sharded(
    ddf,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)
```

### Ray

```python
import ray
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.ray import write_sharded

config = WriteConfig(num_dbs=8, s3_prefix="s3://bucket/prefix")
ds = ray.data.from_items([{"id": i, "payload": b"..."} for i in range(1000)])

result = write_sharded(
    ds,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)
```

### Python

```python
from shardyfusion import WriteConfig
from shardyfusion.writer.python import write_sharded

config = WriteConfig(num_dbs=8, s3_prefix="s3://bucket/prefix")

result = write_sharded(
    records,
    config,
    key_fn=lambda r: r["id"],
    value_fn=lambda r: r["payload"],
)
```

Manifest + CURRENT behavior:

1. winner shard metadata is assembled into a manifest artifact
2. manifest is published first
3. `_CURRENT` JSON pointer is published second

In parallel, the writer maintains one run record under `output.run_registry_prefix`
(default `runs`). The record starts as `running`, stores `manifest_ref` after
publish, and ends as `succeeded` or `failed`. Readers do not consume this
record; it is operational metadata for future cleanup workflows.

See [Architecture](how-it-works.md) for end-to-end data flow details.

## Run Registry

Writers now publish one YAML run record per invocation:

- location: `s3://bucket/prefix/<run_registry_prefix>/<timestamp>_run_id=<run_id>_<uuid>/run.yaml`
- config: `WriteConfig.output.run_registry_prefix` (default `runs`)
- result surface: `BuildResult.run_record_ref`

The run record is intentionally separate from the manifest:

- manifests stay reader-facing and contain only snapshot metadata readers need
- run records hold writer lifecycle state for operators and future deferred cleanup jobs
- inline loser cleanup via `cleanup_losers()` remains best-effort

## Single-Shard Writers

For datasets that don't need sharding (lookup tables, small reference data, configuration snapshots), use `write_single_db` to write everything into a single shard database. Available for Spark, Dask, and Ray backends.

### When to Use

- Small datasets where sharding adds unnecessary complexity
- Lookup tables that fit in a single shard
- Datasets where you want sorted key order within the database

### Spark

```python
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.spark import write_single_db

config = WriteConfig(num_dbs=1, s3_prefix="s3://bucket/prefix")

result = write_single_db(
    df,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    sort_keys=True,               # default: sort by key before writing
    num_partitions=None,           # default: use Spark's default parallelism
    prefetch_partitions=True,      # default: prefetch next partition during write
    cache_input=True,              # default: persist input DataFrame
    storage_level=None,            # default: MEMORY_AND_DISK
    spark_conf_overrides=None,     # optional Spark conf overrides
    max_writes_per_second=None,    # optional rate limit
)
```

### Dask

```python
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.dask import write_single_db

config = WriteConfig(num_dbs=1, s3_prefix="s3://bucket/prefix")

result = write_single_db(
    ddf,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    sort_keys=True,
    num_partitions=None,
    prefetch_partitions=True,
    cache_input=True,
    max_writes_per_second=None,
)
```

### Ray

```python
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.ray import write_single_db

config = WriteConfig(num_dbs=1, s3_prefix="s3://bucket/prefix")

result = write_single_db(
    ds,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    sort_keys=True,
    max_writes_per_second=None,
)
```

### Parameters

| Parameter | Spark | Dask | Ray | Default | Description |
|---|---|---|---|---|---|
| `sort_keys` | Yes | Yes | Yes | `True` | Sort rows by key before writing |
| `num_partitions` | Yes | Yes | — | `None` | Number of partitions for distributed sorting |
| `prefetch_partitions` | Yes | Yes | — | `True` | Prefetch next partition during write |
| `cache_input` | Yes | Yes | — | `True` | Persist/materialize input before processing |
| `storage_level` | Yes | — | — | `None` | Spark StorageLevel for caching |
| `spark_conf_overrides` | Yes | — | — | `None` | Temporary Spark configuration |
| `max_writes_per_second` | Yes | Yes | Yes | `None` | Rate limit (ops/sec) |

## Rate Limiting

All writer entry points accept an optional `max_writes_per_second` parameter
to throttle writes to the underlying shard databases.

```python
max_writes_per_second: float | None = None  # default: unlimited
```

When set, a thread-safe **token bucket** is created with capacity and refill
rate equal to the given value.  Each `write_batch()` call acquires tokens
equal to the number of rows in the batch, blocking until enough tokens are
available.  When `None` (the default), no bucket is created and writes
proceed at full speed.

### How it works

1. The bucket starts full — `rate` tokens available (= `max_writes_per_second`).
2. Before each batch write, `acquire(len(batch))` is called.
3. If enough tokens are available they are consumed immediately.
4. Otherwise the caller sleeps until tokens replenish (at `rate` tokens/second).
5. The internal lock is **released during the sleep**, preventing thread starvation.

### Per-backend behavior

Each writer backend creates its bucket(s) at a different scope, which affects
the aggregate write rate:

| Backend | Writer Function | Bucket Scope | Effective Meaning |
|---------|----------------|--------------|-------------------|
| Spark sharded | `write_sharded` | 1 bucket per partition (= per shard) | Each shard independently limited to `rate` rows/sec |
| Spark single-db | `write_single_db` (spark) | 1 shared bucket | All writes limited to `rate` rows/sec total |
| Dask sharded | `write_sharded` (dask) | 1 bucket per shard | Each shard independently limited to `rate` rows/sec |
| Dask single-db | `write_single_db` (dask) | 1 shared bucket | All writes limited to `rate` rows/sec total |
| Ray sharded | `write_sharded` (ray) | 1 bucket per shard | Each shard independently limited to `rate` rows/sec |
| Ray single-db | `write_single_db` (ray) | 1 shared bucket | All writes limited to `rate` rows/sec total |
| Python sequential | `write_sharded` | 1 shared bucket for all shards | All shards collectively limited to `rate` rows/sec |
| Python parallel | `write_sharded(parallel=True)` | 1 bucket per worker process | Each shard independently limited to `rate` rows/sec |

Key differences:

- **Sharded Spark/Dask/Ray**: independent per-shard buckets — aggregate write rate
  across all shards = `rate × num_dbs`.
- **Python sequential**: single shared bucket — aggregate write rate = `rate`
  (all shards compete for the same tokens).
- **Python parallel**: `multiprocessing.spawn` gives each worker its own
  bucket, so aggregate behavior matches Spark/Dask/Ray sharded.
- **Single-db writers**: always one bucket — straightforward `rate` rows/sec.

### Usage examples

```python
# Spark: from shardyfusion.writer.spark import write_sharded
result = write_sharded(
    df, config, key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    max_writes_per_second=10_000.0,
)

# Dask: from shardyfusion.writer.dask import write_sharded
result = write_sharded(
    ddf, config, key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    max_writes_per_second=10_000.0,
)

# Ray: from shardyfusion.writer.ray import write_sharded
result = write_sharded(
    ds, config, key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    max_writes_per_second=10_000.0,
)

# Python: from shardyfusion.writer.python import write_sharded
result = write_sharded(
    records, config, key_fn=lambda r: r["id"],
    value_fn=lambda r: r["payload"],
    max_writes_per_second=10_000.0,
)
```

### Bytes-per-second Rate Limiting

In addition to ops/sec limiting (`max_writes_per_second`), all backends support `max_write_bytes_per_second` to throttle by payload size:

```python
result = write_sharded(
    df, config, key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    max_writes_per_second=10_000.0,        # ops/sec limit
    max_write_bytes_per_second=50_000_000.0,  # 50 MB/sec limit
)
```

The two limits use independent `TokenBucket` instances — both must be satisfied before a batch is written. The ops bucket acquires tokens equal to the number of rows; the bytes bucket acquires tokens equal to the serialized payload size in bytes.

### Interaction with `batch_size`

`batch_size` (from `WriteConfig`, default 50,000) controls how many rows are
buffered before a single `write_batch()` call.  The rate limiter acquires
tokens equal to the number of rows in each batch — the two settings are
orthogonal:

- Smaller `batch_size` → more frequent but smaller acquire calls.
- Larger `batch_size` → fewer but bigger acquire calls.
- Neither setting affects the other; they can be tuned independently.

## Writer Retry

`WriteConfig.shard_retry` enables retry for the writer paths that can safely replay input:

```python
from shardyfusion import WriteConfig, RetryConfig

config = WriteConfig(
    num_dbs=8,
    s3_prefix="s3://bucket/prefix",
    shard_retry=RetryConfig(max_retries=2, initial_backoff=timedelta(seconds=1.0), backoff_multiplier=2.0),
)
```

Supported paths:

- Dask and Ray sharded writers: per-shard retry.
- Spark, Dask, and Ray `write_single_db()`: whole-database retry.
- Python `write_sharded(..., parallel=True)`: per-shard retry using a durable local spool file per shard.

Every retry writes to a fresh attempt path (`attempt=00`, `attempt=01`, ...) with a fresh adapter and fresh rate limiters. Non-retryable errors propagate immediately.

| Setting | Default | Description |
|---|---|---|
| `max_retries` | `3` | Maximum number of retry attempts after the initial failure |
| `initial_backoff` | `timedelta(seconds=1.0)` | Delay before the first retry |
| `backoff_multiplier` | `2.0` | Multiplier applied to the delay after each retry |

**Spark sharded** writes still rely on Spark task retry/speculation rather than shardyfusion-managed `shard_retry`, but their task attempts are still isolated by `attempt=NN`.

When `shard_retry` is `None` (the default), all writers make a single attempt with no retry — preserving existing behavior.

## Structured Logging in Writers

All writer pipelines integrate with shardyfusion's [structured logging](observability.md#structured-logging) system. Key fields (`run_id`, `db_id`, `attempt`) are automatically bound to log context during writes via `LogContext`.

### Enabling JSON Logging

```python
from shardyfusion.logging import configure_logging, LogContext

# Enable structured JSON logging for the shardyfusion.* hierarchy
configure_logging(level=logging.INFO, json_format=True)

# Wrap a write call with additional context
with LogContext(pipeline="daily-build", env="production"):
    result = write_sharded(df, config, key_col="id", value_spec=ValueSpec.binary_col("payload"))
```

Log output includes both the writer's internal fields and your custom context:

```json
{"timestamp": "2026-03-14 10:30:00,000", "level": "INFO", "logger": "shardyfusion.writer", "event": "write_started", "run_id": "abc123", "pipeline": "daily-build", "env": "production"}
```

See [Observability](observability.md) for the full logging API (LogContext nesting, FailureSeverity, JsonFormatter).
