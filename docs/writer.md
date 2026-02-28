# Writer Side

Primary entrypoint:

- `write_sharded(df, config, *, key_col, value_spec, sort_within_partitions=False, spark_conf_overrides=None, cache_input=False, storage_level=None, max_writes_per_second=None)`

Key guarantees:

- one-writer-per-db partitioning contract (`num_dbs` partitions)
- retry/speculation safety via attempt-isolated shard output URLs
- deterministic winner selection per `db_id`

Typical usage:

```python
from slatedb_spark_sharded import WriteConfig, ValueSpec, write_sharded

config = WriteConfig(
    num_dbs=8,
    s3_prefix="s3://bucket/prefix",
)

result = write_sharded(
    df,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    spark_conf_overrides={"spark.speculation": "false"},
)
```

Manifest + CURRENT behavior:

1. winner shard metadata is assembled into a manifest artifact
2. manifest is published first
3. `_CURRENT` JSON pointer is published second

See `Architecture` for end-to-end data flow details.

## Rate Limiting

All writer entry points accept an optional `max_writes_per_second` parameter
to throttle writes to the underlying SlateDB shards.

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
| Spark single-db | `write_single_db_spark` | 1 shared bucket | All writes limited to `rate` rows/sec total |
| Dask sharded | `write_sharded_dask` | 1 bucket per shard | Each shard independently limited to `rate` rows/sec |
| Dask single-db | `write_single_db_dask` | 1 shared bucket | All writes limited to `rate` rows/sec total |
| Python sequential | `write_sharded` | 1 shared bucket for all shards | All shards collectively limited to `rate` rows/sec |
| Python parallel | `write_sharded(parallel=True)` | 1 bucket per worker process | Each shard independently limited to `rate` rows/sec |

Key differences:

- **Sharded Spark/Dask**: independent per-shard buckets — aggregate write rate
  across all shards = `rate × num_dbs`.
- **Python sequential**: single shared bucket — aggregate write rate = `rate`
  (all shards compete for the same tokens).
- **Python parallel**: `multiprocessing.spawn` gives each worker its own
  bucket, so aggregate behavior matches Spark/Dask sharded.
- **Single-db writers**: always one bucket — straightforward `rate` rows/sec.

### Usage examples

```python
# Spark: limit each shard to 10,000 rows/sec
result = write_sharded(
    df, config, key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    max_writes_per_second=10_000.0,
)

# Dask: same parameter
result = write_sharded_dask(
    ddf, config, key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    max_writes_per_second=10_000.0,
)

# Python: shared bucket across all shards in sequential mode
result = write_sharded(
    records, config, key_fn=lambda r: r["id"],
    value_fn=lambda r: r["payload"],
    max_writes_per_second=10_000.0,
)
```

### Interaction with `batch_size`

`batch_size` (from `WriteConfig`, default 50,000) controls how many rows are
buffered before a single `write_batch()` call.  The rate limiter acquires
tokens equal to the number of rows in each batch — the two settings are
orthogonal:

- Smaller `batch_size` → more frequent but smaller acquire calls.
- Larger `batch_size` → fewer but bigger acquire calls.
- Neither setting affects the other; they can be tuned independently.
