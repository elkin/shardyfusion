# Build a SlateDB snapshot with the Spark writer

Use the **Spark writer** to build a sharded SlateDB snapshot from a Spark `DataFrame`. The writer partitions the DataFrame by the sharding strategy and runs one shard write per partition.

## When to use

- Records already live in a Spark `DataFrame` (lakehouse query, batch ETL output).
- Dataset is too large to stream through a single host.
- You have a Spark cluster (or local Spark) with Java 17+.
- You want SlateDB on the read side.

## When NOT to use

- No existing Spark pipeline — the Python writer ([`build-python-slatedb.md`](build-python-slatedb.md)) is much simpler.
- Dataset fits in memory and you don't need a cluster.
- You want SQLite shards — use [`build-spark-sqlite.md`](build-spark-sqlite.md).

## Install

```bash
uv add 'shardyfusion[writer-spark]'
```

`pyspark>=3.3` (and `pandas>=2.2`, `pyarrow>=15.0`) come with the extra. The CI matrix exercises Spark 3.5 and Spark 4.

## Minimal example

```python
from pyspark.sql import SparkSession
from shardyfusion import WriteConfig
from shardyfusion.writer.spark import write_sharded
from shardyfusion.serde import ValueSpec

spark = SparkSession.builder.appName("sf-build").getOrCreate()
df = spark.read.parquet("s3://lake/users/")  # columns: id, payload (binary)

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users",
)

result = write_sharded(
    df,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),    # value column is already bytes
)
print(result.manifest_ref.ref)
```

## Configuration

Spark-specific knobs on `write_sharded` (`shardyfusion/writer/spark/writer.py:105`):

| Param | Default | Purpose |
|---|---|---|
| `key_col` | required | DataFrame column used for routing. |
| `value_spec` | required | How to encode rows to bytes — `binary_col`, `json_cols`, or callable. |
| `sort_within_partitions` | `False` | Sort each partition by `key_col` before writing (helps SlateDB compaction). |
| `spark_conf_overrides` | `None` | `dict[str, str]` of Spark conf overrides applied for the duration of the build. |
| `cache_input` | `False` | `df.persist(...)` the input — useful when the DataFrame is expensive to recompute. |
| `storage_level` | `None` | StorageLevel for `cache_input`. |
| `verify_routing` | `True` | Re-verify writer-side routing matches reader-side router. Disable only after profiling. |
| `max_writes_per_second` / `max_write_bytes_per_second` | `None` | Rate limits across all executors. |
| `vector_fn` / `vector_columns` | `None` | For unified KV+vector mode (see vector use cases). |

`config: WriteConfig` is the same as for the Python writer; SlateDB is the default `adapter_factory`.

## Single-DB variant

`write_single_db` (`writer/spark/single_db_writer.py:40`) writes to a single SlateDB shard (enforces `config.num_dbs == 1`). Useful for small reference tables. Knobs include `num_partitions`, `prefetch_partitions`, `cache_input`.

## Functional properties

- **Partitioned by shard**: input is repartitioned by computed `db_id` so each Spark task writes one shard.
- **Routing**: `route_key` runs in `mapInArrow` for HASH and CEL strategies (`writer/spark/sharding.py`).
- **Atomic publish**: same two-phase publish as all writers — driver waits for all tasks, builds manifest, publishes manifest then `_CURRENT`.

## Non-functional properties

- **Driver work**: routing planning, manifest assembly, S3 publish — all on the driver. Big drivers help for very large `num_dbs`.
- **Executor work**: shard writes. Each task opens its adapter, writes batches of `config.batch_size`, calls `checkpoint()`.
- **No Python UDFs**: routing uses Arrow `mapInArrow` to avoid UDF overhead. Don't introduce row-at-a-time Python UDFs in your pipeline upstream of `write_sharded` if you can avoid them.

## Guarantees

- Same as Python writer: successful return ⇒ manifest + `_CURRENT` are published.
- `verify_routing=True` (default) re-checks the writer-reader sharding contract before publishing — a guard against routing drift.

## Weaknesses

- **Java 17+ required.** Without Java the build fails immediately.
- **Driver bottleneck for very wide manifests.** With thousands of shards, manifest assembly becomes serial driver work.
- **`write_vector_sharded` not re-exported.** It exists at `shardyfusion/writer/spark/writer.py:702` but is not in `__init__.py`'s `__all__`. Import directly from the module if you need it.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Bad config | `ConfigValidationError` | Fix config; nothing was written. |
| Routing mismatch (writer/reader disagreement) | `ShardAssignmentError` (raised when `verify_routing=True`) | Indicates a bug in the routing change you made; do not disable `verify_routing` to silence it. |
| Spark task fails with backend error | task retried by Spark; if `config.shard_retry` set, additional shard-level retry | Tune Spark task retries + `config.shard_retry`. |
| All attempts of a shard fail | `ShardCoverageError` | Investigate executor logs. |
| Manifest publish fails | `PublishManifestError` | Transient — rerun. |
| `_CURRENT` publish fails | `PublishCurrentError` | Manifest exists; rerun publishes a new pointer. |

## Reading the snapshot

The reader is **decoupled from the writer flavor**. A snapshot built with the Spark writer reads with the same `ShardedReader` / `AsyncShardedReader` as one built with Python, Dask, or Ray. There's no Spark requirement on the read side.

```python
from shardyfusion import ShardedReader

with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
) as reader:
    print(reader.get(42))
    print(reader.snapshot_info())
    with reader.reader_for_key(42) as handle:
        ...  # native SlateDB handle for the shard
```

See [`read-sync-slatedb.md`](read-sync-slatedb.md) for the full reader API.

## See also

- [`architecture/writer-core.md`](../architecture/writer-core.md) — what every writer shares.
- [`architecture/sharding.md`](../architecture/sharding.md) — HASH vs CEL.
- [`build-spark-sqlite.md`](build-spark-sqlite.md) — same writer, SQLite backend.
- [`build-dask-slatedb.md`](build-dask-slatedb.md) / [`build-ray-slatedb.md`](build-ray-slatedb.md) — when you don't want Spark.
- [`contributing/adding-a-writer.md`](../contributing/adding-a-writer.md) — for adding new writer flavors.
