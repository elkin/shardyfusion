# Build a SQLite snapshot with the Spark writer

Same as [`build-spark-slatedb.md`](build-spark-slatedb.md), but each shard is a SQLite `.db` file.

## When to use

- Spark-scale ingestion + SQLite read side (SQL on each shard, range-read, offline distribution).

## When NOT to use

- Want SlateDB shards — use [`build-spark-slatedb.md`](build-spark-slatedb.md).

## Install

```bash
uv add 'shardyfusion[writer-spark-sqlite]'
```

## Minimal example

```python
from pyspark.sql import SparkSession
from shardyfusion import WriteConfig
from shardyfusion.sqlite_adapter import SqliteFactory
from shardyfusion.writer.spark import write_sharded
from shardyfusion.serde import ValueSpec

spark = SparkSession.builder.appName("sf-build-sqlite").getOrCreate()
df = spark.read.parquet("s3://lake/users/")

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    adapter_factory=SqliteFactory(),
)

result = write_sharded(df, config, key_col="id",
                      value_spec=ValueSpec.binary_col("payload"))
```

## Configuration

Same as [`build-spark-slatedb.md`](build-spark-slatedb.md). The only change is `WriteConfig.adapter_factory = SqliteFactory()`.

## Functional / Non-functional properties

Inherits from [`build-spark-slatedb.md`](build-spark-slatedb.md). Per-shard cost adds a full SQLite file PUT (multipart for large shards) instead of SlateDB's incremental object layout.

## Guarantees / Weaknesses / Failure modes

See [`build-spark-slatedb.md`](build-spark-slatedb.md) and [`build-python-sqlite.md`](build-python-sqlite.md) for SQLite-specific failure modes (`SqliteAdapterError`).

## Reading the snapshot

The reader is **decoupled from the writer flavor** — there's no Spark dependency on the read side. The reader provides **unified routed access** across all SQLite shards, and you can reach a single shard directly when you need raw SQL or want to bypass routing. SQLite shards can be accessed either by **download-and-cache** (one S3 GET per shard, then local) or **remotely via S3 range requests** (no full download).

```python
from shardyfusion import ShardedReader
from shardyfusion.sqlite_adapter import SqliteReaderFactory, SqliteRangeReaderFactory

# Routed access across all shards
with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteReaderFactory(),  # or SqliteRangeReaderFactory() for remote pages
) as reader:
    value = reader.get(b"user-123")
    # Single-shard SQL
    with reader.reader_for_key(b"user-123") as handle:
        cur = handle.reader.connection.cursor()
        cur.execute("SELECT count(*) FROM kv")
```

See [`read-sync-sqlite.md`](read-sync-sqlite.md) for the full reader API and details on the two SQLite reader strategies.

## See also

- [`build-spark-slatedb.md`](build-spark-slatedb.md), [`build-python-sqlite.md`](build-python-sqlite.md).
- [`architecture/adapters.md`](../architecture/adapters.md) — SQLite read strategies.
- [`read-sync-sqlite.md`](read-sync-sqlite.md), [`read-async-sqlite.md`](read-async-sqlite.md).
