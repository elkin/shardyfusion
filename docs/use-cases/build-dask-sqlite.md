# Build a SQLite snapshot with the Dask writer

Same as [`build-dask-slatedb.md`](build-dask-slatedb.md), but each shard is a SQLite `.db` file.

## When to use

- Dask pipeline + SQLite read side.

## When NOT to use

- Want SlateDB shards — [`build-dask-slatedb.md`](build-dask-slatedb.md).

## Install

```bash
uv add 'shardyfusion[writer-dask-sqlite]'
```

## Minimal example

```python
import dask.dataframe as dd
from shardyfusion import WriteConfig
from shardyfusion.sqlite_adapter import SqliteFactory
from shardyfusion.writer.dask import write_sharded
from shardyfusion.serde import ValueSpec

ddf = dd.read_parquet("s3://lake/users/")

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    adapter_factory=SqliteFactory(),
)

write_sharded(ddf, config, key_col="id",
              value_spec=ValueSpec.binary_col("payload"))
```

## Configuration / Functional / Non-functional / Guarantees / Failure modes

Inherits from [`build-dask-slatedb.md`](build-dask-slatedb.md). SQLite-specific costs and failures from [`build-python-sqlite.md`](build-python-sqlite.md) apply per shard.

## Reading the snapshot

The reader is **decoupled from the writer flavor** — no Dask dependency at read time. The reader provides **unified routed access across all SQLite shards**, with a choice between **download-and-cache** and **remote range-read VFS** (no full download). You can also reach a single shard for raw SQL.

```python
from shardyfusion import ShardedReader
from shardyfusion.sqlite_adapter import SqliteReaderFactory  # or SqliteRangeReaderFactory

with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteReaderFactory(),
) as reader:
    print(reader.get(b"user-123"))
    with reader.reader_for_key(b"user-123") as handle:
        cur = handle.reader.connection.cursor()
        cur.execute("SELECT count(*) FROM kv")
```

See [`read-sync-sqlite.md`](read-sync-sqlite.md) for the full reader API and the two SQLite reader strategies.

## See also

- [`build-dask-slatedb.md`](build-dask-slatedb.md).
- [`read-sync-sqlite.md`](read-sync-sqlite.md), [`read-async-sqlite.md`](read-async-sqlite.md).
- [`architecture/adapters.md`](../architecture/adapters.md).
