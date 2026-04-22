# Build a SQLite snapshot with the Ray writer

Use the **Ray writer** with the SQLite adapter to build a sharded SQLite snapshot from a `ray.data.Dataset`.

## When to use

- Records live in a Ray Dataset.
- You want SQLite shards (single-file `.sqlite` per shard, downloadable or range-readable).
- You want SQLite-side reads (cache or range-read VFS).

## When NOT to use

- You want SlateDB shards — use [`build-ray-slatedb.md`](build-ray-slatedb.md).

## Install

```bash
uv add 'shardyfusion[writer-ray-sqlite]'
```

Pulls `ray[data]>=2.20` plus the SQLite writer factory.

## Minimal example

```python
import ray
from shardyfusion import WriteConfig
from shardyfusion.writer.ray import write_sharded
from shardyfusion.serde import ValueSpec
from shardyfusion.sqlite_adapter import SqliteFactory

ray.init()
ds = ray.data.read_parquet("s3://lake/users/")

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users",
    adapter_factory=SqliteFactory(),
)

result = write_sharded(
    ds,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)
```

## Configuration

Identical to [`build-ray-slatedb.md`](build-ray-slatedb.md) for Ray-side knobs. SQLite-side knobs live on `SqliteFactory(page_size=4096, cache_size_pages=-2000, journal_mode="OFF", synchronous="OFF", temp_store="MEMORY", mmap_size=0)`.

## Functional / Non-functional properties

- One `.sqlite` file per shard, sealed and uploaded after `checkpoint()`.
- Atomic two-phase publish.
- Same process-global shuffle-strategy swap as the SlateDB variant.

## Guarantees

- Successful return ⇒ manifest + `_CURRENT` published.
- File is fully written and `fsync`'d before upload.

## Weaknesses

- Whole-file upload per shard ⇒ shard size is bounded by available memory/disk during build.
- Same process-global Ray Data context mutation as the SlateDB variant.

## Failure modes & recovery

Same matrix as [`build-ray-slatedb.md`](build-ray-slatedb.md). Add: SQLite-specific I/O errors surface as `DbAdapterError`.

## Reading the snapshot

The reader is **decoupled from the writer flavor** — no Ray dependency at read time. The reader provides **unified routed access across all SQLite shards** via `ShardedReader` + a SQLite reader factory. SQLite shards can be accessed locally (download-and-cache) or **remotely via S3 range requests** (no full download). You can also reach a single shard for raw SQL.

```python
from shardyfusion import ShardedReader
from shardyfusion.sqlite_adapter import SqliteReaderFactory  # or SqliteRangeReaderFactory

with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteReaderFactory(),
) as reader:
    print(reader.get(b"user-123"))
    with reader.reader_for_key(b"user-123") as handle:
        cur = handle.reader.connection.cursor()
        cur.execute("SELECT count(*) FROM kv")
```

See [`read-sync-sqlite.md`](read-sync-sqlite.md) for the full reader API and SQLite reader strategies.

## See also

- [`build-ray-slatedb.md`](build-ray-slatedb.md).
- [`read-sync-sqlite.md`](read-sync-sqlite.md).
- [`architecture/adapters.md`](../architecture/adapters.md).
