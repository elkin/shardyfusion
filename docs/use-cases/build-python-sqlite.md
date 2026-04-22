# Build a SQLite snapshot with the Python writer

Use the **Python writer** to build a sharded **SQLite** snapshot from any Python iterable. Same flow as [`build-python-slatedb.md`](build-python-slatedb.md), but each shard is a standalone `.db` file.

## When to use

- You want SQL queries on each shard at read time (download-and-cache reader).
- You want page-level lazy access over HTTP range requests (range-read reader).
- You want one self-contained file per shard for offline distribution.
- Reader environment can install `sqlite3` or `apsw`.

## When NOT to use

- You want SlateDB's append-only LSM characteristics — use [`build-python-slatedb.md`](build-python-slatedb.md).
- You also need vector search on the same shard — use [`build-python-sqlite-vec.md`](build-python-sqlite-vec.md) (unified KV+vector).

## Install

```bash
uv add 'shardyfusion[writer-python-sqlite]'
```

## Minimal example

```python
from shardyfusion import WriteConfig
from shardyfusion.sqlite_adapter import SqliteFactory
from shardyfusion.writer.python import write_sharded

records = [{"id": i, "payload": f"row-{i}".encode()} for i in range(10_000)]

config = WriteConfig(
    num_dbs=4,
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    adapter_factory=SqliteFactory(),    # <-- the only difference vs SlateDB
)

result = write_sharded(
    records,
    config,
    key_fn=lambda r: r["id"],
    value_fn=lambda r: r["payload"],
)
print(result.manifest_ref.ref)
```

`SqliteFactory` lives at `shardyfusion/sqlite_adapter.py:59`. It is **not** in the public `__all__` — import it from its module.

## Configuration

Same `WriteConfig` fields as [`build-python-slatedb.md`](build-python-slatedb.md), with one swap:

| Field | Value |
|---|---|
| `adapter_factory` | `SqliteFactory()` (instead of `None` / `SlateDbFactory()`) |

`SqliteFactory()` has no required arguments. It accepts S3 connection options that get propagated to the upload step.

## Functional properties

- Each shard is a single `.db` file uploaded to `s3://<prefix>/shards/db=NNNNN/...`.
- Writes go through the standard `sqlite3` driver locally, then the entire file is uploaded on `checkpoint()`.
- KV pairs are stored in a `kv(key BLOB PRIMARY KEY, value BLOB)` schema (see `sqlite_adapter.py`).

## Non-functional properties

- **File size**: each shard is a complete SQLite database; no compression beyond what SQLite applies.
- **Upload cost**: one PUT per shard (multipart for large shards).
- **Local disk**: each shard staged at `output.local_root / db=NNNNN/` before upload. Plan disk for `num_dbs × shard_size`.

## Guarantees

- Same as [`build-python-slatedb.md`](build-python-slatedb.md): atomic two-phase publish, deterministic sharding, manifest-pinned reads.

## Weaknesses

- **Whole-file rewrite per attempt**: SQLite shards don't support incremental publishing. A retry uploads the entire shard again.
- **No async writer**: the Python writer is synchronous. Async is a read-side feature (see [`read-async-sqlite.md`](read-async-sqlite.md)).

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| `sqlite3.OperationalError` (disk full, corruption) | wrapped in `SqliteAdapterError` | Investigate local disk; rerun. |
| Upload PUT fails | `S3TransientError` (auto-retry) or terminal `PublishManifestError` | Auto-retried; manual rerun if terminal. |
| Mid-shard process crash | `ShardWriteError`, then `ShardCoverageError` if no successful attempt | Set `config.shard_retry`, rerun. |

See [`architecture/error-model.md`](../architecture/error-model.md) for the full hierarchy.

## Reading the snapshot

The reader is **completely decoupled** from the writer flavor — a snapshot built here reads identically to one built by Spark/Dask/Ray + SQLite. Two reader strategies are available for SQLite:

**Unified routed access (across all shards):**

```python
from shardyfusion import ShardedReader
from shardyfusion.sqlite_adapter import SqliteReaderFactory

with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteReaderFactory(),  # download-and-cache
) as reader:
    print(reader.get(b"user-123"))                # routed lookup, picks the right shard
    print(reader.multi_get([b"u-1", b"u-2"]))     # batched, deduplicated by shard
```

**Remote shard access (no full download — pages fetched via S3 range requests):**

```python
from shardyfusion.sqlite_adapter import SqliteRangeReaderFactory

reader = ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users-sqlite",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteRangeReaderFactory(page_cache_pages=1024),
)
```

Each `.db` file lives in S3; the range-read VFS makes it usable in place — no full download, ~zero local disk.

**Single-shard access (SQL on one shard only):**

You can also reach a single shard directly — useful when you want SQL on just the shard a key lives in, or want to operate on a known shard:

```python
# Borrow the SQLite connection for the shard a key routes to
with reader.reader_for_key(b"user-123") as handle:
    cur = handle.reader.connection.cursor()
    cur.execute("SELECT count(*) FROM kv")
    print(cur.fetchone())

# Or enumerate shards and pick by db_id
for shard in reader.shard_details():
    print(shard.db_id, shard.db_url, shard.row_count)
```

The `db_url` is a stable S3 URL — you can also download it independently or open it in any SQLite tool. See [`read-sync-sqlite.md`](read-sync-sqlite.md) for the full reader API surface.

## See also

- [`architecture/adapters.md`](../architecture/adapters.md) — SQLite adapter internals (download-and-cache vs range-read).
- [`build-python-slatedb.md`](build-python-slatedb.md) — same flow with SlateDB.
- [`read-sync-sqlite.md`](read-sync-sqlite.md) — read-side details for both SQLite reader strategies.
- [`build-python-sqlite-vec.md`](build-python-sqlite-vec.md) — adds vector search to the same shards.
