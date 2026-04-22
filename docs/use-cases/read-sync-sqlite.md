# Read a SQLite snapshot synchronously

Use **`ShardedReader`** with a SQLite reader factory to read sharded SQLite snapshots synchronously. Two sub-flavors selected by the factory:

- **Download-and-cache** (`SqliteReaderFactory`) — the entire shard `.db` is downloaded once, then all reads are local.
- **Range-read VFS** (`SqliteRangeReaderFactory`, requires `apsw`) — SQLite issues HTTP range GETs against the S3 object; **no full download required**, the file effectively lives remotely.

The reader gives you **unified routed access across all shards** (just call `.get(key)` and it picks the right shard) — but you can also reach a **single shard directly** for SQL queries or scans. See [Single-shard access](#single-shard-access-sql-scans) below.

## When to use

- Snapshot was built with the SQLite adapter (any writer flavor: Python / Spark / Dask / Ray).
- You're in synchronous code.
- Pick **range-read** when shards are large and you only touch a few keys per shard, or when you want to avoid disk usage proportional to total snapshot size.
- Pick **download-and-cache** when shards are small and you want the lowest per-key latency once warm.

## When NOT to use

- SlateDB snapshot — use [`read-sync-slatedb.md`](read-sync-slatedb.md).
- Async code — use [`read-async-sqlite.md`](read-async-sqlite.md).

## Install

```bash
# Download-and-cache (uses stdlib sqlite3)
uv add 'shardyfusion[read-sqlite]'

# Range-read VFS (no full download; requires apsw)
uv add 'shardyfusion[read-sqlite-range]'
```

## Minimal example

```python
from shardyfusion import ShardedReader
from shardyfusion.sqlite_adapter import SqliteReaderFactory  # or SqliteRangeReaderFactory

reader = ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteReaderFactory(),  # download-and-cache
)

value = reader.get(b"user-123")
many = reader.multi_get([b"user-1", b"user-2"])
```

For **range-read** (remote shard access without download), swap the factory:

```python
from shardyfusion.sqlite_adapter import SqliteRangeReaderFactory

reader = ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",  # used only for tiny manifest cache
    reader_factory=SqliteRangeReaderFactory(page_cache_pages=1024),  # ~4 MiB LRU
)
```

The range-read VFS translates each SQLite page fetch (default 4 KiB) into an S3 `Range` request, with an LRU page cache in memory.

## Configuration

The reader constructor and full API surface (`get`, `multi_get`, `route_key`, `shard_for_key`, `reader_for_key`, `refresh`, `snapshot_info`, `shard_details`, `health`) are identical to the SlateDB variant — see [`read-sync-slatedb.md`](read-sync-slatedb.md#reader-api). The only thing that changes is the `reader_factory`.

`SqliteRangeReaderFactory` extra parameters:

| Field | Default | Purpose |
|---|---|---|
| `page_cache_pages` | `1024` | Page-cache LRU size (4 KiB per page → ~4 MiB). `0` disables caching. |
| `s3_connection_options` | `None` | Endpoint URL, region, retries — passed to the VFS. |
| `credential_provider` | `None` | S3 credentials. |

## Single-shard access (SQL & scans)

The unified routed reader is convenient for KV lookups, but SQLite shards are real SQLite databases — you can run arbitrary SQL on a single shard. Use `reader_for_key` to borrow that shard's underlying handle:

```python
with reader.reader_for_key(b"user-123") as handle:
    # `handle.reader` exposes the underlying SQLite shard reader.
    # Both download-and-cache and range-read expose a `.connection` property
    # giving access to the live sqlite3 / apsw connection.
    cur = handle.reader.connection.cursor()
    cur.execute("SELECT count(*) FROM kv WHERE k LIKE ?", (b"user-%",))
    print(cur.fetchone())
```

You can also enumerate shards from the manifest and connect to one explicitly:

```python
for shard in reader.shard_details():
    print(shard.db_id, shard.db_url, shard.row_count)

# Or look up the shard for a specific key without opening it via the reader:
meta = reader.shard_for_key(b"user-123")
print(meta.db_url)   # s3://my-bucket/snapshots/users/shards/db=00003/run_id=.../...db
```

The `db_url` in `shard_details()` is a stable S3 URL pointing at the `.db` file for that shard in the *currently pinned* manifest. You can:

- Download it independently (`aws s3 cp` or any S3 client).
- Open it directly in any SQLite tool (DBeaver, `sqlite3` CLI).
- Open it remotely via the same range-read VFS without going through `ShardedReader` — instantiate `SqliteRangeReaderFactory()(db_url=..., local_dir=..., checkpoint_id=...)` directly.

## Functional properties

- **Download-and-cache**: first access pulls the whole shard `.db` to `local_root`; subsequent reads are local-disk-fast.
- **Range-read VFS**: per-page S3 GETs with an LRU cache; cold-cache latency higher but no full-shard download. Works against any S3-compatible store (Garage, MinIO, etc.).
- **Same routed API** as SlateDB (`get`, `multi_get`, `refresh`, etc. — see [`read-sync-slatedb.md`](read-sync-slatedb.md#reader-api)).
- **Direct SQL** on any shard via `reader_for_key().reader.connection`.

## Non-functional properties

- **Disk usage**:
  - Download-and-cache: `sum(shard_size)` once warm.
  - Range-read: ~`page_cache_pages × 4 KiB` of memory; ~zero disk.
- **Latency**:
  - Download-and-cache: cold = full shard download; warm = local-disk SQLite.
  - Range-read: cold per-page = one S3 GET (~tens of ms); warm pages = memory.
- **S3 GET cost**: range-read amplifies request count when scanning; prefer download-and-cache for full-shard scans.

## Guarantees

- Same routing/snapshot pinning guarantees as SlateDB. See [`read-sync-slatedb.md`](read-sync-slatedb.md#guarantees).
- Both reader strategies see the same `.db` file — the byte content is identical; only the access path differs.

## Weaknesses

- Download-and-cache: cold-shard latency proportional to shard size; disk pressure proportional to total snapshot size.
- Range-read: higher per-key latency, more S3 GETs, requires `apsw` (LGPL).
- SQLite has no built-in compression — shard size = raw `(key, value)` bytes plus index overhead.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Disk full (download-and-cache) | `DbAdapterError` | Free disk or switch to range-read. |
| Range-read 416 / connection drop | `DbAdapterError` (wrapped `S3VfsError`) | Retry; check S3 connectivity. |
| Missing `_CURRENT` / malformed manifest | Same as SlateDB — see [`read-sync-slatedb.md`](read-sync-slatedb.md#failure-modes-recovery). |
| `apsw` not installed for range-read | `ImportError` at factory call | `uv add 'shardyfusion[read-sqlite-range]'`. |

## See also

- [`read-sync-slatedb.md`](read-sync-slatedb.md) — full reader API surface (shared with this page).
- [`read-async-sqlite.md`](read-async-sqlite.md) — async equivalent.
- [`architecture/adapters.md`](../architecture/adapters.md) — SQLite adapter internals (download-and-cache vs range-read VFS).
