# Read a SQLite snapshot asynchronously

Use **`AsyncShardedReader`** with the SQLite async factory to read sharded SQLite snapshots from `asyncio` code. As with the sync variant, the routed reader provides **unified access across all shards**, and you can also reach a single shard directly for SQL queries.

## When to use

- Async service against a SQLite-backed snapshot (any writer flavor: Python / Spark / Dask / Ray).
- You're OK wiring the SQLite async factory explicitly (it's not the default).

## When NOT to use

- SlateDB snapshot — use [`read-async-slatedb.md`](read-async-slatedb.md).
- Synchronous code — use [`read-sync-sqlite.md`](read-sync-sqlite.md).

## Install

```bash
uv add 'shardyfusion[sqlite-async]'
```

## Minimal example

Download-and-cache (the entire shard `.db` is fetched once):

```python
from shardyfusion.reader.async_reader import AsyncShardedReader
from shardyfusion.sqlite_adapter import AsyncSqliteReaderFactory

async def main():
    reader = await AsyncShardedReader.open(
        s3_prefix="s3://my-bucket/snapshots/users",
        local_root="/var/cache/shardy/users",
        reader_factory=AsyncSqliteReaderFactory(),
    )
    async with reader:
        value = await reader.get(b"user-123")
```

Range-read VFS (no full download — pages fetched on demand via S3 range requests):

```python
from shardyfusion.sqlite_adapter import AsyncSqliteRangeReaderFactory

reader = await AsyncShardedReader.open(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
    reader_factory=AsyncSqliteRangeReaderFactory(page_cache_pages=1024),
)
```

## Configuration

Same constructor surface as [`read-async-slatedb.md`](read-async-slatedb.md). The only difference is `reader_factory=AsyncSqliteReaderFactory()` (download-and-cache) or `AsyncSqliteRangeReaderFactory(...)` (remote range-read).

## Reader API

The async SQLite reader exposes the same routed and direct-shard surface as the SlateDB async reader: `get`, `multi_get`, `route_key`, `shard_for_key`, `reader_for_key`, `refresh`, `snapshot_info`, `shard_details`, `health`. See [`read-async-slatedb.md`](read-async-slatedb.md#reader-api) for the call patterns.

For **single-shard SQL access**, borrow a shard handle and use the underlying connection — the same pattern as the sync variant ([`read-sync-sqlite.md`](read-sync-sqlite.md#single-shard-access-sql-scans)). Note that SQLite operations are inherently blocking; the async wrapper hops to a thread for `get`. If you need to issue raw SQL on a borrowed handle, do so in a `to_thread` if it's potentially long-running.

## Functional / Non-functional properties

- **Download-and-cache**: cold-shard fetch is awaited; blocks the awaiting task only. Cached shards are local-disk-fast.
- **Range-read**: per-page S3 GETs awaited; ~zero local disk; works against any S3-compatible store (Garage, MinIO).
- Routed `get` / `multi_get` semantics identical to SlateDB async reader.

## Guarantees

- Snapshot pinning + routing same as other readers — see [`read-sync-slatedb.md`](read-sync-slatedb.md#guarantees).

## Weaknesses

- SQLite async factories (both `AsyncSqliteReaderFactory` and `AsyncSqliteRangeReaderFactory`) are **not exported from the top-level package** — import from `shardyfusion.sqlite_adapter`.
- The async wrappers schedule blocking SQLite calls onto a thread; under very high concurrency tune `max_concurrency` to bound the thread-pool.
- Range-read variant requires `apsw` (LGPL).

## Failure modes & recovery

Same as [`read-sync-sqlite.md`](read-sync-sqlite.md#failure-modes-recovery), surfaced as awaited exceptions.

## See also

- [`read-async-slatedb.md`](read-async-slatedb.md) — full async reader API surface (shared with this page).
- [`read-sync-sqlite.md`](read-sync-sqlite.md) — sync equivalent and single-shard SQL access.
- [`architecture/adapters.md`](../architecture/adapters.md) — SQLite adapter internals.
