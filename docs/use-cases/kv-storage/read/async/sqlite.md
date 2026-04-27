# Read a SQLite snapshot asynchronously

Use **`AsyncShardedReader`** with an async SQLite reader factory for async lookups against sharded SQLite snapshots.

## When to use

- You're in `asyncio` code.
- Snapshot was built with the SQLite adapter.

## When NOT to use

- Synchronous code — use [sync SQLite](../sync/sqlite.md).
- SlateDB snapshot — use [async SlateDB](slatedb.md).

## Install

```bash
# Async SQLite wrappers (download-and-cache)
uv add 'shardyfusion[sqlite-async]'

# If you also need range-read VFS
uv add 'shardyfusion[sqlite-async]' 'shardyfusion[read-sqlite-range]'

# Adaptive (auto-pick download or range per snapshot — recommended).
# Pulls aiobotocore + both backends so AsyncAdaptiveSqliteReaderFactory works
# without a follow-up install.
uv add 'shardyfusion[sqlite-adaptive-async]'
```

## Minimal example

```python
from shardyfusion.reader.async_reader import AsyncShardedReader
from shardyfusion.sqlite_adapter import AsyncSqliteReaderFactory

async def main():
    reader = await AsyncShardedReader.open(
        s3_prefix="s3://my-bucket/snapshots/users",
        local_root="/var/cache/shardy/users",
        reader_factory=AsyncSqliteReaderFactory(),
    )
    value = await reader.get(b"user-123")
    await reader.close()
```

## Configuration

Same constructor as [async SlateDB](slatedb.md), with `reader_factory=AsyncSqliteReaderFactory()` or `AsyncSqliteRangeReaderFactory()`.

The API surface (`get`, `multi_get`, `route_key`, `shard_for_key`, `reader_for_key`, `refresh`, `snapshot_info`, `shard_details`, `health`) is identical to the SlateDB async variant.

## Functional properties

- Download-and-cache: first access pulls the whole shard `.db`; subsequent reads are local.
- Range-read VFS: per-page S3 GETs with LRU cache (if `apsw` is available).
- Same routed API as SlateDB.

## Guarantees

- Same routing/snapshot pinning guarantees as SlateDB. See [sync SlateDB](../sync/slatedb.md#guarantees).

## Weaknesses

- Async SQLite factories are not imported by `async_reader.py` by default — you must wire them explicitly.
- Same SQLite-specific limits as the sync variant: no built-in compression, whole-file download for download-and-cache.

## Failure modes & recovery

Same matrix as [sync SQLite](../sync/sqlite.md), surfaced as awaited exceptions.

## See also

- [KV Storage Overview](../../overview.md)
- [Sync SQLite](../sync/sqlite.md) — full details on download-and-cache vs range-read
- [Async SlateDB](slatedb.md)
- [`architecture/adapters.md`](../../../../architecture/adapters.md)
