# Read a SlateDB snapshot asynchronously

Use **`AsyncShardedReader`** to do point-key and multi-key lookups against a published SlateDB snapshot from `asyncio` code.

## When to use

- You're in `asyncio` code (FastAPI, async workers).
- SlateDB-backed snapshot.

## When NOT to use

- Synchronous code — use [sync SlateDB](../sync/slatedb.md).
- SQLite snapshot — use [async SQLite](sqlite.md).

## Install

```bash
uv add 'shardyfusion[read-slatedb-async]'
```

Pulls SlateDB plus `aiobotocore`.

## Minimal example

```python
from shardyfusion.reader.async_reader import AsyncShardedReader

async def main():
    reader = await AsyncShardedReader.open(
        s3_prefix="s3://my-bucket/snapshots/users",
        local_root="/var/cache/shardy/users",
    )
    try:
        value = await reader.get(b"user-123")
        many = await reader.multi_get([b"a", b"b", b"c"])
    finally:
        await reader.close()
```

Or `async with`:

```python
async with await AsyncShardedReader.open(...) as reader:
    value = await reader.get(b"user-123")
```

## Configuration

`AsyncShardedReader.__init__` (`shardyfusion/reader/async_reader.py:240`):

| Param | Default | Purpose |
|---|---|---|
| `s3_prefix` | required | Snapshot root. |
| `local_root` | required | Local cache directory. |
| `manifest_store` | auto | Read-only async store. |
| `current_pointer_key` | `"_CURRENT"` | Pointer. |
| `reader_factory` | `AsyncSlateDbReaderFactory()` | Adapter factory (SlateDB only by default). |
| `max_concurrency` | `None` | Optional `asyncio.Semaphore` limit on `multi_get`. |
| `max_fallback_attempts` | `3` | Retries on transient failures. |
| `metrics_collector` | `None` | Observability. |
| `rate_limiter` | `None` | Token-bucket. |

State must be loaded via `await AsyncShardedReader.open(...)` — direct `__init__` does not load the manifest.

## Reader API

The async reader mirrors the sync reader API:

```python
# Lookups
value: bytes | None = await reader.get(b"user-123")
results: dict = await reader.multi_get([b"a", b"b", b"c"])

# Routing introspection (sync — pure local computation)
db_id: int = reader.route_key(b"user-123")
meta = reader.shard_for_key(b"user-123")

# Direct shard access — borrow the underlying async shard handle
async with reader.reader_for_key(b"user-123") as handle:
    raw = await handle.reader.get(reader.encode_key(b"user-123"))

# Snapshot inspection (sync)
info = reader.snapshot_info()
shards = reader.shard_details()
health = reader.health()

# Refresh / close (async)
changed: bool = await reader.refresh()
await reader.close()
```

`reader_for_key` returns an `AsyncShardReaderHandle` — use it with `async with` so the reader can release its borrow counter.

## Functional properties

- `multi_get` uses `asyncio.TaskGroup` and an optional `Semaphore`.
- `refresh()` is async; same explicit-refresh model as the sync reader.

## Guarantees

- Reads pinned to manifest at last `open`/`refresh`.
- Routing matches writer.
- Same cold-start fallback as sync reader (up to `max_fallback_attempts` previous manifests).

## Weaknesses

- Default `AsyncSlateDbReaderFactory` is **SlateDB-only**. SQLite async factory exists in `sqlite_adapter.py` but must be wired explicitly. See [async SQLite](sqlite.md).
- `AsyncManifestStore` is read-only; publishing must go through the sync writer path.

## Failure modes & recovery

Same matrix as [sync SlateDB](../sync/slatedb.md), surfaced as awaited exceptions.

## See also

- [KV Storage Overview](../../overview.md)
- [Sync SlateDB](../sync/slatedb.md) — canonical reference for shared API
- [Async SQLite](sqlite.md)
- [`architecture/manifest-and-current.md`](../../../../architecture/manifest-and-current.md)
