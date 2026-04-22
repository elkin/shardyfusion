# Read a KV+vector snapshot asynchronously

Use **`AsyncUnifiedShardedReader`** for both point-key lookups and vector nearest-neighbor search from `asyncio` code.

## When to use

- You need both `get(key)` and `search(query_vector, top_k)` in `asyncio` code.

## When NOT to use

- Synchronous code — use [sync KV+Vector reader](sync.md) (`UnifiedShardedReader`).
- Vector-only — use [async vector reader](../../vector/read/async.md) (`AsyncShardedVectorReader`).

## Install

```bash
# For composite (LanceDB) snapshots
uv add 'shardyfusion[unified-vector,read-async]'

# For unified (sqlite-vec) snapshots
uv add 'shardyfusion[unified-vector-sqlite,read-async]'
```

## Minimal example

```python
from shardyfusion.reader.async_unified_reader import AsyncUnifiedShardedReader
import numpy as np

async def main():
    reader = await AsyncUnifiedShardedReader.open(
        s3_prefix="s3://my-bucket/snapshots/items",
        local_root="/tmp/unified",
    )

    # KV lookup
    val = await reader.get(b"item-123")

    # Vector search
    query = np.random.randn(384).astype(np.float32)
    results = await reader.search(query, top_k=10)

    await reader.close()
```

## Configuration

`AsyncUnifiedShardedReader.open` (`shardyfusion/reader/async_unified_reader.py:112`):

| Param | Default | Purpose |
|---|---|---|
| `s3_prefix` | required | Snapshot root. |
| `local_root` | required | Local cache directory. |
| `max_concurrency` | `None` | `asyncio.Semaphore` for concurrent operations. |
| `max_fallback_attempts` | `3` | Fallback to previous manifests. |

## Reader API

```python
# KV lookups
value = await reader.get(b"item-123")
many = await reader.multi_get([b"item-1", b"item-2"])

# Vector search
results = await reader.search(query_vector, top_k=10)

# Snapshot inspection (sync)
info = reader.snapshot_info()
shards = reader.shard_details()

# Refresh / lifecycle (async)
changed = await reader.refresh()
await reader.close()
```

## Functional properties

- Extends `AsyncShardedReader` with async vector search.
- Auto-dispatches backend based on `manifest.vector.backend`.
- `search()` fans out across shards using `asyncio.TaskGroup`.

## Guarantees

- Same as `AsyncShardedReader`: reads pinned to manifest, no partial views.
- `search()` and `get()` see the same snapshot atomically.

## Weaknesses

- `AsyncUnifiedShardedReader` is loaded via top-level `__getattr__`.
- Not re-exported at top level in all configurations — import from `shardyfusion.reader.async_unified_reader` if needed.

## Failure modes & recovery

Same matrix as [sync KV+Vector reader](sync.md), surfaced as awaited exceptions.

## See also

- [KV+Vector Overview](../overview.md)
- [Sync KV+Vector reader](sync.md)
- [Async vector reader](../../vector/read/async.md)
