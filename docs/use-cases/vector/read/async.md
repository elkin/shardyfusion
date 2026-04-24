# Read a vector snapshot asynchronously

Use **`AsyncShardedVectorReader`** for approximate nearest-neighbor (ANN) search across a sharded vector snapshot from `asyncio` code.

## When to use

- Pure vector workload in `asyncio` code (FastAPI, aiohttp).

## When NOT to use

- Synchronous code — use [sync vector reader](sync.md) (`ShardedVectorReader`).
- You also need KV lookups — use [KV+Vector async reader](../../kv-vector/read/async.md) (`AsyncUnifiedShardedReader`).

## Install

```bash
# LanceDB backend
uv add 'shardyfusion[vector-lancedb,read-async]'

# sqlite-vec backend
uv add 'shardyfusion[vector-sqlite,read-async]'
```

## Minimal example

```python
from shardyfusion.vector.async_reader import AsyncShardedVectorReader
import numpy as np

async def main():
    reader = await AsyncShardedVectorReader.open(
        s3_prefix="s3://my-bucket/snapshots/embeddings",
        local_root="/tmp/vectors",
    )

    query = np.random.randn(384).astype(np.float32)
    results = await reader.search(query, top_k=10)

    for res in results:
        print(res.id, res.score, res.payload)

    await reader.close()
```

## Configuration

`AsyncShardedVectorReader.open` (`shardyfusion/vector/async_reader.py:53`):

| Param | Default | Purpose |
|---|---|---|
| `s3_prefix` | required | Snapshot root. |
| `local_root` | required | Local cache directory. |
| `manifest_store` | auto | Async manifest store. |
| `max_concurrency` | `None` | `asyncio.Semaphore` for concurrent shard searches. |
| `max_fallback_attempts` | `3` | Fallback to previous manifests. |
| `rate_limiter` | `None` | Token-bucket rate limit. |

## Reader API

```python
# ANN search
results = await reader.search(
    query_vector,
    top_k=10,
    shard_ids=None,
    num_probes=None,
    routing_context=None,
)

# Snapshot inspection (sync)
info = reader.snapshot_info()
shards = reader.shard_details()
health = reader.health()

# Refresh / lifecycle (async)
changed = await reader.refresh()
await reader.close()
```

## Functional properties

- `search` fans out across target shards using `asyncio.TaskGroup`.
- Same lazy shard loading and LRU eviction as the sync variant.

## Guarantees

- Reads pinned to manifest at last `open`/`refresh`.
- Routing matches writer.
- Same fallback behavior as KV readers.

## Weaknesses

- `AsyncShardedVectorReader` is **not re-exported** at top level — import from `shardyfusion.vector`.
- Same vector-specific limits as the sync variant.

## Failure modes & recovery

Same matrix as [sync vector reader](sync.md), surfaced as awaited exceptions.

## See also

- [Vector Overview](../overview.md)
- [Sync vector reader](sync.md)
- [KV+Vector async reader](../../kv-vector/read/async.md)
