# Build a vector-only snapshot (LanceDB)

Use the **standalone vector writer** with the LanceDB backend to build a sharded vector index — no KV side.

## When to use

- Pure ANN workload — no point-key lookups needed.
- You want LanceDB's IVF/HNSW tuning and full metric set (`cosine`, `l2`, `dot_product`).

## When NOT to use

- You also need KV lookups — use [KV+Vector composite](../../kv-vector/build/composite.md) or [unified](../../kv-vector/build/unified.md).
- You want a single-file SQLite-based vector store — use [sqlite-vec](sqlite-vec.md).

## Install

```bash
uv add 'shardyfusion[vector-lancedb]'
```

## Minimal example

```python
from shardyfusion.vector import (
    VectorRecord, VectorShardedWriteConfig, write_sharded,
)
from shardyfusion.vector.config import VectorIndexConfig, VectorShardingConfig
from shardyfusion.vector.adapters.lancedb_adapter import LanceDbFactory
from shardyfusion.vector.types import DistanceMetric

records = [
    VectorRecord(id="a", vector=[0.1, 0.2, ...], payload={"category": "x"}),
    # ...
]

config = VectorShardedWriteConfig(
    sharding=VectorShardingConfig(num_dbs=16),
    s3_prefix="s3://my-bucket/snapshots/embeddings",
    index_config=VectorIndexConfig(dim=384, metric=DistanceMetric.COSINE),
    adapter_factory=LanceDbFactory(),
)

result = write_sharded(records, config)
```

## Configuration

- `VectorShardedWriteConfig(index_config, sharding=VectorShardingConfig(num_dbs=...), storage=..., adapter=..., rate_limits=...)` at `vector/config.py`.
- `VectorIndexConfig(dim, metric, ...)` — `metric ∈ {cosine, l2, dot_product}` for LanceDB.
- Sharding strategies: `CLUSTER` (default; k-means), `LSH`, `EXPLICIT` (use `VectorRecord.shard_id`), `CEL` (route on `routing_context`).

## Functional properties

- One LanceDB table per shard.
- Atomic two-phase publish (vector manifest + `_CURRENT`).

## Guarantees

- Successful return ⇒ vector manifest published, all shards queryable via `ShardedVectorReader`.

## Weaknesses

- `write_sharded` and `ShardedVectorReader` are **not re-exported** at top level — import from `shardyfusion.vector`.
- `CLUSTER` sharding requires sampling pass over the data.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Dim mismatch in record | `ConfigValidationError` | Filter or fix records. |
| LanceDB index build fails | `VectorIndexError` | Check disk; rerun. |
| Shard cluster too small | `VectorIndexError` (LanceDB IVF requires N ≥ 256 by default) | Reduce `num_dbs` or skip IVF training. |

## Distributed engines

If your vectors already live in a Spark, Dask, or Ray dataset, use the distributed vector writers instead of the Python iterator-based writer:

- **[Spark → vector](spark.md)** — `write_sharded(df, config, VectorColumnInput(...))`
- **[Dask → vector](dask.md)** — `write_sharded(ddf, config, VectorColumnInput(...))`
- **[Ray → vector](ray.md)** — `write_sharded(ds, config, VectorColumnInput(...))`

Distributed writers accept `VectorShardedWriteConfig` plus `VectorColumnInput` and shard directly from the dataframe/dataset without collecting everything into the driver first.

## See also

- [Vector Overview](../overview.md) — routing strategies, scatter-gather flow
- [sqlite-vec](sqlite-vec.md) — SQLite-based alternative
- [Read → Sync](../read/sync.md) — `ShardedVectorReader`
- [Read → Async](../read/async.md) — `AsyncShardedVectorReader`
