# Build a vector-only snapshot (LanceDB, standalone)

Use the **standalone vector writer** with the LanceDB backend to build a sharded vector index — no KV side.

## When to use

- Pure ANN workload — no point-key lookups needed.
- You want LanceDB's IVF/HNSW tuning and full metric set (`cosine`, `l2`, `dot_product`).

## When NOT to use

- You also need KV lookups — use [`build-python-slatedb-lancedb.md`](build-python-slatedb-lancedb.md).
- You want a single-file SQLite-based vector store — use [`build-vector-sqlite-vec-standalone.md`](build-vector-sqlite-vec-standalone.md).

## Install

```bash
uv add 'shardyfusion[vector-lancedb]'
```

## Minimal example

```python
from shardyfusion.vector import (
    VectorRecord, VectorWriteConfig, write_vector_sharded,
)
from shardyfusion.vector.config import VectorIndexConfig
from shardyfusion.vector.adapters.lancedb_adapter import LanceDbFactory

records = [
    VectorRecord(id="a", vector=[0.1, 0.2, ...], payload={"category": "x"}),
    # ...
]

config = VectorWriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/embeddings",
    index_config=VectorIndexConfig(dim=384, metric="cosine"),
    adapter_factory=LanceDbFactory(),
)

result = write_vector_sharded(records, config)
```

## Configuration

- `VectorWriteConfig(num_dbs, s3_prefix, index_config, sharding=VectorShardingSpec.cluster(), adapter_factory, batch_size=10_000, ...)` at `vector/config.py:74`.
- `VectorIndexConfig(dim, metric, ...)` — `metric ∈ {cosine, l2, dot_product}` for LanceDB.
- Sharding strategies: `CLUSTER` (default; k-means), `LSH`, `EXPLICIT` (use `VectorRecord.shard_id`), `CEL` (route on `routing_context`).

## Functional / Non-functional properties

- One LanceDB table per shard.
- Atomic two-phase publish (vector manifest + `_CURRENT`).

## Guarantees

- Successful return ⇒ vector manifest published, all shards queryable via `ShardedVectorReader`.

## Weaknesses

- `write_vector_sharded` and `ShardedVectorReader` are **not re-exported** at top level — import from `shardyfusion.vector`.
- `CLUSTER` sharding requires sampling pass over the data.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Dim mismatch in record | `ConfigValidationError` | Filter or fix records. |
| LanceDB index build fails | `VectorIndexError` | Check disk; rerun. |
| Shard cluster too small | `VectorIndexError` (LanceDB IVF requires N ≥ 256 by default) | Reduce `num_dbs` or skip IVF training. |

## See also

- [`build-vector-sqlite-vec-standalone.md`](build-vector-sqlite-vec-standalone.md).
- [`build-python-slatedb-lancedb.md`](build-python-slatedb-lancedb.md).
- [`architecture/sharding.md`](../architecture/sharding.md).
