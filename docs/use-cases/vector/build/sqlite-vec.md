# Build a vector-only snapshot (sqlite-vec)

Use the **standalone vector writer** with the sqlite-vec backend to build a sharded vector index where each shard is one SQLite file.

## When to use

- Pure ANN workload, single-file shards preferred.
- `cosine` or `l2` are sufficient.
- You want minimal operational surface (no LanceDB native deps).

## When NOT to use

- You need `dot_product` — use [LanceDB](lancedb.md).
- You also need KV — use [KV+Vector unified](../../kv-vector/build/unified.md).

## Install

```bash
uv add 'shardyfusion[vector-sqlite]'
```

## Minimal example

```python
from shardyfusion.vector import (
    VectorRecord, VectorShardedWriteConfig, write_sharded,
)
from shardyfusion.vector.config import VectorIndexConfig, VectorShardingConfig
from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy
from shardyfusion import VectorSpec

vector_spec = VectorSpec(dim=384, metric="cosine")

config = VectorShardedWriteConfig(
    sharding=VectorShardingConfig(
        num_dbs=16,
        strategy=VectorShardingStrategy.CLUSTER,
        train_centroids=True,
    ),
    s3_prefix="s3://my-bucket/snapshots/embeddings",
    index_config=VectorIndexConfig(dim=384, metric=DistanceMetric.COSINE),
    adapter_factory=SqliteVecFactory(vector_spec=vector_spec),
)

result = write_sharded(records, config)
```

## Configuration

- Allowed metrics: `cosine`, `l2`. `dot_product` rejected.
- Same `VectorShardingConfig` strategies as the LanceDB variant.

## Functional properties

- One `.sqlite` file per shard (vector data only — no KV table).
- Atomic two-phase publish.

## Guarantees

- Successful return ⇒ all shards queryable via `ShardedVectorReader`.

## Weaknesses

- No `dot_product`.
- No IVF/HNSW tuning surface.

## Failure modes & recovery

Same as [LanceDB](lancedb.md), minus IVF cluster-size errors.

## Distributed engines

If your vectors already live in a Spark, Dask, or Ray dataset, use the distributed vector writers instead of the Python iterator-based writer:

- **[Spark → vector](spark.md)** — `write_sharded(df, config, VectorColumnInput(...))`
- **[Dask → vector](dask.md)** — `write_sharded(ddf, config, VectorColumnInput(...))`
- **[Ray → vector](ray.md)** — `write_sharded(ds, config, VectorColumnInput(...))`

Distributed writers accept `VectorShardedWriteConfig` plus `VectorColumnInput` and shard directly from the dataframe/dataset without collecting everything into the driver first.

## See also

- [Vector Overview](../overview.md) — routing strategies, scatter-gather flow
- [LanceDB](lancedb.md)
- [Read → Sync](../read/sync.md) — `ShardedVectorReader`
- [Read → Async](../read/async.md) — `AsyncShardedVectorReader`
