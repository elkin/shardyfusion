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
    VectorRecord, VectorWriteConfig, write_vector_sharded,
)
from shardyfusion.vector.config import VectorIndexConfig
from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
from shardyfusion import VectorSpec

vector_spec = VectorSpec(dim=384, metric="cosine")

config = VectorWriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/embeddings",
    index_config=VectorIndexConfig(dim=384, metric="cosine"),
    adapter_factory=SqliteVecFactory(vector_spec=vector_spec),
)

result = write_vector_sharded(records, config)
```

## Configuration

- Allowed metrics: `cosine`, `l2`. `dot_product` rejected.
- Same `VectorShardingSpec` strategies as the LanceDB variant.

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

- **[Spark → vector](spark.md)** — `write_vector_sharded(df, config, vector_col=..., id_col=...)`
- **[Dask → vector](dask.md)** — `write_vector_sharded(ddf, config, vector_col=..., id_col=...)`
- **[Ray → vector](ray.md)** — `write_vector_sharded(ds, config, vector_col=..., id_col=...)`

Distributed writers accept `VectorWriteConfig` (or build one via `VectorWriteConfig.from_vector_spec()`) and shard directly from the dataframe/dataset without collecting everything into the driver first.

## See also

- [Vector Overview](../overview.md) — routing strategies, scatter-gather flow
- [LanceDB](lancedb.md)
- [Read → Sync](../read/sync.md) — `ShardedVectorReader`
- [Read → Async](../read/async.md) — `AsyncShardedVectorReader`
