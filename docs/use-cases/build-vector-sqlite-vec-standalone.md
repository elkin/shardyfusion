# Build a vector-only snapshot (sqlite-vec, standalone)

Use the **standalone vector writer** with the sqlite-vec backend to build a sharded vector index where each shard is one SQLite file.

## When to use

- Pure ANN workload, single-file shards preferred.
- `cosine` or `l2` are sufficient.
- You want minimal operational surface (no LanceDB native deps).

## When NOT to use

- You need `dot_product` — use [`build-vector-lancedb-standalone.md`](build-vector-lancedb-standalone.md).
- You also need KV — use [`build-python-sqlite-vec.md`](build-python-sqlite-vec.md).

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

## Functional / Non-functional properties

- One `.sqlite` file per shard (vector data only — no KV table).
- Atomic two-phase publish.

## Guarantees

- Successful return ⇒ all shards queryable via `ShardedVectorReader`.

## Weaknesses

- No `dot_product`.
- No IVF/HNSW tuning surface.

## Failure modes & recovery

Same as [`build-vector-lancedb-standalone.md`](build-vector-lancedb-standalone.md), minus IVF cluster-size errors.

## See also

- [`build-vector-lancedb-standalone.md`](build-vector-lancedb-standalone.md).
- [`build-python-sqlite-vec.md`](build-python-sqlite-vec.md).
