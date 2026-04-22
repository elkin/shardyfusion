# Build a unified KV+vector snapshot (sqlite-vec, single backend)

Use the **sqlite-vec adapter** to write **one SQLite file per shard** that holds both KV rows and vector index — the simplest unified backend.

## When to use

- You want KV + vector under one file per shard.
- You're happy with sqlite-vec's metric set (`cosine`, `l2`).
- You want straightforward downloadable shards or range-read VFS.

## When NOT to use

- You need `dot_product` — sqlite-vec rejects it; use [`build-python-slatedb-lancedb.md`](build-python-slatedb-lancedb.md).
- You only need vector — use [`build-vector-sqlite-vec-standalone.md`](build-vector-sqlite-vec-standalone.md).
- KV-only — use [`build-python-sqlite.md`](build-python-sqlite.md).

## Install

```bash
uv add 'shardyfusion[unified-vector-sqlite,writer-python]'
```

`unified-vector-sqlite` = `vector-sqlite` + `cel`.

## Minimal example

```python
from shardyfusion import WriteConfig, VectorSpec
from shardyfusion.writer.python import write_sharded
from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

vector_spec = VectorSpec(
    dim=384,
    metric="cosine",
)

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/items",
    adapter_factory=SqliteVecFactory(vector_spec=vector_spec),
    vector_spec=vector_spec,
)

result = write_sharded(
    records,
    config,
    key_fn=lambda r: r["id"].encode(),
    value_fn=lambda r: r["payload"],
    vector_fn=lambda r: r["embedding"],
)
```

## Configuration

- `SqliteVecFactory(vector_spec, page_size=4096, cache_size_pages=-2000, ...)` at `sqlite_vec_adapter.py:105`.
- Allowed metrics: `cosine`, `l2`. `dot_product` raises `ConfigValidationError`.
- The manifest records `vector.backend = "sqlite-vec"` automatically; `UnifiedShardedReader` dispatches accordingly.

## Functional / Non-functional properties

- One `.sqlite` file per shard.
- KV rows and vector index in the same file.
- Atomic publish.

## Guarantees

- Successful return ⇒ both KV and vector queryable via `UnifiedShardedReader`.

## Weaknesses

- No `dot_product`.
- Recall/QPS lower than LanceDB at large scale (no IVF/HNSW tuning surface).

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Unsupported metric | `ConfigValidationError` | Use `cosine` or `l2`, or switch to LanceDB. |
| Dim mismatch | `ConfigValidationError` | Fix `VectorSpec.dim`. |
| Shard write failure | `ShardCoverageError` | `shard_retry`; rerun. |

## See also

- [`build-python-slatedb-lancedb.md`](build-python-slatedb-lancedb.md).
- [`build-vector-sqlite-vec-standalone.md`](build-vector-sqlite-vec-standalone.md).
- [`architecture/adapters.md`](../architecture/adapters.md).
