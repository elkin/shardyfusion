# Build a unified KV+vector snapshot (sqlite-vec)

Use the **sqlite-vec adapter** to write **one SQLite file per shard** that holds both KV rows and vector index — the simplest unified backend.

## When to use

- You want KV + vector under one file per shard.
- You're happy with sqlite-vec's metric set (`cosine`, `l2`).
- You want straightforward downloadable shards or range-read VFS.

## When NOT to use

- You need `dot_product` — sqlite-vec rejects it; use [composite LanceDB](composite.md).
- You only need vector — use [vector sqlite-vec](../../vector/build/sqlite-vec.md).
- KV-only — use [KV SQLite](../../kv-storage/build/python.md).

## Install

```bash
uv add 'shardyfusion[unified-vector-sqlite,writer-python]'
```

`unified-vector-sqlite` = `vector-sqlite` + `cel`.

## Minimal example

=== "Python"

    ```python
    from shardyfusion import WriteConfig, VectorSpec
    from shardyfusion.writer.python import write_sharded
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

    vector_spec = VectorSpec(dim=384, metric="cosine")

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

=== "Spark"

    ```python
    from shardyfusion import WriteConfig, VectorSpec
    from shardyfusion.writer.spark import write_sharded
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.serde import ValueSpec

    vector_spec = VectorSpec(dim=384, metric="cosine")

    config = WriteConfig(
        num_dbs=16,
        s3_prefix="s3://my-bucket/snapshots/items",
        adapter_factory=SqliteVecFactory(vector_spec=vector_spec),
        vector_spec=vector_spec,
    )

    result = write_sharded(
        df,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        vector_fn=lambda r: (r["id"], r["embedding"], None),
        vector_columns={"embedding": "embedding"},
    )
    ```

=== "Dask"

    ```python
    from shardyfusion import WriteConfig, VectorSpec
    from shardyfusion.writer.dask import write_sharded
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.serde import ValueSpec

    vector_spec = VectorSpec(dim=384, metric="cosine")

    config = WriteConfig(
        num_dbs=16,
        s3_prefix="s3://my-bucket/snapshots/items",
        adapter_factory=SqliteVecFactory(vector_spec=vector_spec),
        vector_spec=vector_spec,
    )

    result = write_sharded(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        vector_fn=lambda r: (r["id"], r["embedding"], None),
        vector_columns={"embedding": "embedding"},
    )
    ```

=== "Ray"

    ```python
    from shardyfusion import WriteConfig, VectorSpec
    from shardyfusion.writer.ray import write_sharded
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.serde import ValueSpec

    vector_spec = VectorSpec(dim=384, metric="cosine")

    config = WriteConfig(
        num_dbs=16,
        s3_prefix="s3://my-bucket/snapshots/items",
        adapter_factory=SqliteVecFactory(vector_spec=vector_spec),
        vector_spec=vector_spec,
    )

    result = write_sharded(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        vector_fn=lambda r: (r["id"], r["embedding"], None),
        vector_columns={"embedding": "embedding"},
    )
    ```

## Configuration

- `SqliteVecFactory(vector_spec, page_size=4096, cache_size_pages=-2000, ...)` at `sqlite_vec_adapter.py:105`.
- Allowed metrics: `cosine`, `l2`. `dot_product` raises `ConfigValidationError`.
- The manifest records `vector.backend = "sqlite-vec"` automatically; `UnifiedShardedReader` dispatches accordingly.

## Functional properties

- One `.sqlite` file per shard.
- KV rows and vector index in the same file.
- Atomic publish.

## Guarantees

- Successful return => both KV and vector queryable via `UnifiedShardedReader`.

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

- [KV+Vector Overview](../overview.md)
- [Composite LanceDB](composite.md)
- [Read -> Sync](../read/sync.md) — `UnifiedShardedReader`
- [Read -> Async](../read/async.md) — `AsyncUnifiedShardedReader`
