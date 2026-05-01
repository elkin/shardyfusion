# Build a composite KV+vector snapshot (SlateDB + LanceDB)

Use the **composite adapter** to write **two backends per shard** — SlateDB for KV and LanceDB for vector search — published under one manifest.

## When to use

- You need both point-key lookups (KV) and approximate nearest-neighbor search over the same shard layout.
- You want LanceDB's mature vector backend with HNSW/IVF tuning.
- You are happy to pay the cost of two adapters per shard.

## When NOT to use

- You want a single-file unified backend — use [unified sqlite-vec](unified.md).
- KV-only or vector-only — see [KV storage](../../kv-storage/overview.md) or [vector search](../../vector/overview.md).

## Install

```bash
uv add 'shardyfusion[unified-vector,writer-python]'
```

`unified-vector` = `vector-lancedb` + `cel`.

## Minimal example

=== "Python"

    ```python
    from shardyfusion import HashWriteConfig, VectorSpec
    from shardyfusion.writer.python import write_sharded_by_hash
    from shardyfusion.slatedb_adapter import SlateDbFactory
    from shardyfusion.vector.adapters.lancedb_adapter import LanceDbFactory
    from shardyfusion.composite_adapter import CompositeFactory

    vector_spec = VectorSpec(dim=384, metric="cosine")

    config = HashWriteConfig(
        num_dbs=16,
        s3_prefix="s3://my-bucket/snapshots/items",
        adapter_factory=CompositeFactory(
            kv_factory=SlateDbFactory(),
            vector_factory=LanceDbFactory(),
            vector_spec=vector_spec,
        ),
        vector_spec=vector_spec,
    )

    result = write_sharded_by_hash(
        records,
        config,
        key_fn=lambda r: r["id"].encode(),
        value_fn=lambda r: r["payload"],
        vector_fn=lambda r: r["embedding"],
    )
    ```

=== "Spark"

    ```python
    from shardyfusion import HashWriteConfig, VectorSpec
    from shardyfusion.writer.spark import write_sharded_by_hash
    from shardyfusion.slatedb_adapter import SlateDbFactory
    from shardyfusion.vector.adapters.lancedb_adapter import LanceDbFactory
    from shardyfusion.composite_adapter import CompositeFactory
    from shardyfusion.serde import ValueSpec

    vector_spec = VectorSpec(dim=384, metric="cosine")

    config = HashWriteConfig(
        num_dbs=16,
        s3_prefix="s3://my-bucket/snapshots/items",
        adapter_factory=CompositeFactory(
            kv_factory=SlateDbFactory(),
            vector_factory=LanceDbFactory(),
            vector_spec=vector_spec,
        ),
        vector_spec=vector_spec,
    )

    result = write_sharded_by_hash(
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
    from shardyfusion import HashWriteConfig, VectorSpec
    from shardyfusion.writer.dask import write_sharded_by_hash
    from shardyfusion.slatedb_adapter import SlateDbFactory
    from shardyfusion.vector.adapters.lancedb_adapter import LanceDbFactory
    from shardyfusion.composite_adapter import CompositeFactory
    from shardyfusion.serde import ValueSpec

    vector_spec = VectorSpec(dim=384, metric="cosine")

    config = HashWriteConfig(
        num_dbs=16,
        s3_prefix="s3://my-bucket/snapshots/items",
        adapter_factory=CompositeFactory(
            kv_factory=SlateDbFactory(),
            vector_factory=LanceDbFactory(),
            vector_spec=vector_spec,
        ),
        vector_spec=vector_spec,
    )

    result = write_sharded_by_hash(
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
    from shardyfusion import HashWriteConfig, VectorSpec
    from shardyfusion.writer.ray import write_sharded_by_hash
    from shardyfusion.slatedb_adapter import SlateDbFactory
    from shardyfusion.vector.adapters.lancedb_adapter import LanceDbFactory
    from shardyfusion.composite_adapter import CompositeFactory
    from shardyfusion.serde import ValueSpec

    vector_spec = VectorSpec(dim=384, metric="cosine")

    config = HashWriteConfig(
        num_dbs=16,
        s3_prefix="s3://my-bucket/snapshots/items",
        adapter_factory=CompositeFactory(
            kv_factory=SlateDbFactory(),
            vector_factory=LanceDbFactory(),
            vector_spec=vector_spec,
        ),
        vector_spec=vector_spec,
    )

    result = write_sharded_by_hash(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        vector_fn=lambda r: (r["id"], r["embedding"], None),
        vector_columns={"embedding": "embedding"},
    )
    ```

## Configuration

- `VectorSpec(dim, metric, index_type="hnsw", ...)` — set on `HashWriteConfig.vector_spec` or `CelWriteConfig.vector_spec`.
- `metric` for LanceDB: `cosine`, `l2`, `dot_product` (mapped to `"dot"` internally at `vector/adapters/lancedb_adapter.py:142`).
- The backend (`"lancedb"`) is determined by the adapter factory; the manifest's `vector.backend` field is filled from there and used by `UnifiedShardedReader` to dispatch.

## Functional properties

- Each shard contains a SlateDB store **and** a LanceDB table side by side.
- Two sets of files uploaded per shard.
- Atomic publish across both backends (single manifest entry per shard).

## Guarantees

- Successful return => both KV and vector data are addressable via the same `_CURRENT`.
- `UnifiedShardedReader` dispatches to the right backend based on manifest `vector.backend`.

## Weaknesses

- Roughly 2x shard size and upload time vs KV-only.
- LanceDB index build cost included in writer wall time.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Vector dim mismatch | `ConfigValidationError` at write start | Fix `VectorSpec.dim`. |
| LanceDB index build fail | `VectorIndexError` | Check disk; rerun. |
| Either backend fails on a shard | `ShardCoverageError` after retries | `config.shard_retry`; rerun. |

## See also

- [KV+Vector Overview](../overview.md) — composite vs unified concepts
- [Unified sqlite-vec](unified.md) — single-backend alternative
- [`architecture/adapters.md`](../../../architecture/adapters.md)
- [Read -> Sync](../read/sync.md) — `UnifiedShardedReader`
- [Read -> Async](../read/async.md) — `AsyncUnifiedShardedReader`
