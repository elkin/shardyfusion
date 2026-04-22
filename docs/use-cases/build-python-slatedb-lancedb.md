# Build a unified KV+vector snapshot (SlateDB + LanceDB, composite)

Use the **composite adapter** to write **two backends per shard** — SlateDB for KV and LanceDB for vector search — published under one manifest.

## When to use

- You need both point-key lookups (KV) and approximate nearest-neighbor search over the same shard layout.
- You want the mature LanceDB vector backend.
- You are happy to pay the cost of two adapters per shard (more disk, more upload time).

## When NOT to use

- KV-only — use [`build-python-slatedb.md`](build-python-slatedb.md).
- Vector-only, no KV — use [`build-vector-lancedb-standalone.md`](build-vector-lancedb-standalone.md).
- You want a single-file unified backend — use [`build-python-sqlite-vec.md`](build-python-sqlite-vec.md).

## Install

```bash
uv add 'shardyfusion[unified-vector,writer-python]'
```

`unified-vector` = `vector-lancedb` + `cel`.

## Minimal example

```python
from shardyfusion import WriteConfig, VectorSpec
from shardyfusion.writer.python import write_sharded
from shardyfusion.slatedb_adapter import SlateDbFactory
from shardyfusion.vector.adapters.lancedb_adapter import LanceDbFactory
from shardyfusion.composite_adapter import CompositeFactory

vector_spec = VectorSpec(
    dim=384,
    metric="cosine",
)

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/items",
    adapter_factory=CompositeFactory(
        kv_factory=SlateDbFactory(),
        vector_factory=LanceDbFactory(),
        vector_spec=vector_spec,
    ),
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

- `VectorSpec(dim, metric, index_type="hnsw", ...)` — set on `WriteConfig.vector_spec`. The backend (`"lancedb"`) is determined by the adapter factory you pass to `CompositeFactory`; the manifest's `vector.backend` field is filled in from there and used by `UnifiedShardedReader` to dispatch.
- `metric` for LanceDB: `cosine`, `l2`, `dot_product` (mapped to `"dot"` internally at `vector/adapters/lancedb_adapter.py:142`).

## Functional / Non-functional properties

- Each shard contains a SlateDB store **and** a LanceDB table side by side.
- Two sets of files uploaded per shard.
- Atomic publish across both backends (single manifest entry per shard).

## Guarantees

- Successful return ⇒ both KV and vector data are addressable via the same `_CURRENT`.
- `UnifiedShardedReader` dispatches to the right backend based on manifest `vector.backend`.

## Weaknesses

- Roughly 2× shard size and upload time vs KV-only.
- LanceDB index build cost included in writer wall time.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Vector dim mismatch | `ConfigValidationError` at write start | Fix `VectorSpec.dim`. |
| LanceDB index build fail | `VectorIndexError` | Check disk; rerun. |
| Either backend fails on a shard | `ShardCoverageError` after retries | `config.shard_retry`; rerun. |

## See also

- [`build-python-sqlite-vec.md`](build-python-sqlite-vec.md) — single-backend unified alternative.
- [`build-vector-lancedb-standalone.md`](build-vector-lancedb-standalone.md) — vector-only.
- [`architecture/adapters.md`](../architecture/adapters.md).
