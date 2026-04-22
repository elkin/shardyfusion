# Use Cases

The use-case docs are organized as a **decision tree**. Start with the conceptual overview for your use-case type, then drill down into writer or reader specifics.

## Sharded KV Storage

The foundational use case: write key-value pairs into sharded immutable snapshots, read them back with routed lookups.

- **[Overview](kv-storage/overview.md)** — sharding, manifests, two-phase publish, safety properties (start here)
- **Build**
  - [Choosing a writer](kv-storage/build/index.md)
  - [Python](kv-storage/build/python.md)
  - [Spark](kv-storage/build/spark.md)
  - [Dask](kv-storage/build/dask.md)
  - [Ray](kv-storage/build/ray.md)
- **Read**
  - [Choosing a reader](kv-storage/read/index.md)
  - [Sync SlateDB](kv-storage/read/sync/slatedb.md)
  - [Sync SQLite](kv-storage/read/sync/sqlite.md)
  - [Async SlateDB](kv-storage/read/async/slatedb.md)
  - [Async SQLite](kv-storage/read/async/sqlite.md)

## Sharded KV Storage with Vector Search

Write both KV pairs and vector embeddings in the same snapshot. Query by key or by nearest-neighbor search.

- **[Overview](kv-vector/overview.md)** — composite vs unified backends
- **Build**
  - [Composite (SlateDB + LanceDB)](kv-vector/build/composite.md)
  - [Unified (sqlite-vec)](kv-vector/build/unified.md)
- **Read**
  - [Sync](kv-vector/read/sync.md) — `UnifiedShardedReader`
  - [Async](kv-vector/read/async.md) — `AsyncUnifiedShardedReader`

## Sharded Vector Search

Vector-only: write embeddings into sharded indices, query by approximate nearest-neighbor search.

- **[Overview](vector/overview.md)** — routing strategies, scatter-gather flow
- **Build**
  - [LanceDB](vector/build/lancedb.md)
  - [sqlite-vec](vector/build/sqlite-vec.md)
- **Read**
  - [Sync](vector/read/sync.md) — `ShardedVectorReader`
  - [Async](vector/read/async.md) — `AsyncShardedVectorReader`

## Operate

Operational tasks for all use-case types:

- [CLI](../operate/cli.md)
- [History & rollback](../operate/history-rollback.md)
- [Prometheus metrics](../operate/prometheus-metrics.md)
- [OTel metrics](../operate/otel-metrics.md)
