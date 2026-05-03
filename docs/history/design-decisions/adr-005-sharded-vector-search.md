# ADR-005: Sharded vector search architecture

**Status:** Accepted (2026-04-04)
**Source:** [`historical-notes/2026-04-04-sharded-vector-search.md`](../historical-notes/2026-04-04-sharded-vector-search.md)

## Context

Vector search workloads need:

- Horizontal scale across shards (a single ANN index doesn't fit).
- Sharding strategies tuned to vector data (k-means clusters, LSH, explicit, CEL).
- Coexistence with KV data when applications need both.

A naive "one big index" approach doesn't scale and doesn't compose with the existing KV sharding model.

## Decision

Build a **parallel sharded vector path** that mirrors the KV path:

- `VectorShardedWriteConfig` mirrors `WriteConfig` (num_dbs, s3_prefix, manifest, two-phase publish).
- `VectorShardingSpec` strategies: `CLUSTER` (k-means, default), `LSH`, `EXPLICIT` (`VectorRecord.shard_id`), `CEL`.
- Standalone reader: `ShardedVectorReader.search(query, top_k, ...)` scatter-gathers across shards.
- Composition with KV: two paths
  - **Composite**: separate KV adapter + vector adapter per shard, glued by `CompositeFactory`. Used with LanceDB.
  - **Unified**: single backend per shard that handles both (sqlite-vec).
- `UnifiedShardedReader` dispatches based on manifest's `vector.backend` field.

## Consequences

- Vector path reuses manifest, two-phase publish, run registry.
- Two coexistence patterns trade simplicity vs flexibility.
- Vector exports kept under `shardyfusion.vector` (only `VectorSpec` re-exported at top level).

## Related

- [`architecture/sharding.md`](../../architecture/sharding.md)
- [`architecture/adapters.md`](../../architecture/adapters.md)
- ADR-006 (LanceDB backend choice).
