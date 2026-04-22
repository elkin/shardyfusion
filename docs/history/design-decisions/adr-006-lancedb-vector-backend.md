# ADR-006: LanceDB as vector backend

**Status:** Accepted (2026-04-19)
**Source:** [`historical-notes/2026-04-19-lancedb-vector-migration.md`](../historical-notes/2026-04-19-lancedb-vector-migration.md), [`historical-notes/2026-04-06-vector-search-review.md`](../historical-notes/2026-04-06-vector-search-review.md)

## Context

The initial vector path (ADR-005) used sqlite-vec exclusively — simple single-file shards, but limited:

- Metric set: `cosine`, `l2` only. No `dot_product`.
- Recall/QPS ceiling at scale (no IVF/HNSW tuning surface).
- Mixed-workload (KV + vector) requires the unified-table model — fine for small/medium scale, not for production-grade ANN.

Production vector workloads require richer indexing primitives.

## Decision

Adopt **LanceDB** as the primary production vector backend, alongside sqlite-vec:

- New `LanceDbFactory` adapter + `vector-lancedb` extra.
- Supports `cosine`, `l2`, `dot_product` (the latter mapped internally to `"dot"` at `vector/adapters/lancedb_adapter.py:142`).
- Composition with KV via `CompositeFactory(kv_factory, vector_factory, vector_spec)` — separate adapters per shard.
- Manifest records `vector.backend ∈ {"lancedb", "sqlite-vec"}`; readers dispatch on this field.
- sqlite-vec retained for the unified single-file flavor and KV+vector use cases that don't need LanceDB's tuning.

## Consequences

- Two backends, two extras (`vector-lancedb`, `vector-sqlite`); two unified extras (`unified-vector`, `unified-vector-sqlite`).
- `unified-vector` extra = `vector-lancedb` + `cel`. `unified-vector-sqlite` = `vector-sqlite` + `cel`.
- LanceDB pulls native deps; sqlite-vec stays pure-Python-friendly.
- Operators choose backend per snapshot; mixing backends across shards in one snapshot is not supported.

## Related

- ADR-005 (sharded vector architecture).
- [`architecture/adapters.md`](../../architecture/adapters.md)
- [`use-cases/build-python-slatedb-lancedb.md`](../../use-cases/build-python-slatedb-lancedb.md)
- [`use-cases/build-python-sqlite-vec.md`](../../use-cases/build-python-sqlite-vec.md)
