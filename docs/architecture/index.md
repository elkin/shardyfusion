# Architecture

Internal design notes. Not user-facing API documentation — for that, see [Reference](../reference/api.md).

## Pages

- [Writer core](writer-core.md) — shared writer pipeline (`_writer_core`).
- [Sharding](sharding.md) — strategies (HASH / CEL / range / vector-CLUSTER / LSH / EXPLICIT).
- [Routing](routing.md) — `SnapshotRouter` and reader-side dispatch.
- [Manifest & `_CURRENT`](manifest-and-current.md) — two-phase publish.
- [Manifest stores](manifest-stores.md) — filesystem/S3 vs Postgres.
- [Run registry](run-registry.md) — run lifecycle, `RunRecord`, deferred cleanup.
- [Retry & cleanup](retry-and-cleanup.md) — `shard_retry`, framework retry stacking.
- [Adapters](adapters.md) — `DbAdapter` protocol, factory contract.
- [Observability](observability.md) — `MetricsCollector`, `MetricEvent`.
- [Error model](error-model.md) — exception hierarchy.
- [Optional imports](optional-imports.md) — extras and lazy import patterns.
- [SQLite B-tree sidecar](sqlite-btree-sidecar.md) — `shard.btreemeta` writer-side artifact.
