# Operations

This section covers operating shardyfusion snapshots, validating deployments, and running the project verification matrix.

- **Runtime**
  - [CLI](cli.md) — `shardy` command-line tool for inspection, lookups, rollback, cleanup
  - [History & rollback](history-rollback.md) — manifest history and atomic rollback
- **Metrics**
  - [Prometheus metrics](prometheus-metrics.md) — pull-based metrics
  - [OTel metrics](otel-metrics.md) — push-based OTel metrics
- **Production**
  - [Production guide](production.md) — CI workflows, local checks, health monitoring, DB manifest stores, and troubleshooting
  - [Cloud testing](cloud-testing.md) — manual validation against real cloud infrastructure
  - [Tox & dependency matrix](tox-matrix.md) — supported combinations and verification environments

All operations are read-side or meta-data mutations. Writing data always goes through a [writer](../use-cases/kv-storage/build/index.md).
