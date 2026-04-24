# Operate

This section covers operating shardyfusion snapshots in production.

- **[CLI](cli.md)** — `shardy` command-line tool for inspection, lookups, rollback, cleanup
- **[History & rollback](history-rollback.md)** — manifest history and atomic rollback
- **[Prometheus metrics](prometheus-metrics.md)** — pull-based metrics
- **[OTel metrics](otel-metrics.md)** — push-based OTel metrics

All operations are read-side or meta-data mutations. Writing data always goes through a [writer](../use-cases/kv-storage/build/index.md).
