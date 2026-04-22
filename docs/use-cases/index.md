# Use Cases

Each use-case page follows the same template: when to use, when not to use, install, minimal example, configuration, functional and non-functional properties, guarantees, weaknesses, failure modes & recovery, and related links.

For a visual map of all use cases, see the [landing page](../index.md).

## Build (writer flavors)

### Spark
- [build-spark-slatedb](build-spark-slatedb.md) — Spark writer, SlateDB backend
- [build-spark-sqlite](build-spark-sqlite.md) — Spark writer, SQLite backend

### Dask
- [build-dask-slatedb](build-dask-slatedb.md) — Dask writer, SlateDB backend
- [build-dask-sqlite](build-dask-sqlite.md) — Dask writer, SQLite backend

### Ray
- [build-ray-slatedb](build-ray-slatedb.md) — Ray writer, SlateDB backend
- [build-ray-sqlite](build-ray-sqlite.md) — Ray writer, SQLite backend

### Python (single-process)
- [build-python-slatedb](build-python-slatedb.md) — Python writer, SlateDB backend
- [build-python-sqlite](build-python-sqlite.md) — Python writer, SQLite backend

### Unified KV + vector
- [build-python-slatedb-lancedb](build-python-slatedb-lancedb.md) — SlateDB KV + LanceDB vectors per shard
- [build-python-sqlite-vec](build-python-sqlite-vec.md) — SQLite + sqlite-vec unified per shard

### Vector standalone
- [build-vector-lancedb-standalone](build-vector-lancedb-standalone.md) — vector-only sharded build with LanceDB
- [build-vector-sqlite-vec-standalone](build-vector-sqlite-vec-standalone.md) — vector-only sharded build with sqlite-vec

## Read

- [read-sync-slatedb](read-sync-slatedb.md) — synchronous reader, SlateDB backend
- [read-sync-sqlite](read-sync-sqlite.md) — synchronous reader, SQLite backend
- [read-async-slatedb](read-async-slatedb.md) — async reader, SlateDB backend
- [read-async-sqlite](read-async-sqlite.md) — async reader, SQLite backend

## Operate

- [operate-cli](operate-cli.md) — `shardy` CLI for inspection, routing, history, rollback, cleanup
- [operate-manifest-history-and-rollback](operate-manifest-history-and-rollback.md) — manifest log and atomic rollback
- [operate-prometheus-metrics](operate-prometheus-metrics.md) — Prometheus metrics collector
- [operate-otel-metrics](operate-otel-metrics.md) — OpenTelemetry metrics collector
