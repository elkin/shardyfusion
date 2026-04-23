# Choosing a writer and backend

This section covers building sharded KV snapshots. Pick your writer based on your data source and infrastructure:

| Writer | Input type | Java required | Cluster required | Best for |
|---|---|---|---|---|
| **[Python](python.md)** | `Iterable[T]` | No | No | Single-host, streaming, simplicity |
| **[Spark](spark.md)** | PySpark `DataFrame` | Yes | Optional (local mode works) | Large-scale ETL, existing Spark pipeline |
| **[Dask](dask.md)** | Dask `DataFrame` | No | Optional | Distributed scale-out without JVM |
| **[Ray](ray.md)** | Ray `Dataset` | No | Optional | ML preprocessing pipelines, actor scheduling |

All writers share the same core behavior: deterministic routing, attempt-isolated paths, deterministic winner selection, two-phase publish, and run records. See [KV Storage Overview](../overview.md) for the conceptual model.

## Choosing a backend

| Backend | Read-side access | When to use |
|---|---|---|
| **SlateDB** (default) | Point-key `get` / `multi_get` | Lowest friction, LSM characteristics, default for most users |
| **SQLite** | Point-key + SQL queries + range-read VFS | Need SQL, single-file shards, or remote page-level access |

Backend selection is a single config swap (`adapter_factory=SqliteFactory()` instead of default). Everything else — routing, publishing, reading — works identically.
