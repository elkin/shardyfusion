# Glossary

**Adapter** — Per-shard storage backend (SlateDB, SQLite, sqlite-vec, LanceDB). Implements the `DbAdapter` protocol.

**Adapter factory** — Callable that constructs an adapter for a given shard. Signature: `factory(*, db_url, local_dir)`.

**Composite adapter** — `CompositeFactory(kv_factory, vector_factory, vector_spec)` — pairs a KV adapter with a vector adapter per shard. Used with LanceDB for KV+vector workloads.

**`_CURRENT`** — Mutable pointer object naming the active manifest. Default key. Written atomically; the only mutable object in the snapshot layout.

**Manifest** — Immutable record listing per-shard databases for one published snapshot. Lives at `manifests/<timestamp>_run_id=<run_id>/manifest`.

**Manifest format version** — Integer in `{1, 2, 3}`. v3 required for `routing_values`. Distinct from `CurrentPointer.format_version` (currently `1`).

**Manifest store** — Backend for manifest read/write. Default is filesystem/S3. Postgres-backed store uses three tables (builds, shards, pointer) in a single transaction.

**Run record** — YAML record at `runs/<timestamp>_run_id=<run_id>_<uuidhex>/run.yaml` capturing one writer run's lifecycle. `RunStatus ∈ {RUNNING, SUCCEEDED, FAILED}`.

**Run registry** — The collection of run records. Consulted by `cleanup` to determine which S3 objects are safe to reap.

**Sharding** — Mapping from records to shard IDs. Strategies: hash, range, CEL (categorical), and (for vectors) CLUSTER, LSH, EXPLICIT, CEL.

**Routing** — Reverse mapping from query (key or vector) to shard. Done locally from the manifest by `SnapshotRouter`.

**`UnknownRoutingTokenError`** — Raised by `SnapshotRouter` when a CEL routing token is not in the manifest's `routing_values`. Inherits from `ValueError`, **not** `ShardyfusionError`.

**Two-phase publish** — (1) write manifest, (2) swap `_CURRENT`. Atomic visibility flips on the second step.

**Winner sort key** — Tuple used to break ties between concurrent runs writing the same shard: `(attempt, task_attempt_id_or_2**63-1, db_url_or_empty)`. Sentinel is signed int64 max.

**`KeyEncoding`** — How routing keys are encoded into bytes: `U64BE`, `U32BE`, `UTF8`, `RAW`.

**`VectorSpec`** — `(dim, metric, index_type, index_params, quantization, ...)`. Set on `WriteConfig.vector_spec` for unified KV+vector flows. Backend (`"lancedb"` / `"sqlite-vec"`) is determined by the adapter factory, not by `VectorSpec`. The chosen backend is recorded in the manifest's `vector.backend` field; readers dispatch on it.

**Standalone vector** — Vector path with no KV side. Writer: `write_vector_sharded` from `shardyfusion.vector`. Reader: `ShardedVectorReader`.

**Composite (KV+vector)** — Two adapters per shard. `CompositeFactory` glues them. Used with LanceDB.

**Unified (KV+vector)** — One adapter per shard handling both. `SqliteVecFactory` is the only such adapter.

**`UnifiedShardedReader`** — Reader that auto-dispatches between composite and unified backends based on manifest's `vector.backend`.

**Snapshot** — A consistent published state: one manifest plus the shards it references plus the run records that produced them.

**Snapshot pinning** — Once a reader loads a manifest, all reads use that manifest until `refresh()` is called.

**`shard_retry`** — Per-shard retry budget for transient adapter failures. Lives on `WriteConfig`. Independent of framework-level task retry.

**`ShardCoverageError`** — Raised when one or more shards failed to produce a database after `shard_retry` is exhausted.

**`ShardAssignmentError`** — Raised by Spark/Dask/Ray writers when post-shuffle records don't match expected shard IDs (with `verify_routing=True`).

**Two-backend extras** — `vector-lancedb` (LanceDB) and `vector-sqlite` (sqlite-vec). `unified-vector` = `vector-lancedb` + `cel`. `unified-vector-sqlite` = `vector-sqlite` + `cel`.

**Reader factory** — Construct a per-shard reader. Sync: `SlateDbReaderFactory`, `SqliteReaderFactory` (download-and-cache), `SqliteRangeReaderFactory` (range-read VFS). Async: `AsyncSlateDbReaderFactory`, `AsyncSqliteReaderFactory`.

**ConcurrentShardedReader** — Sync reader with thread-pool fan-out for `multi_get`. Same constructor as `ShardedReader` plus `max_workers`.

**MetricsCollector** — Optional protocol implemented by `PrometheusCollector` and `OtelCollector`. `metrics=None` disables emission entirely.

**MetricEvent** — `str, Enum` of named events: `write.started`, `write.shard.completed`, `s3.retry`, `vector.reader.search`, etc.
