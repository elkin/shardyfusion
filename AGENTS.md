# Repository Guidelines

`shardyfusion` — sharded snapshot writer/reader for S3-compatible object storage.

> **Keep this file accurate.** After any code change that affects architecture, module organization, CLI commands, or public API, verify that `AGENTS.md` remains accurate and update it if needed. Do not let it go stale. Breaking or otherwise important changes should also be recorded in `CHANGELOG.md`.

## Project Structure & Module Organization

- Source package: `shardyfusion/`
  - Writer path (Spark): `writer/spark/writer.py`, `writer/spark/sharding.py`, `writer/spark/single_db_writer.py` → `_shard_writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Writer path (Dask): `writer/dask/writer.py`, `writer/dask/sharding.py`, `writer/dask/single_db_writer.py` → `_shard_writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Writer path (Ray): `writer/ray/writer.py`, `writer/ray/sharding.py`, `writer/ray/single_db_writer.py` → `_shard_writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Writer path (Python): `writer/python/writer.py`, `writer/python/_parallel_writer.py` (multiprocessing) → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Reader path (sync): `reader/reader.py`, `reader/concurrent_reader.py`, `reader/_base.py`, `reader/_types.py`, `reader/_state.py` → `routing.py` → `manifest_store.py`, `db_manifest_store.py`
  - Reader path (async): `reader/async_reader.py` → `routing.py` → `async_manifest_store.py` (`AsyncS3ManifestStore`)
  - CLI path: `cli/app.py`, `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`
  - SQLite backend: `sqlite_adapter.py` (writer + download-and-cache reader + range-read VFS reader + async wrappers), `_sqlite_vfs.py` (S3 range-read VFS via `obstore`)
  - Vector search: `vector/types.py`, `vector/config.py`, `vector/sharding.py`, `vector/_merge.py`, `vector/_distributed.py`, `vector/writer.py`, `vector/reader.py`, `vector/async_reader.py`, `vector/adapters/lancedb_adapter.py`
  - Unified KV+vector: `sqlite_vec_adapter.py` (sqlite-vec unified adapter), `composite_adapter.py` (KV+vector composite), `reader/unified_reader.py` (`UnifiedShardedReader`), `reader/async_unified_reader.py` (`AsyncUnifiedShardedReader`)
  - Shared models/protocols: `config.py`, `credentials.py`, `manifest.py`, `manifest_store.py`, `async_manifest_store.py`, `run_registry.py`, `storage.py`, `testing.py`
  - Core utilities: `errors.py`, `type_defs.py`, `sharding_types.py`, `ordering.py`, `logging.py`, `serde.py`, `cel.py`, `_rate_limiter.py`, `_slatedb_symbols.py`, `metrics/`
- Tests: `tests/`
  - Unit: `tests/unit/shared`, `tests/unit/read`, `tests/unit/read_async`, `tests/unit/writer` (incl. `spark/`, `dask/`, `ray/`, `python/`, `core/`), `tests/unit/backend/{slatedb,sqlite}`, `tests/unit/cel`, `tests/unit/metrics`, `tests/unit/cli`, `tests/unit/vector`
  - Integration: `tests/integration/read`, `tests/integration/writer` (incl. `spark/`, `dask/`, `ray/`, `python/`), `tests/integration/backend/sqlite`, `tests/integration/vector`
  - E2E: `tests/e2e/read`, `tests/e2e/writer` (incl. `spark/`, `dask/`, `ray/`, `python/`), `tests/e2e/cli`, `tests/e2e/vector` (Garage S3 via compose)
  - Shared helpers: `tests/helpers/s3_test_scenarios.py`, `tests/helpers/smoke_scenarios.py`, `tests/helpers/run_record_assertions.py`
- Docs: `docs/` with MkDocs config in `mkdocs.yml`
- Local container/dev setup: `docker/ci.Dockerfile`, `.devcontainer/devcontainer.json`

## Development Workflow

Use `just` for all development tasks. Run `just --list` for available recipes. Avoid calling `uv` directly unless absolutely necessary — `just` recipes handle the correct environment setup, flags, and tox orchestration for you.

Key workflows:
- **Bootstrap**: `just setup` (fresh clone), `just doctor` (health check)
- **Dependencies**: `just sync` (full dev), or `just setup` (idempotent bootstrap)
- **Quality**: `just quality` (lint, format, type check, package, docs-check)
- **Tests**: `just unit`, `just integration`, `just d-e2e` (E2E in container). Use `-p=N` for tox parallel envs and `-n=M` for pytest-xdist workers (e.g. `just unit -p=4 -n=6`). **Always prefer `just` over direct `uv`/`pytest` invocations** — `just` runs tests across multiple tox envs with correct extras and environment setup (e.g. `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` for Ray tests).
- **Fix**: `just fix` (auto-format and lint-fix)
- **Docs**: `just docs`, `just docs-serve`
- **Coverage**: `just coverage` (unit tests with HTML report)

**Package manager**: uv (invoked internally by `just` recipes). Core dependency: `pyyaml>=6.0`, `obstore>=0.4`.

Many optional extras are defined in `pyproject.toml` (e.g. `read-slatedb`, `read-slatedb-async`, `read-sqlite`, `read-sqlite-range`, `read-sqlite-adaptive` / `sqlite-adaptive`, `sqlite-async`, `sqlite-adaptive-async`, `writer-spark-slatedb`, `writer-spark-sqlite`, `writer-python-slatedb`, `writer-python-sqlite`, `writer-dask-slatedb`, `writer-dask-sqlite`, `writer-ray-slatedb`, `writer-ray-sqlite`, `cli-minimal`, `cli` (kitchen-sink), `cel`, `metrics-prometheus`, `metrics-otel`, `vector-lancedb`, `vector-sqlite`, `unified-slatedb-lancedb`, `unified-sqlite-vec`, `all`). Every backend is named explicitly in the extra (no implicit default backend). Container commands mirror local ones with a `d-` prefix (e.g. `just d-e2e`). Container runs use an isolated uv project env at `/opt/shardyfusion-venv`.

## CI Pipeline (GitHub Actions)

On every push/PR: **quality → package → unit → integration** (parallel within each stages). E2E is local/container only (`just d-e2e`). Java 17 (temurin) for Spark jobs. **Dask, Ray, and CEL do not run on py3.14.** Weekly scheduled build on Mondays at 06:00 UTC. CI matrix is auto-generated via `just ci-matrix` (runs `scripts/generate_ci_matrix.py`).

## Architecture

The library is split into six independent paths that share config, manifest models, and core logic:

**Writer path (Spark)** (requires PySpark + Java):
`writer/spark/writer.py` → `_shard_writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)
`writer/spark/sharding.py` (shard assignment), `writer/spark/single_db_writer.py` (single-shard variant), `writer/spark/util.py` (`DataFrameCacheContext`)

**Writer path (Dask)** (no Spark/Java needed):
`writer/dask/writer.py` → `_shard_writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)
`writer/dask/sharding.py` (shard assignment), `writer/dask/single_db_writer.py` (single-shard variant)

**Writer path (Ray)** (no Spark/Java needed):
`writer/ray/writer.py` → `_shard_writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)
`writer/ray/sharding.py` (shard assignment), `writer/ray/single_db_writer.py` (single-shard variant), `writer/ray/_compat.py` (pandas 3.0+ compatibility patches)

**Writer path (Python)** (no Spark/Java needed):
`writer/python/writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)
`writer/python/_parallel_writer.py` (multiprocessing with shared memory — used when `parallel=True`)

**Reader path (sync)** (no Spark/Java needed):
`reader/reader.py` (`ShardedReader`) / `reader/concurrent_reader.py` (`ConcurrentShardedReader`) → `reader/_base.py` → `reader/_types.py`, `reader/_state.py` → `routing.py` → `manifest_store.py`, `db_manifest_store.py`

**Reader path (async)** (no Spark/Java needed):
`reader/async_reader.py` → `routing.py` → `async_manifest_store.py` (`AsyncS3ManifestStore` via `AsyncObstoreBackend`)

**CLI path** (requires click + pyyaml):
`cli/app.py` → `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`

**Vector writer path** (no Spark/Java needed; requires `vector-lancedb` or `vector-sqlite` extra):
`vector/writer.py` (`write_sharded`) → `vector/sharding.py` (CLUSTER/LSH/EXPLICIT/CEL assignment) → `vector/_distributed.py` (shared vector write core) → `vector/adapters/lancedb_adapter.py` (LanceDB HNSW)

**Vector reader path** (no Spark/Java needed):
`vector/reader.py` (`ShardedVectorReader`) → `vector/sharding.py` (query routing) → `vector/adapters/lancedb_adapter.py` or `sqlite_vec_adapter.py` → `vector/_merge.py` (top-k heap merge)
`vector/async_reader.py` (`AsyncShardedVectorReader`) → same routing + merge path

**Unified KV + vector path** (no Spark/Java needed):
`reader/unified_reader.py` (`UnifiedShardedReader`, extends `ShardedReader`) → `vector/_merge.py` for search merge
`reader/async_unified_reader.py` (`AsyncUnifiedShardedReader`, extends `AsyncShardedReader`) → `vector/_merge.py` for async search merge
`composite_adapter.py` (`CompositeAdapter`) → wraps KV adapter + vector adapter per shard (LanceDB sidecar)
`sqlite_vec_adapter.py` (`SqliteVecAdapter`) → unified KV+vector in a single SQLite DB per shard (sqlite-vec)

### Module Dependency Graph

```
Layer 0 — Core types & errors: errors.py (incl. VectorIndexError, VectorSearchError), type_defs.py (incl. RetryConfig), sharding_types.py, ordering.py, logging.py (incl. LogContext/JsonFormatter), metrics/ (package: _events.py, _protocol.py, prometheus.py, otel.py)
Layer 1 — Config & serialization: config.py, serde.py, _rate_limiter.py, cel.py (CEL compile/evaluate/boundaries, optional cel extra), run_registry.py (RunRecord, S3RunRegistry, RunRecordLifecycle), credentials.py
Layer 2 — Storage, routing, manifest, run registry: storage.py (`StorageBackend` + `AsyncStorageBackend` Protocols, `ObstoreBackend`/`AsyncObstoreBackend`, `MemoryBackend`/`AsyncMemoryBackend`, `create_s3_store()`), manifest.py, routing.py, manifest_store.py, async_manifest_store.py, db_manifest_store.py (PostgresManifestStore)
Layer 3 — Writer core: _writer_core.py (shared by all writers; assemble_build_result, cleanup_losers, CleanupAction, cleanup_stale_attempts, cleanup_old_runs), _shard_writer.py (shared shard-write loop + retry: write_shard_core, write_shard_with_retry, ShardWriteParams)
Layer 4 — Entry points: writer/{spark,dask,ray,python}/*.py, reader/ (reader.py, concurrent_reader.py, _base.py, _types.py, _state.py), reader/async_reader.py, reader/unified_reader.py, reader/async_unified_reader.py, cli/app.py
Layer vec — Vector search: vector/types.py, vector/config.py (vector types/protocols/enums), vector/sharding.py (CLUSTER/LSH/EXPLICIT/CEL routing), vector/_merge.py (top-k heap merge), vector/_distributed.py (shared write core), vector/writer.py (write_sharded), vector/reader.py (ShardedVectorReader), vector/async_reader.py (AsyncShardedVectorReader)
Layer 5 — Adapters & testing: slatedb_adapter.py, sqlite_adapter.py (optional apsw for range-read VFS), _sqlite_vfs.py (S3 range-read VFS with page-aligned LRU cache via obstore), sqlite_vec_adapter.py (unified KV+vector SQLite), vector/adapters/lancedb_adapter.py (LanceDB HNSW sidecar), composite_adapter.py (KV+vector composite), testing.py
Layer opt — Optional publishers: metrics/prometheus.py (metrics-prometheus extra), metrics/otel.py (metrics-otel extra)
Layer internal — Symbols & compatibility: _slatedb_symbols.py (lazy SlateDB symbol checks), writer/ray/_compat.py (pandas 3.0+ patches)
```

### Write Pipeline

**Spark writer** (`write_hash_sharded` / `write_cel_sharded`):
1. Entry point in `writer/spark/writer.py`. Optionally applies Spark conf overrides via `SparkConfOverrideContext`.
2. `writer/spark/sharding.py` adds `_shard_id` column via `mapInArrow` (hash or CEL, all Python-based), then converts the DataFrame to a pair RDD partitioned so partition index = db_id.
3. Each partition writes one shard to S3 at a shard path (`shards/run_id=.../db=XXXXX/attempt=YY/`).
4. The driver streams results via `toLocalIterator()` and selects deterministic winners (lowest attempt → task_attempt_id → URL).
5. A manifest artifact is built and published, the `_CURRENT` pointer is updated, and the run record is marked terminal (`succeeded` or `failed`).

**Dask writer** (`write_hash_sharded` / `write_cel_sharded`):
1. Entry point in `writer/dask/writer.py`. Accepts `dd.DataFrame` with `key_col`/`value_spec`.
2. `writer/dask/sharding.py` adds `_shard_id` column via Python routing function applied per partition. For CEL sharding, uses `pandas_rows_to_contexts()` + `route_cel()` with full row context (supporting non-key columns).
3. Shuffles by `_shard_id`, then `map_partitions` writes each shard. Empty shards (no rows in partition) are omitted from the manifest; `select_winners()` filters them out.
4. Optional rate limiting via `config.rate_limits.max_writes_per_second` (token-bucket). Routing verification via `verify_routing_agreement()`.
5. Uses the same `_writer_core.py` functions for winner selection, manifest building, publishing, and the same run-record lifecycle helper.

**Ray writer** (`write_hash_sharded` / `write_cel_sharded`):
1. Entry point in `writer/ray/writer.py`. Accepts `ray.data.Dataset` with `key_col`/`value_spec`.
2. `writer/ray/sharding.py` adds `_shard_id` column via Arrow batch format (`map_batches` with `batch_format="pyarrow"`, `zero_copy_batch=True`). For CEL sharding, uses `route_cel_batch()` directly on Arrow batches (same approach as Spark).
3. Repartitions by `_shard_id` using hash shuffle (`DataContext.shuffle_strategy = "HASH_SHUFFLE"`; saved/restored). Then `map_batches` writes each shard with `batch_format="pandas"`. Empty shards (no rows in partition) are omitted from the manifest; `select_winners()` filters them out.
4. Optional rate limiting via `config.rate_limits.max_writes_per_second` (token-bucket). Routing verification via `_verify_routing_agreement()`.
5. Uses the same `_writer_core.py` functions for winner selection, manifest building, publishing, and the same run-record lifecycle helper.

**Python writer** (`write_hash_sharded` / `write_cel_sharded`):
1. Entry point in `writer/python/writer.py`. Accepts `Iterable[T]` with `key_fn`/`value_fn` callables.
2. Supports single-process (all adapters open simultaneously via `contextlib.ExitStack`) or multi-process (`parallel=True`, one worker per shard via `multiprocessing.spawn`).
3. Multi-process mode uses `writer/python/_parallel_writer.py` with shared memory for passing rows to worker processes. Configurable memory budgets via `max_parallel_shared_memory_bytes` and `max_parallel_shared_memory_bytes_per_worker`.
4. Optional rate limiting via `config.rate_limits.max_writes_per_second` and `config.rate_limits.max_write_bytes_per_second` (token-bucket in `_rate_limiter.py`). Single-process mode supports global memory ceilings via `options.buffering.max_total_batched_items` and `options.buffering.max_total_batched_bytes` to prevent OOM when buffering across many shards.
5. Uses the same `_writer_core.py` functions for routing, winner selection, manifest building, publishing, and the same run-record lifecycle helper.

### Read Pipeline

1. `ShardedReader` / `ConcurrentShardedReader` in `reader/reader.py` loads the `_CURRENT` pointer from S3 and dereferences the manifest. On cold start, if `load_manifest()` raises `ManifestParseError`, the reader falls back to previous manifests via `list_manifests()` (up to `max_fallback_attempts`, default 3).
2. Builds a `SnapshotRouter` from the manifest sharding metadata (mirrors write-time sharding logic). For empty shards (absent from manifest, padded by router with `db_url=None`), a `_NullShardReader` is used instead of opening a real DB.
3. `get(key, *, routing_context=None)` / `multi_get(keys, *, routing_context=None)` routes keys to shard IDs, then reads from the appropriate shard. Keys routed to empty shards return `None` immediately via the null reader. The optional `routing_context: dict[str, object] | None` parameter is forwarded to the router for CEL-based sharding (provides column values used in the CEL expression).
4. `shard_for_key(key, *, routing_context=None)` / `shards_for_keys(keys)` return `RequiredShardMeta` for routing inspection without DB access. `reader_for_key(key)` / `readers_for_keys(keys)` return borrowed `ShardReaderHandle` handles for direct shard access. `route_key(key, *, routing_context=None)` also accepts `routing_context` for CEL routing.
5. `ShardedReader` swaps state directly on refresh. `ConcurrentShardedReader` serialises refresh I/O via `_refresh_lock` and atomically swaps readers using reference counting with a compare-and-swap guard for safe cleanup. Borrowed `ShardReaderHandle` handles hold refcount increments, preventing cleanup of old state while borrows are outstanding. On `refresh()`, a malformed manifest is silently skipped (returns `False`), keeping the previous good state.
6. `ConcurrentShardedReader` provides thread safety via `threading.Lock` (default) or `ThreadPoolExecutor` pool mode (`thread_safety` config). Pool mode supports configurable checkout timeout (`pool_checkout_timeout`, default 30s).

### Async Read Pipeline

`AsyncShardedReader` in `reader/async_reader.py` mirrors the sync reader API using asyncio:

1. Constructed via `@classmethod async def open(...)` (since `__init__` can't do async I/O). On cold start, if `load_manifest()` raises `ManifestParseError`, falls back to previous manifests via `list_manifests()` (up to `max_fallback_attempts`, default 3).
2. S3 manifest loading uses `AsyncS3ManifestStore` (native `AsyncObstoreBackend`).
3. `async get(key)` / `async multi_get(keys)` route keys and read from async shard readers. Both hold borrow count increments during reads, preventing `refresh()`/`close()` from closing shard readers while reads are in-flight. The borrow is acquired after the rate limiter to avoid blocking state cleanup during throttling. `multi_get` fans out via `asyncio.TaskGroup` with optional `asyncio.Semaphore` for concurrency limiting (`max_concurrency`).
4. `async refresh()` serialises via `asyncio.Lock`. Swaps state and defers cleanup of retired readers via `state.aclose()` (scheduled as a task via `loop.create_task()`) when their borrow count hits 0. `_AsyncReaderState` supports the async context manager protocol. A malformed manifest is silently skipped (returns `False`), keeping the previous good state.
5. Sync metadata methods (`key_encoding`, `snapshot_info()`, `shard_details()`, `route_key()`) and borrow methods (`reader_for_key()`, `readers_for_keys()`) work identically to the sync reader.
6. Supports `async with` context manager via `__aenter__`/`__aexit__`.

### Unified & Vector Readers

**Unified KV+vector readers** combine point lookups and ANN search in a single reader instance:

- `UnifiedShardedReader` (sync, `reader/unified_reader.py`): extends `ShardedReader` with `search(query, top_k)` and `batch_search(queries, top_k)` methods. Auto-detects backend from manifest metadata (`sqlite-vec` vs composite LanceDB). Lazy-imported via `__getattr__` in the top-level package to avoid pulling in numpy for non-vector users.
- `AsyncUnifiedShardedReader` (async, `reader/async_unified_reader.py`): extends `AsyncShardedReader` with async `search()` and `batch_search()`. Same auto-detection logic. Also lazy-imported.

**Vector-only readers** for ANN search without KV lookup:

- `ShardedVectorReader` (sync, `vector/reader.py`): loads vector manifest, fans out searches across shards using a thread-pool. Supports `search()`, `batch_search()`, `route_vector()`, `shard_for_id()`, `refresh()`, `close()`.
- `AsyncShardedVectorReader` (async, `vector/async_reader.py`): async counterpart. Constructed via `open()` classmethod. Same API with async methods.

### CLI (`shardy`)

Entry point: `cli/app.py:main`. Subcommands: `get KEY`, `multiget KEY [KEY ...]`, `info`, `shards`, `health [--staleness-threshold SECONDS]`, `route KEY`, `search VECTOR [--vector-file PATH] [--top-k N] [--shard-ids IDS]`, `exec --script FILE`, `history [--limit N]`, `rollback (--ref REF | --run-id RUN_ID | --offset N)`, `cleanup [--dry-run] [--include-old-runs] [--older-than DURATION] [--keep-last N] [--max-retries N]`. No subcommand → interactive REPL (`cmd.Cmd` with `shardy> ` prompt; REPL-only commands include `refresh`, `history [LIMIT]`, and `use (--offset N | --ref REF | --latest)`).

Global options `--ref REF` and `--offset N` (mutually exclusive) load a specific manifest by reference or position in history before executing any subcommand. These are non-mutating (read-only manifest selection; they do not update `_CURRENT`). REPL commands `use` and `refresh` are also session-local and non-mutating. `rollback` is the only CLI command that mutates `_CURRENT`.

Additional global options: `--credentials PATH` (explicit credentials.toml), `--output-format json|jsonl|table|text`, `--s3-option KEY=VALUE` (repeatable, e.g. `addressing_style=path`), `--reader-backend slatedb|sqlite`, `--sqlite-mode {download,range,auto}` (default `auto`; only consulted for `sqlite` backend), `--sqlite-auto-per-shard-bytes BYTES` (default 16 MiB), `--sqlite-auto-total-bytes BYTES` (default 2 GiB). All four sqlite-mode flags override the corresponding `[reader]` keys (`sqlite_mode`, `sqlite_auto_per_shard_threshold_bytes`, `sqlite_auto_total_budget_bytes`) in `reader.toml`.

Key coercion: CLI keys are strings; when manifest uses integer encoding (`u64be`/`u32be`), keys are auto-coerced to `int` via `cli/config.py:coerce_cli_key()`.

The `health` subcommand outputs reader health status with exit codes: 0=healthy, 1=degraded, 2=unhealthy.

The `cleanup` subcommand deletes stale attempt directories and old run data from S3, supporting `--dry-run` for preview.

The `search` subcommand performs vector ANN search against a vector or unified snapshot, using `ShardedVectorReader` or `UnifiedShardedReader` automatically.

### Manifest & CURRENT Data Formats

**Manifest** (SQLite, published to `s3_prefix/manifests/{timestamp}_run_id={run_id}/manifest`, `format_version=4`):

A serialized SQLite database with two tables:
- `build_meta` — single row with `RequiredBuildMeta` fields (`run_id`, `num_dbs`, `s3_prefix`, `sharding` as JSON, `key_encoding`, `format_version=4`, …)
- `shards` — one row per non-empty shard (`db_id`, `db_url`, `checkpoint_id`, `row_count`, `db_bytes` (mandatory; 0 for SlateDB shards), …)
- `custom` — user-defined fields from `WriterManifestConfig.custom_manifest_fields` (JSON TEXT in `build_meta`)

Manifest S3 keys are timestamp-prefixed (e.g., `2026-03-14T10:30:00.000000Z_run_id=abc123/manifest`) for chronological listing via S3 `CommonPrefixes`. Readers strictly accept only `format_version=4`; older manifests fail with `ManifestParseError`.

**CURRENT pointer** (JSON, published to `s3_prefix/_CURRENT`):
```json
{"manifest_ref": "s3://bucket/.../manifest", "manifest_content_type": "application/x-sqlite3",
 "run_id": "...", "updated_at": "...", "format_version": 1}
```

Two-phase publish: manifest is written first, then `_CURRENT` pointer is updated. Both use `storage.put_bytes()` with exponential-backoff retries (3 attempts, 1s→2s→4s) for transient S3 errors.

**Run record** (YAML, published to `s3_prefix/<run_registry_prefix>/{timestamp}_run_id={run_id}_{uuid}/run.yaml`):
```yaml
format_version: 1
run_id: "..."
writer_type: "spark"  # or dask / ray / python / single-db variant
status: running       # running | succeeded | failed
started_at: "..."
updated_at: "..."
lease_expires_at: "..."
s3_prefix: "s3://bucket/prefix"
shard_prefix: "shards"
db_path_template: "db={db_id:05d}"
manifest_ref: null
error_type: null
error_message: null
```

Run records are operational writer metadata. Readers do not consume them. `BuildResult.run_record_ref` points to the record when one was published.

Default S3 run-registry resolution uses writer-level `credential_provider` and `s3_connection_options`. `manifest.credential_provider` and `manifest.s3_connection_options` are manifest-store scoped and are not used for run-registry S3 access. Pass an explicit `run_registry` if the run registry needs different settings.

**PostgreSQL manifest store** (`db_manifest_store.py:PostgresManifestStore`) uses `shardyfusion_manifests_builds` and `shardyfusion_manifests_shards` tables for build metadata and shard details (derived from the configurable `table_name` prefix, default `"shardyfusion_manifests"`), and an append-only `shardyfusion_pointer` table for current-pointer tracking (one INSERT per `publish()` or `set_current()` call). `load_current()` reads the newest pointer row. Accepts a zero-arg `connection_factory` callable for connection management (typically a psycopg2 connection pool).

### Key Abstractions

These are all Protocols, allowing user-provided implementations:

| Protocol | Default Implementation | Purpose |
|---|---|---|
| — | `SqliteManifestBuilder` | Manifest serialization (SQLite format, used internally by `S3ManifestStore`) |
| `StorageBackend` | `ObstoreBackend`, `MemoryBackend` | Sync S3/object-store I/O (get/put/delete/list) |
| `AsyncStorageBackend` | `AsyncObstoreBackend`, `AsyncMemoryBackend` | Async S3/object-store I/O (get/put/delete/list) |
| `ManifestStore` | `S3ManifestStore`, `PostgresManifestStore` | Unified manifest read/write (publish + load + list + set_current) |
| `RunRegistry` | `S3RunRegistry` | Run-scoped writer lifecycle record storage |
| `AsyncManifestStore` | `AsyncS3ManifestStore` | Async read-only manifest loading (load + list) |
| `AsyncShardReader` | `_SlateDbAsyncShardReader` | Async shard reader (get/close) |
| `AsyncShardReaderFactory` | `AsyncSlateDbReaderFactory` | Async factory for opening shard readers |
| `DbAdapterFactory` | `SlateDbFactory` | How to open shard databases |

SQLite backend alternatives (imported from `shardyfusion.sqlite_adapter`):
- **Writer**: `SqliteFactory` → `SqliteAdapter` (builds local SQLite DB, uploads to S3 on close; `checkpoint()` computes SHA-256 hash)
- **Reader (download-and-cache)**: `SqliteReaderFactory` → `SqliteShardReader` (one S3 GET, then local lookups)
- **Reader (range-read VFS)**: `SqliteRangeReaderFactory` → `SqliteRangeShardReader` (S3 range requests via `obstore` with page-aligned LRU cache; requires `apsw` and `obstore`; uses per-instance UUID-based VFS names to avoid collisions)
- **Reader (adaptive)**: `AdaptiveSqliteReaderFactory` — composes the two above and resolves to one per snapshot via `decide_access_mode(db_bytes_per_shard, per_shard_threshold, total_budget)` (OR-semantics; defaults 16 MiB / 2 GiB). Caches the resolved sub-factory keyed on `manifest.required_build.run_id`; rebuilt on `refresh()` when the run_id rotates. Pluggable policy via `SqliteAccessPolicy` Protocol + `make_threshold_policy()`.
- **Async wrappers**: `AsyncSqliteReaderFactory`, `AsyncSqliteRangeReaderFactory`, `AsyncAdaptiveSqliteReaderFactory` (delegate blocking I/O via `asyncio.to_thread`)

Vector backend alternatives:
- **LanceDB** (`vector/adapters/lancedb_adapter.py`): `LanceDbWriter` / `LanceDbWriterFactory` (build HNSW index locally, upload to S3), `LanceDbShardReader` / `LanceDbReaderFactory` (search via remote LanceDB dataset), `AsyncLanceDbShardReader` / `AsyncLanceDbReaderFactory` (async variants)
- **sqlite-vec** (`sqlite_vec_adapter.py`): `SqliteVecAdapter` / `SqliteVecFactory` (unified KV+vector in single SQLite file), `SqliteVecShardReader` / `SqliteVecReaderFactory` (KV get + vector search), `AsyncSqliteVecShardReader` / `AsyncSqliteVecReaderFactory` (async variants)
- **Composite** (`composite_adapter.py`): `CompositeAdapter` / `CompositeFactory` (wraps separate KV + vector adapters, LanceDB sidecar), `CompositeShardReader` / `CompositeReaderFactory`, `AsyncCompositeShardReader` / `AsyncCompositeReaderFactory`

`ManifestRef` (frozen dataclass in `manifest.py`) is the backend-agnostic pointer to a published manifest, carrying `ref`, `run_id`, and `published_at`. `ManifestStore.load_current()` returns `ManifestRef | None` (not `CurrentPointer`); the S3 implementation parses the `_CURRENT` JSON into a `ManifestRef` internally.

`ValueSpec` controls how DataFrame rows are serialized to bytes: `binary_col`, `json_cols`, or a callable encoder.

### Metrics & Observability

`MetricsCollector` (Protocol in `metrics/_protocol.py`) is an optional observer: set `observability.metrics_collector` on writer configs or pass it to the reader. The `MetricEvent` enum (in `metrics/_events.py`) defines events across writer lifecycle (e.g., `WRITE_STARTED`, `SHARD_WRITE_COMPLETED`), reader lifecycle (`READER_GET`, `READER_REFRESHED`), infrastructure (`S3_RETRY`, `RATE_LIMITER_THROTTLED`), writer retry (`SHARD_WRITE_RETRIED`, `SHARD_WRITE_RETRY_EXHAUSTED`), vector writer (`VECTOR_WRITE_STARTED`, `VECTOR_SHARD_WRITE_COMPLETED`, `VECTOR_WRITE_COMPLETED`), and vector reader (`VECTOR_READER_INITIALIZED`, `VECTOR_SEARCH`, `VECTOR_SHARD_SEARCH`, `VECTOR_READER_REFRESHED`, `VECTOR_READER_CLOSED`). Collectors receive `(event, payload_dict)` and should silently ignore unknown events for forward compatibility.

The `metrics/` package (`_events.py`, `_protocol.py`) is always available. Optional publishers require extras:
- **`PrometheusCollector`** (`metrics/prometheus.py`): requires `metrics-prometheus` extra. Uses isolated `CollectorRegistry` for test safety.
- **`OtelCollector`** (`metrics/otel.py`): requires `metrics-otel` extra. Accepts optional `meter_provider` for test isolation.

### Rate Limiting

Token-bucket rate limiter in `_rate_limiter.py`. `RateLimiter` Protocol + `TokenBucket` implementation.
- **KV ops/sec**: `KvWriteRateLimitConfig.max_writes_per_second` on `config.rate_limits` — limits write_batch calls.
- **KV bytes/sec**: `KvWriteRateLimitConfig.max_write_bytes_per_second` on `config.rate_limits` — limits payload bytes. Each uses an independent `TokenBucket` instance.
- **Vector ops/sec**: `VectorWriteRateLimitConfig.max_writes_per_second` on `VectorShardedWriteConfig.rate_limits` — limits vector shard write calls.
- **Reader-side**: readers accept `rate_limiter` parameter for reads/sec limiting. `AsyncShardedReader` calls `acquire_async()` directly on the limiter.
- **`try_acquire()`**: non-blocking, pure arithmetic — safe for async code without `to_thread()`.
- **`acquire_async()`**: part of the `RateLimiter` Protocol. Uses `asyncio.sleep` — safe for event loops.

### Logging

Structured logging in `logging.py`:
- **`get_logger(name)`**: returns a logger in the `shardyfusion.*` hierarchy.
- **`log_event()` / `log_failure()`**: emit structured log lines with `extra={"slatedb": fields}`.
- **`LogContext`**: context manager that binds fields to all log calls within its scope (via `contextvars`). Nesting supported.
- **`JsonFormatter`**: formats log records as single-line JSON with `timestamp`, `level`, `logger`, `event`, plus slatedb fields.
- **`configure_logging()`**: convenience to set handler/level on the `shardyfusion` logger hierarchy.

### Reader Health

`ReaderHealth` dataclass (in `reader/reader.py`) returned by `reader.health()`:
- `status` (`"healthy"`, `"degraded"`, `"unhealthy"`), `manifest_ref`, `manifest_age_seconds`, `num_shards`, `is_closed`.

### Key Encodings

The `key_encoding` field on `KeyValueWriteConfig` (`config.kv.key_encoding`, default `"u64be"`) controls how keys are serialized to bytes in SlateDB:

- **`u64be`** (default): 8-byte big-endian unsigned integer. Keys in `[0, 2^64-1]`.
- **`u32be`**: 4-byte big-endian unsigned integer. Keys in `[0, 2^32-1]`. Cuts key storage in half.
- **`utf8`**: UTF-8 encoded string keys. Variable length.
- **`raw`**: Raw bytes passed through without transformation.

Key encoding controls how keys are serialized to bytes in SlateDB storage. Hash routing is encoding-independent — it uses `canonical_bytes()` (int → 8-byte signed LE, str → UTF-8, bytes → passthrough) regardless of the storage encoding. The encoding is stored in the manifest and used by the reader/CLI for key coercion and lookup encoding.

### Sharding Strategies

- **Hash** (default): `xxh3_64(canonical_bytes(key), seed=0) % num_dbs` — supports int, string, and bytes keys uniformly. All writers use the same Python-based hash via `routing.xxh3_db_id()`. `num_dbs` is either provided explicitly or computed from `max_keys_per_shard` (= `ceil(count / max_keys_per_shard)`).
- **CEL**: User-provided CEL expression evaluated at write time (shard assignment) and read time (routing). The expression must produce **consecutive 0-based integer shard IDs** (e.g., `shard_hash(key) % 100u` produces IDs 0–99). A custom `shard_hash()` function is available in CEL expressions (wraps `xxh3_64`). `num_dbs` is always discovered from data (`max(db_id) + 1`); it must not be provided explicitly. Optional `boundaries` field enables `bisect_right`-based routing for range-like patterns. Requires `cel` extra (`cel-expr-python`). `ShardingSpec` fields: `cel_expr`, `cel_columns`, `boundaries` (optional).

The `SnapshotRouter` in `routing.py` mirrors the writer's sharding logic exactly for consistent key routing at read time. Manifests are sparse: only shards with data appear in the `shards` list. `SnapshotRouter.__init__` pads missing db_ids with synthetic `RequiredShardMeta(db_url=None, row_count=0)` entries so `router.shards[db_id]` always works.

## Critical Invariants

### Sharding Invariant (xxh3_64)

**This is the most important correctness property.** The reader and writer MUST compute identical shard IDs for every key. A violation means reads silently go to the wrong shard.

The invariant: `xxhash.xxh3_64_intdigest(canonical_bytes(key), seed=0) % num_dbs` where:
- **Hash**: xxh3_64 (from `xxhash` package), seed=0
- **Canonical bytes**: `int → 8-byte signed little-endian`, `str → UTF-8`, `bytes → as-is` (see `routing.canonical_bytes()`)
- **Modulo**: unsigned 64-bit digest `%` positive `num_dbs`
- **Key types**: int (signed 64-bit range), str, bytes — all supported uniformly
- **All writers use the same Python code** — Spark uses `mapInArrow`, not JVM-side hashing

**Safety mechanisms:**
1. `verify_routing_agreement()` in `writer/spark/writer.py`, `writer/dask/writer.py`, and `writer/ray/writer.py` — runtime spot-check comparing framework-assigned shard IDs vs Python routing on a sample of written rows. Controlled by `verify_routing=True` (default).
2. `tests/unit/writer/core/test_routing_contract.py` — hypothesis property tests (Python-only) for int/str/bytes keys. `tests/unit/writer/spark/test_routing_contract.py` — Spark `add_db_id_column` vs Python cross-checks. `tests/unit/writer/dask/test_dask_routing_contract.py` — Dask-vs-Python cross-checks. `tests/unit/writer/ray/test_ray_routing_contract.py` — Ray-vs-Python cross-checks.
3. Single implementation in `routing.py:xxh3_db_id()` imported by all writer paths (never reimplemented).

### Error Hierarchy

All package-specific errors inherit from `ShardyfusionError` and expose `retryable: bool`.

| Error | retryable | Source | Notes |
|---|---|---|---|
| `ShardyfusionError` | `False` | `errors.py` | Base class for package-specific exceptions |
| `ConfigValidationError` | `False` | `errors.py` | Invalid config or unsupported parameter combination |
| `ShardAssignmentError` | `False` | `errors.py` | Routing mismatch or invalid shard IDs |
| `ShardCoverageError` | `False` | `errors.py` | Writer results did not cover expected shard IDs |
| `ShardWriteError` | `True` | `errors.py` | Shard write failed with a potentially transient error; wraps unknown adapter exceptions |
| `DbAdapterError` | `False` | `errors.py` | SlateDB binding unavailable/incompatible or shard reader failure |
| `ManifestBuildError` | `False` | `errors.py` | Manifest builder failed during artifact creation |
| `PublishManifestError` | `True` | `errors.py` | Manifest upload failed; safe to retry |
| `PublishCurrentError` | `True` | `errors.py` | `_CURRENT` update failed after manifest publish |
| `ManifestParseError` | `False` | `errors.py` | Manifest or pointer payload malformed/invalid |
| `ReaderStateError` | `False` | `errors.py` | Reader used after close or before valid state exists |
| `PoolExhaustedError` | `True` | `errors.py` | Concurrent reader pool checkout timed out |
| `ManifestStoreError` | `True` | `errors.py` | Transient DB-backed manifest-store failure |
| `S3TransientError` | `True` | `errors.py` | Retryable S3 throttle/5xx/timeout |
| `VectorIndexError` | `False` | `errors.py` | Vector index construction or serialization failed |
| `VectorSearchError` | `False` | `errors.py` | Vector search operation failed |
| `SqliteAdapterError` | `False` | `sqlite_adapter.py` | SQLite adapter lifecycle or local DB misuse |

## Public API Summary

Top-level exports from `shardyfusion.__init__` (`__all__`, grouped only for readability):

- Config/build: `BaseShardedWriteConfig`, `HashShardedWriteConfig`, `CelShardedWriteConfig`, `SingleDbWriteConfig`, `WriterStorageConfig`, `WriterOutputConfig`, `WriterManifestConfig`, `KeyValueWriteConfig`, `HashShardingConfig`, `CelShardingConfig`, `WriterRetryConfig`, `KvWriteRateLimitConfig`, `VectorWriteRateLimitConfig`, `WriterObservabilityConfig`, `WriterLifecycleConfig`, `PythonRecordInput`, `ColumnWriteInput`, `VectorColumnInput`, `PythonWriteOptions`, `SparkWriteOptions`, `DaskWriteOptions`, `RayWriteOptions`, `SingleDbWriteOptions`, `SharedMemoryOptions`, `BufferingOptions`, `VectorSpec`, `UnifiedVectorWriteConfig`, `BuildDurations`, `BuildResult`, `BuildStats`, `CurrentPointer`, `WriterInfo`
- Credentials/connectivity: `CredentialProvider`, `EnvCredentialProvider`, `RetryConfig`, `S3ConnectionOptions`, `S3Credentials`, `StaticCredentialProvider`
- Errors: `ConfigValidationError`, `DbAdapterError`, `ManifestBuildError`, `ManifestParseError`, `ManifestStoreError`, `PoolExhaustedError`, `PublishCurrentError`, `PublishManifestError`, `ReaderStateError`, `S3TransientError`, `ShardAssignmentError`, `ShardCoverageError`, `ShardWriteError`, `ShardyfusionError`
- Manifest/store: `InMemoryManifestStore`, `ManifestArtifact`, `ManifestRef`, `ManifestShardingSpec`, `ManifestStore`, `ParsedManifest`, `RequiredBuildMeta`, `RequiredShardMeta`, `S3ManifestStore`, `SQLITE_MANIFEST_CONTENT_TYPE`, `SqliteManifestBuilder`, `SqliteShardLookup`, `load_sqlite_build_meta`, `parse_manifest_payload`, `parse_sqlite_manifest`
- Run registry: `InMemoryRunRegistry`, `RunRecord`, `RunRecordLifecycle`, `RunRegistry`, `RunStatus`, `S3RunRegistry`
- Reader/routing: `AsyncManifestStore`, `AsyncS3ManifestStore`, `AsyncShardReader`, `AsyncShardReaderFactory`, `AsyncShardReaderHandle`, `AsyncShardedReader`, `AsyncSlateDbReaderFactory`, `AsyncUnifiedShardedReader` (lazy import via `__getattr__`), `ConcurrentShardedReader`, `ReaderHealth`, `ShardDetail`, `ShardLookup`, `ShardReader`, `ShardReaderFactory`, `ShardReaderHandle`, `ShardedReader`, `SlateDbReaderFactory`, `SnapshotInfo`, `SnapshotRouter`, `UnifiedShardedReader` (lazy import via `__getattr__`)
- Adapters: `DbAdapter`, `DbAdapterFactory`, `SlateDbFactory`
- Rate limiting: `AcquireResult`, `RateLimiter`, `ThreadSafeTokenBucket`, `TokenBucket`
- Logging/metrics: `FailureSeverity`, `JsonFormatter`, `LogContext`, `MetricEvent`, `MetricsCollector`, `configure_logging`, `get_logger`
- Sharding/serialization: `CelColumn`, `CelType`, `KeyEncoding`, `ShardingSpec`, `ShardingStrategy`, `ValueSpec`, `cel_sharding`, `cel_sharding_by_columns`

Writer functions are imported from subpackages (not re-exported at top level):
- **Spark:** `from shardyfusion.writer.spark import write_hash_sharded, write_cel_sharded, write_single_db, DataFrameCacheContext, SparkConfOverrideContext`
- **Dask:** `from shardyfusion.writer.dask import write_hash_sharded, write_cel_sharded, write_single_db, DaskCacheContext`
- **Ray:** `from shardyfusion.writer.ray import write_hash_sharded, write_cel_sharded, write_single_db, RayCacheContext`
- **Python:** `from shardyfusion.writer.python import write_hash_sharded, write_cel_sharded` (KV + unified KV+vector via `VectorSpec` on `HashShardedWriteConfig` / `CelShardedWriteConfig`)

> **Hard break note:** KV writers now expose per-strategy functions (`write_hash_sharded` / `write_cel_sharded`) with four-argument public signatures: `(data, config, input, options=None)`. Use `HashShardedWriteConfig` / `CelShardedWriteConfig` plus `PythonRecordInput` or `ColumnWriteInput`. Standalone vector writers use `write_sharded(records, config, input=None, options=None)` from `shardyfusion.vector`; distributed vector writers use `write_sharded(data, config, input, options=None)` from `shardyfusion.writer.{spark,dask,ray}.vector_writer` or from the framework package. The package-level `write_vector_sharded` name remains as a compatibility alias to the refactored vector writer.

SQLite backend adapters are imported from their module (not re-exported at top level):
- `from shardyfusion.sqlite_adapter import SqliteFactory, SqliteReaderFactory, SqliteRangeReaderFactory, AdaptiveSqliteReaderFactory`
- `from shardyfusion.sqlite_adapter import AsyncSqliteReaderFactory, AsyncSqliteRangeReaderFactory, AsyncAdaptiveSqliteReaderFactory`

Vector types and functions are imported from subpackages (not re-exported at top level):
- `from shardyfusion.vector.writer import write_sharded`
- `from shardyfusion.vector.reader import ShardedVectorReader`
- `from shardyfusion.vector.async_reader import AsyncShardedVectorReader`
- `from shardyfusion.vector.types import VectorRecord, SearchResult, VectorSearchResponse, DistanceMetric, VectorShardingStrategy`
- `from shardyfusion.vector.config import VectorIndexConfig, VectorShardingConfig, VectorShardingSpec, VectorShardedWriteConfig, VectorRecordInput, VectorWriteOptions`
- `from shardyfusion.writer.spark import write_sharded` or `from shardyfusion.writer.spark.vector_writer import write_sharded` for Spark vector writes
- `from shardyfusion.writer.dask import write_sharded` or `from shardyfusion.writer.dask.vector_writer import write_sharded` for Dask vector writes
- `from shardyfusion.writer.ray import write_sharded` or `from shardyfusion.writer.ray.vector_writer import write_sharded` for Ray vector writes

## Testing Guidelines

- Framework: `pytest`; parallel unit execution uses `pytest-xdist`.
- Test files: `test_*.py`; keep reader and writer tests in their flavor directories.
- Prefer targeted runs while developing:
  - `just unit -- tests/unit/writer`
  - `just unit -- tests/unit/cli`
  - `just integration -- tests/integration/read`
- Integration writer tests require Spark + Java and local S3 emulation (`moto`).
- E2E tests require a container engine and run via `just d-e2e` (preferred). Use `-p=2` to run two tox envs in parallel.
- Contract tests in `tests/unit/writer/core/test_routing_contract.py` verify writer-reader sharding invariant with hypothesis property tests.
- Shared test scenarios in `tests/helpers/s3_test_scenarios.py` are reused across moto and Garage backends.
- Test adapters in `testing.py`: `FakeSlateDbAdapter`, `FileBackedSlateDbAdapter`, `RealSlateDbFileAdapter`.
- Markers: `@pytest.mark.spark`, `@pytest.mark.dask`, `@pytest.mark.ray`, `@pytest.mark.cel`, `@pytest.mark.e2e`, `@pytest.mark.vector`, `@pytest.mark.vector_sqlite`.
- Optional-dep test isolation: Writer test directories use `conftest.py`-level `pytest.importorskip()` to skip entire suites when the framework extra is not installed.
- For behavior changes, add/adjust unit tests first, then integration tests where routing/publishing or Spark behavior is affected.
- **`tests/unit/metrics/`** — `PrometheusCollector` and `OtelCollector` tests. Run via dedicated tox envs (`prometheus-unit`, `otel-unit`) that install the `metrics-prometheus`/`metrics-otel` extras. OTel tests also need `opentelemetry-sdk` (provided via `test` extra). These tests require their respective extras and will fail with `ImportError` without them — run via tox or install the extras explicitly.
- **`tests/unit/backend/sqlite/`** — SQLite adapter unit tests (`test_sqlite_adapter.py`, `test_sqlite_range_reader.py`). Range-read tests require `apsw`.
- **`tests/unit/backend/slatedb/`** — SlateDB adapter unit tests.
- **`tests/unit/vector/`** — Vector search unit tests (types, config, sharding, merge, writer, reader, async reader, adapters, unified reader, CEL vector routing). Included in the `core-unit` tox env. Tests requiring `lancedb` or `sqlite-vec` are skipped via `pytest.importorskip()` when extras are not installed.
- **`tests/integration/`** — S3 via `moto`; Spark writer tests require Spark + Java. `backend/sqlite/` tests SQLite adapter against moto S3. `vector/` tests vector write→read round-trip (mock adapters).
- **`tests/e2e/`** — Garage S3 via compose; `just d-e2e`. Includes `cli/` and `vector/` E2E tests.
- **`tests/helpers/`** — `s3_test_scenarios.py` (shared scenarios for moto/Garage), `smoke_scenarios.py` (shared smoke E2E data and scenarios for HASH/CEL sharding across all writer frameworks), `run_record_assertions.py` (shared run-record content assertions), `tracking.py` (test doubles: `TrackingAdapter`, `TrackingFactory`, `RecordingTokenBucket`, `InMemoryPublisher`), `vector_s3_test_scenarios.py` (shared vector E2E scenarios)
- **CEL tests**: require `cel` extra (`cel-expr-python`). Use `@pytest.mark.cel` marker. Skipped via `pytest.importorskip()` when the extra is not installed.
- **Cleanup tests**: `tests/unit/cli/test_cleanup.py` tests the `cleanup` CLI command; `tests/unit/writer/core/test_cleanup_core.py` tests core cleanup functions (`cleanup_stale_attempts`, `cleanup_old_runs`)
- **Contract tests**: hypothesis property tests in `tests/unit/writer/core/test_routing_contract.py`, framework cross-checks in `writer/spark/`, `writer/dask/`, and `writer/ray/`

## Coding Style & Naming Conventions

- Python 3.11+ with full type hints (pyright + ruff enforced). `E501` ignored (long lines allowed). See `pyproject.toml` for full ruff config.
- Keep Spark logic in writer-side modules; avoid Python UDFs when Spark built-ins are available.
- Dataclasses with `slots=True` for performance-sensitive models.
- Manifest models use Pydantic (`ConfigDict(use_enum_values=False)`) — enum names serialize, not values.
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Commit subjects: imperative, Conventional Commit prefixes recommended (`fix:`, `feat:`, `chore:`, `test:`)

## Commit & Pull Request Guidelines

- **Before requesting a review or concluding changes are ready, always run `just ci d-e2e` to ensure all tests (including E2E) pass.**
- Follow concise, imperative commit subjects; Conventional Commit prefixes are used in history (`fix:`, `chore:`, `test:`, `feat:`) and recommended.
- Keep commits scoped (single concern) and include tests for functional changes.
- PRs should include:
  - What changed and why
  - Impacted area (`read`, `writer`, `cli`, or shared)
  - Commands run (e.g., `tox -m quality`, targeted/unit/integration tests)

## Gotchas & Non-obvious Behavior

- **SlateDB import is deferred**: `slatedb_adapter.py` imports the `slatedb` module inside `__init__`, not at module load. Tests monkeypatch `sys.modules["slatedb"]` to provide fakes.
- **APSW import is deferred**: `sqlite_adapter.py` imports `apsw` inside `SqliteRangeShardReader.__init__`, and `_sqlite_vfs.py` imports it inside `create_apsw_vfs()`. Only the range-read VFS tier requires `apsw`; the download-and-cache tier uses stdlib `sqlite3`.
- **obstore import is deferred**: `_sqlite_vfs.py` imports `obstore` (and `obstore.store.S3Store`) inside `S3ReadOnlyFile.__init__` / `_build_obstore_client` / `_fetch_pages`. Only the range-read VFS tier requires `obstore`.
- **SQLite adapter lifecycle**: `SqliteAdapter` builds a local SQLite DB during writes, then uploads the `.db` file to S3 on `close()`. The reader downloads (or range-reads) the file from S3. The `checkpoint()` method computes a SHA-256 hash of the DB file as the checkpoint ID.
- **SQLite range-read VFS uses per-instance VFS names**: Each `SqliteRangeShardReader` registers a unique APSW VFS (via `uuid4`) to avoid name collisions when multiple readers are open simultaneously. The VFS is unregistered on `close()`.
- **Python writer parallel mode uses `spawn`**: `multiprocessing.get_context("spawn")` is hardcoded. All objects passed to workers must be picklable. Min/max key tracking happens in the main process. Workers communicate via shared memory; configurable budgets via `max_parallel_shared_memory_bytes` and `max_parallel_shared_memory_bytes_per_worker`.
- **Rate limiter thread safety**: `TokenBucket` has no internal lock. Use `ThreadSafeTokenBucket` when sharing a limiter across threads (e.g. `ConcurrentShardedReader`). `ThreadSafeTokenBucket` locks around `try_acquire()` and loops with sleep outside the lock. Writers and readers depend on the `RateLimiter` Protocol, not the concrete class. KV writers support `KvWriteRateLimitConfig` with both ops/sec and bytes/sec limits via independent `TokenBucket` instances. Vector writers support `VectorWriteRateLimitConfig` with an ops/sec limit. Readers accept a `rate_limiter` parameter for reads/sec limiting.
- **Reader reference counting**: `refresh()` serialises I/O via `_refresh_lock`, atomically swaps `_ReaderState` with a compare-and-swap guard, and defers closing old handles until `refcount` drops to zero. Pool-mode checkout has a configurable timeout (default 30s) — raises `PoolExhaustedError` (retryable) when exhausted. Metadata methods (`key_encoding`, `snapshot_info`, `shard_details`, `route_key`) use `_use_state()` and raise `ReaderStateError` on a closed reader.
- **Borrow handle safety nets**: `ShardReaderHandle` and `AsyncShardReaderHandle` implement `__del__()` that logs a warning and releases the borrow if the handle was not explicitly closed. Prefer explicit `close()` or `with`/`async with` context managers.
- **Async reader uses `open()` classmethod**: `AsyncShardedReader.__init__` does no I/O. Use `await AsyncShardedReader.open(...)` and then `await reader.close()`, or `async with await AsyncShardedReader.open(...) as reader`. The default `AsyncS3ManifestStore` requires `aiobotocore` (`read-slatedb-async`, or another install path that brings it in).
- **Async manifest store**: `AsyncShardedReader` accepts `AsyncManifestStore` only (not sync `ManifestStore`). When `manifest_store=None`, uses `AsyncS3ManifestStore` (native `AsyncObstoreBackend`).
- **Reader cold-start fallback**: When `load_manifest()` raises `ManifestParseError` during initialization, readers walk backward through `list_manifests()` to find a valid manifest (up to `max_fallback_attempts`, default 3). Set to 0 to disable fallback. During `refresh()`, a malformed manifest is silently skipped and the reader keeps its current good state (returns `False`).
- **Garage requires path-style addressing**: E2E tests set `addressing_style: "path"` in `S3ClientConfig` because Garage doesn't support virtual-hosted-style.
- **Session-scoped test fixtures**: PySpark and S3 (moto/Garage) fixtures are session-scoped for performance. Tests share the same Spark session and S3 service.
- **Writer scenario imports are deferred**: `tests/helpers/s3_test_scenarios.py` imports writer modules inside function bodies so reader-only test collection doesn't fail.
- **Ray writer temporarily sets `DataContext.shuffle_strategy`**: During repartition, the Ray writer sets `DataContext.shuffle_strategy = "HASH_SHUFFLE"` and restores the previous value in a `try/finally` block, protected by `_SHUFFLE_STRATEGY_LOCK` (`threading.Lock`) to guard against concurrent Ray writer calls in the same driver.
- **Ray tests need `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`**: Ray ≥ 2.47 auto-detects `uv run` in the process tree and overrides worker Python to create fresh venvs. The env var is set in `tox.ini` for raywriter envs. For direct pytest, either set the var or use `uv run --extra writer-ray-slatedb pytest ...`.
- **`acquire_async()` is part of the `RateLimiter` Protocol**: Uses `asyncio.sleep` — safe for event loops. The sync `acquire()` uses `time.sleep` and must not be called from async code.
- **`try_acquire()` is pure arithmetic**: Guaranteed non-blocking (replenish + compare + deduct). Safe to call from async code directly without `to_thread()`.
- **`RetryConfig` usage**: Used in `WriterRetryConfig.shard_retry` (`config.retry.shard_retry`, also exposed as `config.shard_retry`) for retryable writer paths. `shard_retry` currently covers Dask/Ray sharded writes, Spark/Dask/Ray `write_single_db()`, and Python parallel writes. Spark sharded writes still rely on Spark task retry/speculation. When `shard_retry` is set, the retry path writes each attempt to a fresh S3 prefix (`attempt=00`, `attempt=01`, ...). S3 I/O retries are delegated to obstore's Rust layer with shardyfusion-tuned aggressive defaults (`max_retries=5`, `init_backoff=200ms`, `max_backoff=4s`, `retry_timeout=30s`). A single Python-level fallback retry (2s delay) is applied for PUT operations on `GenericError` only.
- **`cel-expr-python` does not support Python 3.14**: The `cel` extra depends on `cel-expr-python` which has no py3.14 wheels. CEL tests are skipped on py3.14 via `pytest.importorskip()`.
- **UnifiedShardedReader and AsyncUnifiedShardedReader are lazily imported**: `__init__.py` uses `__getattr__` to defer the import of both classes (and their numpy transitive dependency) until first access. This prevents `import shardyfusion` from requiring numpy for non-vector users.
- **LanceDB and sqlite-vec imports are deferred**: `lancedb_adapter.py` imports `lancedb.index` inside class methods. `sqlite_vec_adapter.py` imports `sqlite_vec` inside `_load_sqlite_vec()`. Both are optional dependencies.
- **Vector adapters use always-synthetic internal IDs**: LanceDB and sqlite-vec adapters assign monotonically increasing internal IDs to prevent cross-batch collisions between int and string user IDs (e.g., `42` vs `"42"`). The original ID type is preserved via typed JSON in the `id_map`/`vec_id_map` table (`{"v": <value>, "t": "int"|"str"}`).
- **CEL compilation is cached in vector routing**: `route_vector_to_shards()` caches compiled CEL expressions at module level (keyed on expression + columns) to avoid recompilation on every search call.
- **Vector distance semantics**: All backends (LanceDB, sqlite-vec) return distances (lower = more similar) for all metrics including dot product. `merge_results()` uses `heapq.nsmallest` uniformly. There is no `nlargest` path.
- **Composite adapter lifecycle ordering**: `CompositeAdapter` closes the vector writer before the KV writer on `close()` and `flush()` — this ensures the lance index is finalized before the KV adapter uploads. At read time, `CompositeShardReader` opens the KV reader first, then the vector reader.
- **PostgresManifestStore connection factory**: Accepts a zero-arg `connection_factory` callable (not a connection or pool). Each method call (`publish`, `load_current`, `list_manifests`, etc.) creates a new connection and closes it in a `try/finally` block. This allows sharing a connection pool across calls (the factory returns a pooled connection). Table creation (`CREATE TABLE IF NOT EXISTS`) happens lazily on first access when `ensure_table=True`.

## Environment & Configuration Notes

- Spark-based writer flows require **Java 17** (`JAVA_HOME` or `PATH`). Java 21 is not supported with Spark 3.5 (Arrow JVM module access issues). CI uses temurin distribution.
- Spark writer uses `mapInArrow` which requires `pandas>=2.2` and `pyarrow>=15.0` (both included in the `writer-spark-slatedb` and `writer-spark-sqlite` extras).
- `SPARK_LOCAL_IP=127.0.0.1` is set in tox and devcontainer to avoid hostname resolution issues.
- Reader-only, Python writer, Dask writer, and Ray writer usage do not require Spark/Java.
- **Ray does not run on py3.14** (same as Dask and CEL).
- **obstore** (`obstore>=0.4`) is a core dependency used for all S3 I/O (sync and async). No separate `boto3` or `aiobotocore` installation is required.
