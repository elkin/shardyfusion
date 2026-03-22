# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`shardyfusion` is a sharded snapshot writer/reader library. It provides:
- Writer-side Spark APIs to build `num_dbs` independent shard databases
- Dask DataFrame-based writer (no Spark/Java needed)
- Ray Data-based writer (no Spark/Java needed)
- Pure-Python iterator-based writer (no Spark/Java needed)
- Pluggable storage backends: SlateDB (default) and SQLite-on-S3
- Manifest + `_CURRENT` publishing protocol (default S3, pluggable interfaces)
- Reader-side routing helpers for service-side `get` and `multi_get`
- `shardy` CLI for interactive and batch lookups against published snapshots
- Token-bucket rate limiting for all writer paths (`max_writes_per_second`, `max_write_bytes_per_second`)

## Tooling

**Package manager**: uv. Extras: `read`, `read-async` (adds aiobotocore for native async S3), `read-sqlite`, `read-sqlite-range` (adds apsw), `sqlite-async`, `writer-spark` (requires Java), `writer-spark-sqlite`, `writer-python`, `writer-python-sqlite`, `writer-dask`, `writer-dask-sqlite`, `writer-ray`, `writer-ray-sqlite`, `cli`, `cel` (adds cel-expr-python), `metrics-prometheus`, `metrics-otel`. Core dependency: `pyyaml>=6.0`. Full dev: `uv sync --all-extras --dev`.

**Workflows**: Run `just --list` for all local and container (`d-*`) targets. Key entry points: `just setup` (bootstrap a fresh clone), `just doctor` (verify environment), `just fix` (auto-format), `just ci` (quality + unit + integration), `just clean` / `just clean-all` (remove caches/artifacts). Container default engine is Podman; override with `CONTAINER_ENGINE=docker`.

**Test matrix**: tox labels — `quality`, `unit`, `integration`, `e2e`. Targeted pytest: `uv run pytest -q tests/unit/<area>` (areas: `shared`, `read`, `read_async`, `writer`, `backend/sqlite`, `metrics`, `cli`, `cel`).

**Type checking**: `uv run pyright shardyfusion` (all code) or per-path configs in `pyright/` directory.

**JSON schemas**: Available on demand via `shardy schema` (manifest) and `shardy schema --type current-pointer`. Generated at runtime from Pydantic models (`ParsedManifest.model_json_schema()` / `CurrentPointer.model_json_schema()`).

Container venv is at `/opt/shardyfusion-venv`, not the host `.venv`.

## Commands

Common local commands:
- `just setup`, `just doctor`, `just sync`, `just fix`, `just docs`, `just ci-matrix`
- `just quality`, `just unit`, `just integration`, `just ci`, `just d-e2e`
- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright shardyfusion`, `uv build`
- `uv run tox -m quality`, `uv run tox -m unit`, `uv run tox -m integration`, `uv run tox -m e2e`
- `uv run pytest -q tests/unit/{shared,read,read_async,writer,backend/sqlite,metrics,cli,cel}`

## CI Pipeline (GitHub Actions)

On every push/PR: **quality → package → unit → integration** (parallel within each stage). E2E is local/container only (`just d-e2e`). Java 17 (temurin) for Spark jobs. **Dask, Ray, and CEL do not run on py3.14.** Weekly scheduled build on Mondays at 06:00 UTC.

## Architecture

The library is split into six independent paths that share config, manifest models, and core logic:

**Writer path (Spark)** (requires PySpark + Java):
`writer/spark/writer.py` → `writer/spark/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)
`writer/spark/single_db_writer.py` (single-shard variant, distributed sorting)
`writer/spark/util.py` (`DataFrameCacheContext`)

**Writer path (Dask)** (no Spark/Java needed):
`writer/dask/writer.py` → `writer/dask/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)
`writer/dask/single_db_writer.py` (single-shard variant, distributed sorting)

**Writer path (Ray)** (no Spark/Java needed):
`writer/ray/writer.py` → `writer/ray/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)
`writer/ray/single_db_writer.py` (single-shard variant, distributed sorting)

**Writer path (Python)** (no Spark/Java needed):
`writer/python/writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` (or `sqlite_adapter.py`)

**Reader path (sync)** (no Spark/Java needed):
`reader/reader.py` (ShardedReader) / `reader/concurrent_reader.py` (ConcurrentShardedReader) → `reader/_base.py` → `reader/_types.py`, `reader/_state.py` → `routing.py` → `manifest_store.py`, `db_manifest_store.py`

**Reader path (async)** (no Spark/Java needed; optional `aiobotocore` for native async S3):
`reader/async_reader.py` → `routing.py` → `async_manifest_store.py` (`AsyncS3ManifestStore`)

**CLI path** (requires click + pyyaml):
`cli/app.py` → `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`

### Module Dependency Graph

```
Layer 0 — Core types & errors: errors.py, type_defs.py (incl. RetryConfig), sharding_types.py, ordering.py, logging.py (incl. LogContext/JsonFormatter), metrics/ (package: _events.py, _protocol.py)
Layer 1 — Config & serialization: config.py, serde.py, _rate_limiter.py, cel.py (CEL compile/evaluate/boundaries, optional cel extra)
Layer 2 — Storage, routing, manifest: storage.py (w/ RetryConfig, list_prefixes), manifest.py, routing.py, manifest_store.py (delegates to list_prefixes), async_manifest_store.py, db_manifest_store.py
Layer 3 — Writer core: _writer_core.py (shared by all writers; empty_shard_result, cleanup: CleanupAction, cleanup_stale_attempts, cleanup_old_runs)
Layer 4 — Entry points: writer/{spark,dask,ray,python}/*.py, reader/ (reader.py, concurrent_reader.py, _base.py, _types.py, _state.py), reader/async_reader.py, cli/app.py
Layer 5 — Adapters & testing: slatedb_adapter.py, sqlite_adapter.py (optional apsw for range-read VFS), testing.py
Layer opt — Optional publishers: metrics/prometheus.py (metrics-prometheus extra), metrics/otel.py (metrics-otel extra)
```

### Write Pipeline

**Spark writer** (`write_sharded`):
1. Entry point in `writer/spark/writer.py`. Optionally applies Spark conf overrides via `SparkConfOverrideContext`.
2. `writer/spark/sharding.py` adds `_slatedb_db_id` column via `mapInArrow` (hash or CEL, all Python-based), then converts the DataFrame to a pair RDD partitioned so partition index = db_id.
3. Each partition writes one shard to S3 at a shard path (`shards/run_id=.../db=XXXXX/attempt=YY/`).
4. The driver streams results via `toLocalIterator()` and selects deterministic winners (lowest attempt → task_attempt_id → URL).
5. A manifest artifact is built and published, and the `_CURRENT` pointer is updated.

**Dask writer** (`write_sharded`):
1. Entry point in `writer/dask/writer.py`. Accepts `dd.DataFrame` with `key_col`/`value_spec`.
2. `writer/dask/sharding.py` adds `_slatedb_db_id` column via Python routing function applied per partition. For CEL sharding, uses `pandas_rows_to_contexts()` + `route_cel()` with full row context (supporting non-key columns).
3. Shuffles by `_slatedb_db_id`, then `map_partitions` writes each shard. Empty shards (no rows in partition) are omitted from the manifest; `select_winners()` filters them out.
4. Optional rate limiting via `max_writes_per_second` (token-bucket). Routing verification via `verify_routing_agreement()`.
5. Uses the same `_writer_core.py` functions for winner selection, manifest building, and publishing.

**Ray writer** (`write_sharded`):
1. Entry point in `writer/ray/writer.py`. Accepts `ray.data.Dataset` with `key_col`/`value_spec`.
2. `writer/ray/sharding.py` adds `_slatedb_db_id` column via Arrow batch format (`map_batches` with `batch_format="pyarrow"`, `zero_copy_batch=True`). For CEL sharding, uses `route_cel_batch()` directly on Arrow batches (same approach as Spark).
3. Repartitions by `_slatedb_db_id` using hash shuffle (`DataContext.shuffle_strategy = "HASH_SHUFFLE"`; saved/restored). Then `map_batches` writes each shard with `batch_format="pandas"`. Empty shards (no rows in partition) are omitted from the manifest; `select_winners()` filters them out.
4. Optional rate limiting via `max_writes_per_second` (token-bucket). Routing verification via `_verify_routing_agreement()`.
5. Uses the same `_writer_core.py` functions for winner selection, manifest building, and publishing.

**Python writer** (`write_sharded`):
1. Entry point in `writer/python/writer.py`. Accepts `Iterable[T]` with `key_fn`/`value_fn` callables.
2. Supports single-process (all adapters open simultaneously via `contextlib.ExitStack`) or multi-process (`parallel=True`, one worker per shard via `multiprocessing.spawn`).
3. Optional rate limiting via `max_writes_per_second` and `max_write_bytes_per_second` (token-bucket in `_rate_limiter.py`). Single-process mode supports global memory ceilings via `max_total_batched_items` and `max_total_batched_bytes` to prevent OOM when buffering across many shards.
4. Uses the same `_writer_core.py` functions for routing, winner selection, manifest building, and publishing.

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
2. S3 manifest loading uses `AsyncS3ManifestStore` (native aiobotocore).
3. `async get(key)` / `async multi_get(keys)` route keys and read from async shard readers. Both hold borrow count increments during reads, preventing `refresh()`/`close()` from closing shard readers while reads are in-flight. The borrow is acquired after the rate limiter to avoid blocking state cleanup during throttling. `multi_get` fans out via `asyncio.TaskGroup` with optional `asyncio.Semaphore` for concurrency limiting (`max_concurrency`).
4. `async refresh()` serialises via `asyncio.Lock`. Swaps state and defers cleanup of retired readers via `state.aclose()` (scheduled as a task via `loop.create_task()`) when their borrow count hits 0. `_AsyncReaderState` supports the async context manager protocol. A malformed manifest is silently skipped (returns `False`), keeping the previous good state.
5. Sync metadata methods (`key_encoding`, `snapshot_info()`, `shard_details()`, `route_key()`) and borrow methods (`reader_for_key()`, `readers_for_keys()`) work identically to the sync reader.
6. Supports `async with` context manager via `__aenter__`/`__aexit__`.

### CLI (`shardy`)

Entry point: `cli/app.py:main`. Subcommands: `get KEY`, `multiget KEY [KEY ...]`, `info`, `shards`, `route KEY`, `exec --script FILE`, `history [--limit N]`, `rollback (--ref REF | --run-id RUN_ID | --offset N)`, `cleanup [--dry-run] [--include-old-runs] [--older-than DURATION] [--keep-last N] [--max-retries N]`, `schema [--type manifest|current-pointer]`. No subcommand → interactive REPL (`cmd.Cmd` with `shardy> ` prompt; REPL-only commands include `refresh`, `history [LIMIT]`, and `use (--offset N | --ref REF | --latest)`).

Global options `--ref REF` and `--offset N` (mutually exclusive) load a specific manifest by reference or position in history before executing any subcommand. These are non-mutating (read-only manifest selection; they do not update `_CURRENT`). REPL commands `use` and `refresh` are also session-local and non-mutating. `rollback` is the only CLI command that mutates `_CURRENT`.

Key coercion: CLI keys are strings; when manifest uses integer encoding (`u64be`/`u32be`), keys are auto-coerced to `int` via `cli/config.py:coerce_cli_key()`.

### Manifest & CURRENT Data Formats

**Manifest** (YAML, published to `s3_prefix/manifests/{timestamp}_run_id={run_id}/manifest`):
```yaml
required:  # RequiredBuildMeta: run_id, num_dbs, s3_prefix, sharding, key_encoding, ...
shards:    # list of RequiredShardMeta: db_id, db_url, checkpoint_id, row_count, ...
custom:    # user-defined fields from ManifestOptions.custom_manifest_fields
```

Manifest S3 keys are timestamp-prefixed (e.g., `2026-03-14T10:30:00.000000Z_run_id=abc123/manifest`) for chronological listing via S3 `CommonPrefixes`.

**CURRENT pointer** (JSON, published to `s3_prefix/_CURRENT`):
```json
{"manifest_ref": "s3://bucket/.../manifest.yaml", "manifest_content_type": "application/x-yaml",
 "run_id": "...", "updated_at": "...", "format_version": 1}
```

Two-phase publish: manifest is written first, then `_CURRENT` pointer is updated. Both use `storage.put_bytes()` with exponential-backoff retries (3 attempts, 1s→2s→4s) for transient S3 errors.

**DB manifest stores** use a `shardyfusion_manifests` table for manifest payloads and an append-only `shardyfusion_pointer` table for current-pointer tracking (one INSERT per `publish()` or `set_current()` call). `load_current()` reads the newest pointer row; falls back to the newest manifest row if the pointer table is empty.

### Key Abstractions

These are all Protocols, allowing user-provided implementations:

| Protocol | Default Implementation | Purpose |
|---|---|---|
| `ManifestBuilder` | `YamlManifestBuilder` | Manifest serialization format |
| `ManifestStore` | `S3ManifestStore` | Unified manifest read/write (publish + load + list + set_current) |
| `AsyncManifestStore` | `AsyncS3ManifestStore` | Async read-only manifest loading (load + list) |
| `AsyncShardReader` | `_SlateDbAsyncShardReader` | Async shard reader (get/close) |
| `AsyncShardReaderFactory` | `AsyncSlateDbReaderFactory` | Async factory for opening shard readers |
| `DbAdapterFactory` | `SlateDbFactory` | How to open shard databases |

SQLite backend alternatives (imported from `shardyfusion.sqlite_adapter`):
- **Writer**: `SqliteFactory` → `SqliteAdapter` (builds local SQLite DB, uploads to S3 on close)
- **Reader (download-and-cache)**: `SqliteReaderFactory` → `SqliteShardReader` (one S3 GET, then local lookups)
- **Reader (range-read VFS)**: `SqliteRangeReaderFactory` → `SqliteRangeShardReader` (S3 Range requests with LRU page cache; requires `apsw`)
- **Async wrappers**: `AsyncSqliteReaderFactory`, `AsyncSqliteRangeReaderFactory` (delegate blocking I/O via `asyncio.to_thread`)

`ManifestRef` (frozen dataclass in `manifest.py`) is the backend-agnostic pointer to a published manifest, carrying `ref`, `run_id`, and `published_at`. `ManifestStore.load_current()` returns `ManifestRef | None` (not `CurrentPointer`); the S3 implementation parses the `_CURRENT` JSON into a `ManifestRef` internally.

`ValueSpec` controls how DataFrame rows are serialized to bytes: `binary_col`, `json_cols`, or a callable encoder.

### Metrics & Observability

`MetricsCollector` (Protocol in `metrics/_protocol.py`) is an optional observer: set `WriteConfig.metrics_collector` or pass it to the reader. The `MetricEvent` enum (in `metrics/_events.py`) defines events across writer lifecycle (e.g., `WRITE_STARTED`, `SHARD_WRITE_COMPLETED`), reader lifecycle (`READER_GET`, `READER_REFRESHED`), and infrastructure (`S3_RETRY`, `RATE_LIMITER_THROTTLED`). Collectors receive `(event, payload_dict)` and should silently ignore unknown events for forward compatibility.

The `metrics/` package (`_events.py`, `_protocol.py`) is always available. Optional publishers require extras:
- **`PrometheusCollector`** (`metrics/prometheus.py`): requires `metrics-prometheus` extra. Uses isolated `CollectorRegistry` for test safety.
- **`OtelCollector`** (`metrics/otel.py`): requires `metrics-otel` extra. Accepts optional `meter_provider` for test isolation.


### Rate Limiting

Token-bucket rate limiter in `_rate_limiter.py`. `RateLimiter` Protocol + `TokenBucket` implementation.
- **ops/sec**: `max_writes_per_second` on `WriteConfig` — limits write_batch calls.
- **bytes/sec**: `max_write_bytes_per_second` on `WriteConfig` — limits payload bytes. Each uses an independent `TokenBucket` instance.
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

The `key_encoding` field on `WriteConfig` (default `"u64be"`) controls how keys are serialized to bytes in SlateDB:

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
2. `tests/unit/writer/test_routing_contract.py` — hypothesis property tests (Python-only) for int/str/bytes keys. `tests/unit/writer/spark/test_routing_contract.py` — Spark `add_db_id_column` vs Python cross-checks. `tests/unit/writer/dask/test_dask_routing_contract.py` — Dask-vs-Python cross-checks. `tests/unit/writer/ray/test_ray_routing_contract.py` — Ray-vs-Python cross-checks.
3. Single implementation in `routing.py:xxh3_db_id()` imported by all writer paths (never reimplemented).

### Error Hierarchy

All package-specific errors inherit from `ShardyfusionError` and expose `retryable: bool`.

| Error | retryable | Source | Notes |
|---|---|---|---|
| `ShardyfusionError` | `False` | `errors.py` | Base class for package-specific exceptions |
| `ConfigValidationError` | `False` | `errors.py` | Invalid config or unsupported parameter combination |
| `ShardAssignmentError` | `False` | `errors.py` | Routing mismatch or invalid shard IDs |
| `ShardCoverageError` | `False` | `errors.py` | Writer results did not cover expected shard IDs |
| `SlateDbApiError` | `False` | `errors.py` | SlateDB binding unavailable/incompatible or shard reader failure |
| `ManifestBuildError` | `False` | `errors.py` | Manifest builder failed during artifact creation |
| `PublishManifestError` | `True` | `errors.py` | Manifest upload failed; safe to retry |
| `PublishCurrentError` | `True` | `errors.py` | `_CURRENT` update failed after manifest publish |
| `ManifestParseError` | `False` | `errors.py` | Manifest or pointer payload malformed/invalid |
| `ReaderStateError` | `False` | `errors.py` | Reader used after close or before valid state exists |
| `PoolExhaustedError` | `True` | `errors.py` | Concurrent reader pool checkout timed out |
| `ManifestStoreError` | `True` | `errors.py` | Transient DB-backed manifest-store failure |
| `S3TransientError` | `True` | `errors.py` | Retryable S3 throttle/5xx/timeout |
| `SqliteAdapterError` | `False` | `sqlite_adapter.py` | SQLite adapter lifecycle or local DB misuse |

## Public API Summary

Top-level exports from `shardyfusion.__init__` (`__all__`, grouped only for readability):

- Config/build: `BuildDurations`, `BuildResult`, `BuildStats`, `CurrentPointer`, `ManifestOptions`, `OutputOptions`, `WriteConfig`, `WriterInfo`
- Credentials/connectivity: `CredentialProvider`, `EnvCredentialProvider`, `RetryConfig`, `S3ConnectionOptions`, `S3Credentials`, `StaticCredentialProvider`
- Errors: `ConfigValidationError`, `ManifestBuildError`, `ManifestParseError`, `ManifestStoreError`, `PoolExhaustedError`, `PublishCurrentError`, `PublishManifestError`, `ReaderStateError`, `S3TransientError`, `ShardAssignmentError`, `ShardCoverageError`, `ShardyfusionError`, `SlateDbApiError`
- Manifest/store: `InMemoryManifestStore`, `ManifestArtifact`, `ManifestBuilder`, `ManifestRef`, `ManifestShardingSpec`, `ManifestStore`, `RequiredBuildMeta`, `RequiredShardMeta`, `S3ManifestStore`, `YamlManifestBuilder`, `parse_manifest`
- Reader/routing: `AsyncManifestStore`, `AsyncS3ManifestStore`, `AsyncShardReader`, `AsyncShardReaderFactory`, `AsyncShardReaderHandle`, `AsyncShardedReader`, `AsyncSlateDbReaderFactory`, `ConcurrentShardedReader`, `ReaderHealth`, `ShardDetail`, `ShardReader`, `ShardReaderFactory`, `ShardReaderHandle`, `ShardedReader`, `SlateDbReaderFactory`, `SnapshotInfo`, `SnapshotRouter`
- Adapters: `DbAdapter`, `DbAdapterFactory`, `SlateDbFactory`
- Rate limiting: `AcquireResult`, `RateLimiter`, `ThreadSafeTokenBucket`, `TokenBucket`
- Logging/metrics: `FailureSeverity`, `JsonFormatter`, `LogContext`, `MetricEvent`, `MetricsCollector`, `configure_logging`, `get_logger`
- Sharding/serialization: `CelColumn`, `CelType`, `KeyEncoding`, `ShardingSpec`, `ShardingStrategy`, `ValueSpec`, `cel_sharding`, `cel_sharding_by_columns`

Writer functions are imported from subpackages (not re-exported at top level):
- **Spark:** `from shardyfusion.writer.spark import write_sharded, write_single_db, DataFrameCacheContext, SparkConfOverrideContext`
- **Dask:** `from shardyfusion.writer.dask import write_sharded, write_single_db, DaskCacheContext`
- **Ray:** `from shardyfusion.writer.ray import write_sharded, write_single_db, RayCacheContext`
- **Python:** `from shardyfusion.writer.python import write_sharded`

SQLite backend adapters are imported from their module (not re-exported at top level):
- `from shardyfusion.sqlite_adapter import SqliteFactory, SqliteReaderFactory, SqliteRangeReaderFactory`
- `from shardyfusion.sqlite_adapter import AsyncSqliteReaderFactory, AsyncSqliteRangeReaderFactory`

## Testing Notes

- **`tests/unit/`** — fast, no Spark; use pytest-xdist (`-n 2`). Areas: `shared/`, `read/`, `read_async/`, `writer/` (+ `writer/spark/`, `writer/python/`, `writer/dask/`, `writer/ray/`), `backend/sqlite/`, `metrics/`, `cli/`, `cel/`
- **`tests/unit/metrics/`** — `PrometheusCollector` and `OtelCollector` tests. Run via dedicated tox envs (`prometheus-unit`, `otel-unit`) that install the `metrics-prometheus`/`metrics-otel` extras. OTel tests also need `opentelemetry-sdk` (provided via `test` extra). These tests require their respective extras and will fail with `ImportError` without them — run via tox or install the extras explicitly.
- **Optional-dep test isolation** — Writer test directories (`tests/unit/writer/{dask,spark,ray}/`, `tests/integration/writer/{dask,ray}/`, `tests/e2e/writer/ray/`) use `conftest.py`-level `pytest.importorskip()` to skip entire suites when the framework extra is not installed. This allows `uv run pytest tests/` to work without all optional extras.
- **`tests/unit/backend/sqlite/`** — SQLite adapter unit tests (`test_sqlite_adapter.py`, `test_sqlite_range_reader.py`). Range-read tests require `apsw`.
- **`tests/integration/`** — S3 via `moto`; Spark writer tests require Spark + Java. `backend/sqlite/` tests SQLite adapter against moto S3.
- **`tests/e2e/`** — Garage S3 via compose; `just d-e2e`
- **`tests/helpers/`** — `s3_test_scenarios.py` (shared scenarios for moto/Garage), `smoke_scenarios.py` (shared smoke E2E data and scenarios for HASH/CEL sharding across all writer frameworks), `tracking.py` (test doubles: `TrackingAdapter`, `TrackingFactory`, `RecordingTokenBucket`, `InMemoryPublisher`)
- Markers: `@pytest.mark.spark`, `@pytest.mark.dask`, `@pytest.mark.ray`, `@pytest.mark.cel`, `@pytest.mark.e2e`
- **CEL tests**: require `cel` extra (`cel-expr-python`). Use `@pytest.mark.cel` marker. Skipped via `pytest.importorskip()` when the extra is not installed.
- **Contract tests**: hypothesis property tests in `test_routing_contract.py`, framework cross-checks in `writer/spark/`, `writer/dask/`, and `writer/ray/`
- **Cleanup tests**: `tests/unit/cli/test_cleanup.py` tests the `cleanup` CLI command; `tests/unit/writer/test_cleanup_core.py` tests core cleanup functions (`cleanup_stale_attempts`, `cleanup_old_runs`)
- For behavior changes: add/adjust unit tests first, then integration tests where routing/publishing or framework behavior is affected

## Coding Conventions

- Python 3.11+ with full type hints (pyright + ruff enforced). `E501` ignored (long lines allowed). See `pyproject.toml` for full ruff config
- Keep Spark logic in writer-side modules; avoid Python UDFs when Spark built-ins are available
- Dataclasses with `slots=True` for performance-sensitive models
- Manifest models use Pydantic (`ConfigDict(use_enum_values=False)`) — enum names serialize, not values
- Commit subjects: imperative, Conventional Commit prefixes recommended (`fix:`, `feat:`, `chore:`, `test:`)

## Gotchas & Non-obvious Behavior

- **SlateDB import is deferred**: `slatedb_adapter.py` imports the `slatedb` module inside `__init__`, not at module load. Tests monkeypatch `sys.modules["slatedb"]` to provide fakes.
- **APSW import is deferred**: `sqlite_adapter.py` imports `apsw` inside `SqliteRangeShardReader.__init__` and `_create_apsw_vfs()`. Only the range-read VFS tier requires `apsw`; the download-and-cache tier uses stdlib `sqlite3`.
- **SQLite adapter lifecycle**: `SqliteAdapter` builds a local SQLite DB during writes, then uploads the `.db` file to S3 on `close()`. The reader downloads (or range-reads) the file from S3. The `checkpoint()` method computes a SHA-256 hash of the DB file as the checkpoint ID.
- **SQLite range-read VFS uses per-instance VFS names**: Each `SqliteRangeShardReader` registers a unique APSW VFS (via `uuid4`) to avoid name collisions when multiple readers are open simultaneously. The VFS is unregistered on `close()`.
- **Python writer parallel mode uses `spawn`**: `multiprocessing.get_context("spawn")` is hardcoded. All objects passed to workers must be picklable. Min/max key tracking happens in the main process.
- **Rate limiter thread safety**: `TokenBucket` has no internal lock. Use `ThreadSafeTokenBucket` when sharing a limiter across threads (e.g. `ConcurrentShardedReader`). `ThreadSafeTokenBucket` locks around `try_acquire()` and loops with sleep outside the lock. Writers and readers depend on the `RateLimiter` Protocol, not the concrete class. Writers support both `max_writes_per_second` (ops) and `max_write_bytes_per_second` (bytes) via independent `TokenBucket` instances. Readers accept a `rate_limiter` parameter for reads/sec limiting.
- **Reader reference counting**: `refresh()` serialises I/O via `_refresh_lock`, atomically swaps `_ReaderState` with a compare-and-swap guard, and defers closing old handles until `refcount` drops to zero. Pool-mode checkout has a configurable timeout (default 30s) — raises `PoolExhaustedError` (retryable) when exhausted. Metadata methods (`key_encoding`, `snapshot_info`, `shard_details`, `route_key`) use `_use_state()` and raise `ReaderStateError` on a closed reader.
- **Borrow handle safety nets**: `ShardReaderHandle` and `AsyncShardReaderHandle` implement `__del__()` that logs a warning and releases the borrow if the handle was not explicitly closed. Prefer explicit `close()` or `with`/`async with` context managers.
- **Async reader uses `open()` classmethod**: `AsyncShardedReader.__init__` does no I/O. Use `await AsyncShardedReader.open(...)` and then `await reader.close()`, or `async with await AsyncShardedReader.open(...) as reader`. The default `AsyncS3ManifestStore` requires `aiobotocore` (`read-async`, or another install path that brings it in).
- **Async manifest store**: `AsyncShardedReader` accepts `AsyncManifestStore` only (not sync `ManifestStore`). When `manifest_store=None`, uses `AsyncS3ManifestStore` (native aiobotocore).
- **Reader cold-start fallback**: When `load_manifest()` raises `ManifestParseError` during initialization, readers walk backward through `list_manifests()` to find a valid manifest (up to `max_fallback_attempts`, default 3). Set to 0 to disable fallback. During `refresh()`, a malformed manifest is silently skipped and the reader keeps its current good state (returns `False`).
- **Garage requires path-style addressing**: E2E tests set `addressing_style: "path"` in `S3ClientConfig` because Garage doesn't support virtual-hosted-style.
- **Session-scoped test fixtures**: PySpark and S3 (moto/Garage) fixtures are session-scoped for performance. Tests share the same Spark session and S3 service.
- **Writer scenario imports are deferred**: `tests/helpers/s3_test_scenarios.py` imports writer modules inside function bodies so reader-only test collection doesn't fail.
- **Ray writer temporarily sets `DataContext.shuffle_strategy`**: During repartition, the Ray writer sets `DataContext.shuffle_strategy = "HASH_SHUFFLE"` and restores the previous value in a `try/finally` block, protected by `_SHUFFLE_STRATEGY_LOCK` (`threading.Lock`) to guard against concurrent `write_sharded()` calls in the same driver.
- **Ray tests need `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`**: Ray ≥ 2.47 auto-detects `uv run` in the process tree and overrides worker Python to create fresh venvs. The env var is set in `tox.ini` for raywriter envs. For direct pytest, either set the var or use `uv run --extra writer-ray pytest ...`.
- **`acquire_async()` is part of the `RateLimiter` Protocol**: Uses `asyncio.sleep` — safe for event loops. The sync `acquire()` uses `time.sleep` and must not be called from async code.
- **`try_acquire()` is pure arithmetic**: Guaranteed non-blocking (replenish + compare + deduct). Safe to call from async code directly without `to_thread()`.
- **`RetryConfig` is S3-specific**: Lives on `S3ManifestStore` (and `AsyncS3ManifestStore`), not on reader/writer constructors. Controls exponential backoff for transient S3 errors.
- **`cel-expr-python` does not support Python 3.14**: The `cel` extra depends on `cel-expr-python` which has no py3.14 wheels. CEL tests are skipped on py3.14 via `pytest.importorskip()`.

## Environment Notes

- Spark-based writer flows require **Java 17** (`JAVA_HOME` or `PATH`). Java 21 is not supported with Spark 3.5 (Arrow JVM module access issues). CI uses temurin distribution.
- Spark writer uses `mapInArrow` which requires `pandas>=2.2` and `pyarrow>=15.0` (both included in the `writer-spark` extra).
- `SPARK_LOCAL_IP=127.0.0.1` is set in tox and devcontainer to avoid hostname resolution issues.
- Reader-only, Python writer, Dask writer, and Ray writer usage do not require Spark/Java.
- **Ray does not run on py3.14** (same as Dask and CEL).
