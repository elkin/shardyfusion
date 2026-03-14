# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`shardyfusion` is a sharded snapshot writer/reader library for SlateDB. It provides:
- Writer-side Spark APIs to build `num_dbs` independent SlateDB shard databases
- Dask DataFrame-based writer (no Spark/Java needed)
- Ray Data-based writer (no Spark/Java needed)
- Pure-Python iterator-based writer (no Spark/Java needed)
- Manifest + `_CURRENT` publishing protocol (default S3, pluggable interfaces)
- Reader-side routing helpers for service-side `get` and `multi_get`
- `shardy` CLI for interactive and batch lookups against published snapshots
- Token-bucket rate limiter (`max_writes_per_second`) for all writer paths

## Tooling

**Package manager**: uv. Extras: `read`, `read-async` (adds aiobotocore for native async S3), `writer-spark` (requires Java), `writer-python`, `writer-dask`, `writer-ray`, `cli`. Full dev: `uv sync --all-extras --dev`.

**Workflows**: Run `just --list` for all local and container (`d-*`) targets. Key entry points: `just setup` (bootstrap a fresh clone), `just doctor` (verify environment), `just fix` (auto-format), `just ci` (quality + unit + integration), `just clean` / `just clean-all` (remove caches/artifacts). Container default engine is Podman; override with `CONTAINER_ENGINE=docker`.

**Test matrix**: tox labels — `quality`, `unit`, `integration`, `e2e`. Targeted pytest: `uv run pytest -q tests/unit/<area>` (areas: `cli`, `writer`, `read`, `shared`).

**Type checking**: `uv run pyright shardyfusion` (all code) or per-path configs in `pyright/` directory.

**JSON schemas**: Available on demand via `shardy schema` (manifest) and `shardy schema --type current-pointer`. Generated at runtime from Pydantic models (`ParsedManifest.model_json_schema()` / `CurrentPointer.model_json_schema()`).

Container venv is at `/opt/shardyfusion-venv`, not the host `.venv`.

## CI Pipeline (GitHub Actions)

On every push/PR: **quality → package → unit → integration** (parallel within each stage). E2E is local/container only (`just d-e2e`). Java 17 (temurin) for Spark jobs. **Dask and Ray do not run on py3.14.** Weekly scheduled build on Mondays at 06:00 UTC.

## Architecture

The library is split into six independent paths that share config, manifest models, and core logic:

**Writer path (Spark)** (requires PySpark + Java):
`writer/spark/writer.py` → `writer/spark/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`
`writer/spark/single_db_writer.py` (single-shard variant, distributed sorting)
`writer/spark/util.py` (`DataFrameCacheContext`)

**Writer path (Dask)** (no Spark/Java needed):
`writer/dask/writer.py` → `writer/dask/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`
`writer/dask/single_db_writer.py` (single-shard variant, distributed sorting)

**Writer path (Ray)** (no Spark/Java needed):
`writer/ray/writer.py` → `writer/ray/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`
`writer/ray/single_db_writer.py` (single-shard variant, distributed sorting)

**Writer path (Python)** (no Spark/Java needed):
`writer/python/writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`

**Reader path (sync)** (no Spark/Java needed):
`reader/reader.py` → `routing.py` → `manifest_store.py`, `db_manifest_store.py`

**Reader path (async)** (no Spark/Java needed; optional `aiobotocore` for native async S3):
`reader/async_reader.py` → `routing.py` → `async_manifest_store.py` (`AsyncS3ManifestStore`)

**CLI path** (requires click + pyyaml):
`cli/app.py` → `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`

### Module Dependency Graph

```
Layer 0 — Core types & errors: errors.py, type_defs.py (incl. Tracer/Span/RetryConfig), sharding_types.py, ordering.py, logging.py (incl. LogContext/JsonFormatter), metrics/ (package: _events.py, _protocol.py, _payloads.py)
Layer 1 — Config & serialization: config.py, serde.py, _rate_limiter.py, _circuit_breaker.py
Layer 2 — Storage, routing, manifest: storage.py (w/ RetryConfig + CircuitBreaker), manifest.py, routing.py, manifest_store.py, async_manifest_store.py, db_manifest_store.py
Layer 3 — Writer core: _writer_core.py (shared by all writers)
Layer 4 — Entry points: writer/{spark,dask,ray,python}/*.py, reader/reader.py (w/ ReaderHealth), reader/async_reader.py, cli/app.py
Layer 5 — Adapters & testing: slatedb_adapter.py, testing.py
Layer opt — Optional publishers: metrics/prometheus.py (metrics-prometheus extra), metrics/otel.py + metrics/tracing.py (metrics-otel extra)
```

### Write Pipeline

**Spark writer** (`write_sharded`):
1. Entry point in `writer/spark/writer.py`. Optionally applies Spark conf overrides via `SparkConfOverrideContext`.
2. `writer/spark/sharding.py` adds `_slatedb_db_id` column via Spark SQL expressions (hash, range, or custom), then converts the DataFrame to a pair RDD partitioned so partition index = db_id.
3. Each partition writes one shard to S3 at a temporary path (`_tmp/run_id=.../db=XXXXX/attempt=YY/`).
4. The driver collects results and selects deterministic winners (lowest attempt → task_attempt_id → URL).
5. A manifest artifact is built and published, then the `_CURRENT` pointer is updated.

**Dask writer** (`write_sharded`):
1. Entry point in `writer/dask/writer.py`. Accepts `dd.DataFrame` with `key_col`/`value_spec`.
2. `writer/dask/sharding.py` adds `_slatedb_db_id` column via Python routing function applied per partition (not Spark SQL). Range boundaries computed via Dask quantiles. `CUSTOM_EXPR` strategy is explicitly rejected.
3. Shuffles by `_slatedb_db_id`, then `map_partitions` writes each shard. Empty shards get zero-row placeholder results.
4. Optional rate limiting via `max_writes_per_second` (token-bucket). Routing verification via `verify_routing_agreement()`.
5. Uses the same `_writer_core.py` functions for winner selection, manifest building, and publishing.

**Ray writer** (`write_sharded`):
1. Entry point in `writer/ray/writer.py`. Accepts `ray.data.Dataset` with `key_col`/`value_spec`.
2. `writer/ray/sharding.py` adds `_slatedb_db_id` column via Arrow batch format (`map_batches` with `batch_format="pyarrow"`, `zero_copy_batch=True`). Range boundaries computed via sampling. `CUSTOM_EXPR` strategy is explicitly rejected.
3. Repartitions by `_slatedb_db_id` using hash shuffle (`DataContext.shuffle_strategy = "HASH_SHUFFLE"`; saved/restored). Then `map_batches` writes each shard with `batch_format="pandas"`.
4. Optional rate limiting via `max_writes_per_second` (token-bucket). Routing verification via `_verify_routing_agreement()`.
5. Uses the same `_writer_core.py` functions for winner selection, manifest building, and publishing.

**Python writer** (`write_sharded`):
1. Entry point in `writer/python/writer.py`. Accepts `Iterable[T]` with `key_fn`/`value_fn` callables.
2. Supports single-process (all adapters open simultaneously) or multi-process (`parallel=True`, one worker per shard via `multiprocessing.spawn`).
3. Optional rate limiting via `max_writes_per_second` (token-bucket in `_rate_limiter.py`).
4. Uses the same `_writer_core.py` functions for routing, winner selection, manifest building, and publishing.

### Read Pipeline

1. `ShardedReader` / `ConcurrentShardedReader` in `reader/reader.py` loads the `_CURRENT` pointer from S3 and dereferences the manifest.
2. Builds a `SnapshotRouter` from the manifest sharding metadata (mirrors write-time sharding logic).
3. `get(key)` / `multi_get(keys)` routes keys to shard IDs, then reads from the appropriate shard.
4. `shard_for_key(key)` / `shards_for_keys(keys)` return `RequiredShardMeta` for routing inspection without DB access. `reader_for_key(key)` / `readers_for_keys(keys)` return borrowed `ShardReaderHandle` handles for direct shard access.
5. `ShardedReader` swaps state directly on refresh. `ConcurrentShardedReader` serialises refresh I/O via `_refresh_lock` and atomically swaps readers using reference counting with a compare-and-swap guard for safe cleanup. Borrowed `ShardReaderHandle` handles hold refcount increments, preventing cleanup of old state while borrows are outstanding.
6. `ConcurrentShardedReader` provides thread safety via `threading.Lock` (default) or `ThreadPoolExecutor` pool mode (`thread_safety` config). Pool mode supports configurable checkout timeout (`pool_checkout_timeout`, default 30s).

### Async Read Pipeline

`AsyncShardedReader` in `reader/async_reader.py` mirrors the sync reader API using asyncio:

1. Constructed via `@classmethod async def open(...)` (since `__init__` can't do async I/O).
2. S3 manifest loading uses `AsyncS3ManifestStore` (native aiobotocore).
3. `async get(key)` / `async multi_get(keys)` route keys and read from async shard readers. `multi_get` fans out via `asyncio.TaskGroup` with optional `asyncio.Semaphore` for concurrency limiting (`max_concurrency`).
4. `async refresh()` serialises via `asyncio.Lock`. Same swap-and-retire pattern as sync, with deferred cleanup via `loop.create_task()` when retired state's borrow count hits 0.
5. Sync metadata methods (`key_encoding`, `snapshot_info()`, `shard_details()`, `route_key()`) and borrow methods (`reader_for_key()`, `readers_for_keys()`) work identically to the sync reader.
6. Supports `async with` context manager via `__aenter__`/`__aexit__`.

### CLI (`shardy`)

Entry point: `cli/app.py:main`. Subcommands: `get KEY`, `multiget KEY [KEY ...]`, `info`, `shards`, `route KEY`, `exec --script FILE`, `schema [--type manifest|current-pointer]`. No subcommand → interactive REPL (`cmd.Cmd` with `shardy> ` prompt; REPL-only commands include `refresh`).

Key coercion: CLI keys are strings; when manifest uses integer encoding (`u64be`/`u32be`), keys are auto-coerced to `int` via `cli/config.py:coerce_cli_key()`.

### Manifest & CURRENT Data Formats

**Manifest** (JSON, published to `s3_prefix/manifests/run_id={run_id}/{manifest_name}`):
```json
{"required": {/* RequiredBuildMeta: run_id, num_dbs, s3_prefix, sharding, key_encoding, ... */},
 "shards": [/* list of RequiredShardMeta: db_id, db_url, checkpoint_id, row_count, ... */],
 "custom": {/* user-defined fields from ManifestOptions.custom_manifest_fields */}}
```

**CURRENT pointer** (JSON, published to `s3_prefix/_CURRENT`):
```json
{"manifest_ref": "s3://bucket/.../manifest.json", "manifest_content_type": "application/json",
 "run_id": "...", "updated_at": "...", "format_version": 1}
```

Two-phase publish: manifest is written first, then `_CURRENT` pointer is updated. Both use `storage.put_bytes()` with exponential-backoff retries (3 attempts, 1s→2s→4s) for transient S3 errors.

### Key Abstractions

These are all Protocols, allowing user-provided implementations:

| Protocol | Default Implementation | Purpose |
|---|---|---|
| `ManifestBuilder` | `JsonManifestBuilder` | Manifest serialization format |
| `ManifestStore` | `S3ManifestStore` | Unified manifest read/write (publish + load) |
| `AsyncManifestStore` | `AsyncS3ManifestStore` | Async read-only manifest loading |
| `AsyncShardReader` | `_SlateDbAsyncShardReader` | Async shard reader (get/close) |
| `AsyncShardReaderFactory` | `AsyncSlateDbReaderFactory` | Async factory for opening shard readers |
| `DbAdapterFactory` | `SlateDbFactory` | How to open shard databases |

`ValueSpec` controls how DataFrame rows are serialized to bytes: `binary_col`, `json_cols`, or a callable encoder.

### Metrics & Observability

`MetricsCollector` (Protocol in `metrics.py`) is an optional observer: set `WriteConfig.metrics_collector` or pass it to the reader. The `MetricEvent` enum defines events across writer lifecycle (e.g., `WRITE_STARTED`, `SHARD_WRITE_COMPLETED`), reader lifecycle (`READER_GET`, `READER_REFRESHED`), and infrastructure (`S3_RETRY`, `RATE_LIMITER_THROTTLED`). Collectors receive `(event, **kwargs)` and should silently ignore unknown events for forward compatibility.

### Key Encodings

The `key_encoding` field on `WriteConfig` (default `"u64be"`) controls how keys are serialized to bytes in SlateDB:

- **`u64be`** (default): 8-byte big-endian unsigned integer. Keys in `[0, 2^64-1]`.
- **`u32be`**: 4-byte big-endian unsigned integer. Keys in `[0, 2^32-1]`. Cuts key storage in half.

Both encodings produce identical hash routing results for keys in `[0, 2^32-1]` because the routing layer zero-extends to 8-byte little-endian before hashing (matching Spark's `xxhash64(cast(key as long))`). The encoding is stored in the manifest and used by the reader/CLI for key coercion and lookup encoding.

### Sharding Strategies

- **Hash** (default): `pmod(xxhash64(cast(key as long)), num_dbs)` — requires integer key column
- **Range**: Explicit boundaries or computed via `approxQuantile` — supports int/float/string keys
- **Custom**: User-provided Spark SQL expression or column builder callable (Spark writer only; not supported in Dask, Ray, or Python writers)

The `SnapshotRouter` in `routing.py` mirrors the writer's sharding logic exactly for consistent key routing at read time.

## Critical Invariants

### Sharding Invariant (xxhash64)

**This is the most important correctness property.** The reader and writer MUST compute identical shard IDs for every key. A violation means reads silently go to the wrong shard.

The invariant: `pmod(xxhash64(payload, seed=42), num_dbs)` where:
- **Seed**: 42 (Spark's `XxHash64` default seed, hardcoded in `routing.py:_XXHASH64_SEED`)
- **Payload**: 8-byte little-endian representation of the key (matching JVM `Long.reverseBytes`)
- **Signed conversion**: digest is converted to signed int64 (matching JVM's signed long)
- **Modulo**: Python `%` with positive `num_dbs` equals Spark `pmod`

**Safety mechanisms:**
1. `verify_routing_agreement()` in `writer/spark/writer.py`, `writer/dask/writer.py`, and `writer/ray/writer.py` — runtime spot-check comparing framework-assigned shard IDs vs Python routing on a sample of written rows. Controlled by `verify_routing=True` (default).
2. `tests/unit/writer/test_routing_contract.py` — hypothesis property tests (Python-only). `tests/unit/writer/spark/test_routing_contract.py` — Spark-vs-Python cross-checks with ~200 edge-case keys. `tests/unit/writer/dask/test_dask_routing_contract.py` — Dask-vs-Python cross-checks. `tests/unit/writer/ray/test_ray_routing_contract.py` — Ray-vs-Python cross-checks.
3. Single implementation in `routing.py:xxhash64_db_id()` imported by all writer paths (never reimplemented).

### Error Hierarchy

All errors inherit from `ShardyfusionError` with `retryable: bool`. Non-retryable: config/programmer errors (`ConfigValidationError`, `ShardAssignmentError`, `ManifestParseError`, `ReaderStateError`, `SlateDbApiError`, etc.). Retryable: transient infra (`PublishManifestError`, `PublishCurrentError`, `S3TransientError`). See `errors.py` for full catalog.

## Public API Summary

Core types exported from `shardyfusion.__init__` (always available, no optional extras required). See `__init__.py` for the full list. Reader types include `ShardedReader`, `ConcurrentShardedReader`, `ShardReaderHandle`, `ShardDetail`, `SnapshotInfo`, `SlateDbReaderFactory`. Async reader types include `AsyncShardedReader`, `AsyncShardReaderHandle`, `AsyncSlateDbReaderFactory`, `AsyncShardReader`, `AsyncShardReaderFactory`, `AsyncManifestStore`, `AsyncS3ManifestStore`.

Writer functions are imported from subpackages (not re-exported at top level):
- **Spark:** `from shardyfusion.writer.spark import write_sharded, write_single_db, DataFrameCacheContext, SparkConfOverrideContext`
- **Dask:** `from shardyfusion.writer.dask import write_sharded, write_single_db, DaskCacheContext`
- **Ray:** `from shardyfusion.writer.ray import write_sharded, write_single_db, RayCacheContext`
- **Python:** `from shardyfusion.writer.python import write_sharded`

## Testing Notes

- **`tests/unit/`** — fast, no Spark; use pytest-xdist (`-n 2`). Areas: `shared/`, `read/`, `writer/` (+ `writer/spark/`, `writer/python/`, `writer/dask/`, `writer/ray/`), `cli/`
- **`tests/integration/`** — S3 via `moto`; Spark writer tests require Spark + Java
- **`tests/e2e/`** — Garage S3 via compose; `just d-e2e`
- **`tests/helpers/`** — `s3_test_scenarios.py` (shared scenarios for moto/Garage), `tracking.py` (test doubles: `TrackingAdapter`, `TrackingFactory`, `RecordingTokenBucket`, `InMemoryPublisher`)
- Markers: `@pytest.mark.spark`, `@pytest.mark.dask`, `@pytest.mark.ray`, `@pytest.mark.e2e`
- **Contract tests**: hypothesis property tests in `test_routing_contract.py`, framework cross-checks in `writer/spark/`, `writer/dask/`, and `writer/ray/`
- For behavior changes: add/adjust unit tests first, then integration tests where routing/publishing or framework behavior is affected

## Coding Conventions

- Python 3.11+ with full type hints (pyright + ruff enforced). `E501` ignored (long lines allowed). See `pyproject.toml` for full ruff config
- Keep Spark logic in writer-side modules; avoid Python UDFs when Spark built-ins are available
- Dataclasses with `slots=True` for performance-sensitive models
- Manifest models use Pydantic (`ConfigDict(use_enum_values=False)`) — enum names serialize, not values
- Commit subjects: imperative, Conventional Commit prefixes recommended (`fix:`, `feat:`, `chore:`, `test:`)

## Gotchas & Non-obvious Behavior

- **SlateDB import is deferred**: `slatedb_adapter.py` imports the `slatedb` module inside `__init__`, not at module load. Tests monkeypatch `sys.modules["slatedb"]` to provide fakes.
- **Python writer parallel mode uses `spawn`**: `multiprocessing.get_context("spawn")` is hardcoded. All objects passed to workers must be picklable. Min/max key tracking happens in the main process.
- **Rate limiter is not thread-safe**: `TokenBucket` has no internal lock; a `ThreadSafeTokenBucket` subclass will be added when needed. Writers and readers depend on the `RateLimiter` Protocol, not the concrete class. Writers support both `max_writes_per_second` (ops) and `max_write_bytes_per_second` (bytes) via independent `TokenBucket` instances. Readers accept a `rate_limiter` parameter for reads/sec limiting.
- **Reader reference counting**: `refresh()` serialises I/O via `_refresh_lock`, atomically swaps `_ReaderState` with a compare-and-swap guard, and defers closing old handles until `refcount` drops to zero. Pool-mode checkout has a configurable timeout (default 30s) — raises `SlateDbApiError` when exhausted. Metadata methods (`key_encoding`, `snapshot_info`, `shard_details`, `route_key`) use `_use_state()` and raise `ReaderStateError` on a closed reader.
- **Async reader uses `open()` classmethod**: `AsyncShardedReader.__init__` does no I/O. Use `await AsyncShardedReader.open(...)` or `async with AsyncShardedReader(...)`. Requires `aiobotocore` (the `read-async` extra).
- **Async manifest store**: `AsyncShardedReader` accepts `AsyncManifestStore` only (not sync `ManifestStore`). When `manifest_store=None`, uses `AsyncS3ManifestStore` (native aiobotocore). No sync-to-async adapter — if you want async, install `aiobotocore`.
- **Garage requires path-style addressing**: E2E tests set `addressing_style: "path"` in `S3ClientConfig` because Garage doesn't support virtual-hosted-style.
- **Session-scoped test fixtures**: PySpark and S3 (moto/Garage) fixtures are session-scoped for performance. Tests share the same Spark session and S3 service.
- **Writer scenario imports are deferred**: `tests/helpers/s3_test_scenarios.py` imports writer modules inside function bodies so reader-only test collection doesn't fail.
- **Dask writer rejects `CUSTOM_EXPR` sharding**: Unlike the Spark writer, the Dask writer raises `ConfigValidationError` for custom expression sharding. Only `HASH` and `RANGE` are supported.
- **Ray writer rejects `CUSTOM_EXPR` sharding**: Same as Dask — only `HASH` and `RANGE` are supported.
- **Ray writer temporarily sets `DataContext.shuffle_strategy`**: During repartition, the Ray writer sets `DataContext.shuffle_strategy = "HASH_SHUFFLE"` and restores the previous value in a `try/finally` block.
- **Ray tests need `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`**: Ray ≥ 2.47 auto-detects `uv run` in the process tree and overrides worker Python to create fresh venvs. The env var is set in `tox.ini` for raywriter envs. For direct pytest, either set the var or use `uv run --extra writer-ray pytest ...`.

## Environment Notes

- Spark-based writer flows require **Java 17** (`JAVA_HOME` or `PATH`). CI uses temurin distribution.
- `SPARK_LOCAL_IP=127.0.0.1` is set in tox and devcontainer to avoid hostname resolution issues.
- Reader-only, Python writer, Dask writer, and Ray writer usage do not require Spark/Java.
- **Ray does not run on py3.14** (same as Dask).
