# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Changed (breaking)

#### SlateDB upgrade — 0.7 → 0.12

- **Bumped `slatedb` requirement to `>=0.12,<0.13`** (was `>=0.7,<0.8`). The
  legacy top-level `slatedb.SlateDB` / `slatedb.SlateDBReader` API was removed
  upstream in 0.12; all runtime imports now go through `slatedb.uniffi`
  (`Db`, `DbBuilder`, `DbReader`, `DbReaderBuilder`, `WriteBatch`, `Settings`,
  `ObjectStore`, `FlushOptions`, `FlushType`).
- **All slatedb operations are async upstream**. shardyfusion exposes the
  same sync writer/reader Protocols as before by funneling every hop
  through a process-global daemon-thread `asyncio` loop in
  `shardyfusion._slatedb_runtime` (`run_coro(...)`). The bridge is owned by
  shardyfusion and is not user-customizable.
- **Adapter `checkpoint() -> str | None` renamed to `seal() -> None`**
  (breaking for custom `DbAdapter` implementations). Writers now do
  `adapter.flush(); adapter.seal(); checkpoint_id = generate_checkpoint_id()`.
  The `checkpoint_id` field on `RequiredShardMeta` is now an opaque shardyfusion-
  generated `uuid.uuid4().hex` (32 lowercase hex chars), produced centrally by
  `shardyfusion._checkpoint_id.generate_checkpoint_id()`. **Migration**:
  rename your `def checkpoint(self) -> str | None` to `def seal(self) -> None`
  and drop the return value.
- **SQLite / SQLiteVec checkpoint-id semantics changed**: the SHA-256 digest
  of the materialized DB file is no longer the `checkpoint_id`; a UUID is
  used instead. Reader cache identity (per-shard local-cache directory) still
  keys on `checkpoint_id`, so cache invalidation behaviour is unchanged. If
  you persisted SHA-256 fingerprints externally, switch to comparing
  `db_bytes()` or computing your own digest from the downloaded file.
- **`SlateDbReaderFactory` no longer forwards `checkpoint_id` to slatedb**
  (slatedb 0.12 has no equivalent of the legacy `create_checkpoint(scope=...)`
  API; user invariant is single-writer-per-DB with publish-after-finish, plus
  S3 strong consistency, which together remove the need for checkpoint pinning
  on the read path). The `checkpoint_id` argument is still accepted by the
  factory for Protocol symmetry but is ignored. `SQLite*` and `LanceDB*`
  factories continue to use `checkpoint_id` for local-cache identity.
- **New `SlateDbReaderFactory(iterator_chunk_size=1024)` parameter** controls
  the batch size for the sync→async iterator bridge in `scan_iter()`. A single
  bridge hop costs ~15–40 µs; per-row hops would dominate scans by ~30×, so
  chunking is mandatory. Tune downward (e.g. 256) for low-latency partial
  scans, upward (e.g. 4096) for bulk dumps. Lives only on the reader factory.
- **New typed `shardyfusion.SlateDbSettings` dataclass** for slatedb settings
  configuration. Includes a `raw_overrides: dict[str, str]` escape hatch for
  keys not yet promoted to typed fields. The pre-0.12 free-form JSON-dict
  shape is **not** accepted — `SlateDbFactory.settings` is typed
  `SlateDbSettings | None`.
- **`DbAdapter` and `DbAdapterFactory` Protocols moved to
  `shardyfusion._adapter`** (backend-neutral). They were re-exported from
  `shardyfusion.slatedb_adapter` historically but were never SlateDB-specific.
  Public re-exports (`shardyfusion.DbAdapter`, `shardyfusion.DbAdapterFactory`)
  are unchanged. **Migration**: if you imported either Protocol from
  `shardyfusion.slatedb_adapter` directly, switch to `shardyfusion._adapter`
  (or, preferably, the top-level `shardyfusion` namespace).
- **`shardyfusion.credentials.apply_env_file()`** is the new in-house dotenv
  loader (context manager); we no longer rely on `slatedb`'s legacy
  `env_file=` ctor parameter. `SlateDbReaderFactory(env_file=...)` and the
  writer adapter both call it before resolving `ObjectStore`.

#### Test fakes & helpers
- **`shardyfusion.testing.open_slatedb_db(db_url)`** /
  **`open_slatedb_reader(db_url)`** — convenience helpers that return raw
  uniffi `Db` / `DbReader` handles for tests that need to assert directly
  against slatedb state. Caller owns lifecycle (`run_coro(db.shutdown())`).

### Added
- **Performance microbenchmarks** under `tests/integration/perf/`
  (`@pytest.mark.perf`, opt-in only, excluded from default `pytest tests`
  via `addopts = "-ra -m 'not perf'"`). Run with `just perf`. Currently
  covers WriteBatch throughput, point-get round-trip, and chunked-scan
  per-row cost; budgets are deliberately loose to catch order-of-magnitude
  regressions only.
- **`just perf [path]`** recipe — runs `pytest -m perf` against
  `tests/integration/perf/` (or a narrower path) without tox parallelism so
  timings stay stable. Not part of `just ci`.

#### Async bridge (sync→async) public surface
- **`shardyfusion._async_bridge`** with `call_sync(coro, *, timeout=None)` and
  `gather_sync(coros, *, timeout=None, return_exceptions=False)`. Both submit
  to the shardyfusion-owned bridge loop, raise `BridgeTimeoutError`
  (`shardyfusion.errors.BridgeTimeoutError`, `retryable=True`) on timeout,
  cancel the in-flight coroutine(s) on `KeyboardInterrupt`, and refuse re-entry
  from the bridge loop itself (raises `RuntimeError`). The legacy `run_coro`
  remains as a thin alias for back-compat.
- **`gather_sync` fan-out**: pays the sync→async hop cost (~16-18 µs) once for
  N coroutines instead of N times. Wired into `SlateDbReaderFactory.open_many`
  so opening a snapshot with K non-empty shards costs one bridge round-trip
  rather than K. The reader (`shardyfusion.reader.reader`) duck-types this on
  the factory and falls back to per-shard `_open_one_reader` when absent, so
  custom factories without `open_many` keep working.
- **`SlateDbBridgeTimeouts`** dataclass on `shardyfusion.slatedb_adapter` with
  `open_s` / `write_s` / `flush_s` / `shutdown_s` (all default `None`,
  unbounded — zero behavioral change). Pass via
  `SlateDbFactory(..., bridge_timeouts=SlateDbBridgeTimeouts(write_s=30.0))`
  to opt into bounded ops. **`BridgeTimeoutError` is preserved across the
  open path** — it is no longer wrapped in `DbAdapterError`, so the shard
  retry path correctly observes `exc.retryable=True`.

#### Reader/bridge lifecycle correctness
- **`_SlateDbReaderHandle.close()` and `_SlateDbAsyncShardReader.close()`
  now invoke uniffi `DbReader.shutdown()`** to release Tokio/object-store
  resources eagerly. Previously the handle only flipped `_closed` and the
  inner reader survived until Python finalization, leaking resources in
  long-running services that periodically refresh snapshots.
- **`gather_sync(..., return_exceptions=False)` cancels still-pending
  siblings on the first failure** and best-effort closes any already-
  successful results (`shutdown`/`aclose`/`close`). For
  `SlateDbReaderFactory.open_many` this means a partial-failure open no
  longer leaks orphaned background reader builds or fully-built
  `DbReader`s. `return_exceptions=True` mode still returns all results
  unchanged (caller owns lifecycle).

#### SQLite / SQLite-vec download cache concurrency
- **`SqliteShardReader` and the SQLite-vec download-cache reader now use
  process-safe atomic writes plus a per-`local_dir` `fcntl` lock**
  (`shardyfusion._local_snapshot_cache.ensure_cached_snapshot`). Previous
  behaviour was a TOCTOU identity check followed by non-atomic
  `Path.write_bytes`, which under concurrent CLI / test workloads
  (`tox -p N`, multiple CLI processes sharing `/tmp/shardyfusion`) caused
  torn reads (returning bytes from a stale snapshot) and intermittent
  `SIGBUS` from mmap on a half-truncated file. The new helper:
  re-checks identity inside the lock so peers skip duplicate downloads,
  writes the database to `shard.db.tmp.<pid>` + `os.fsync` + `os.replace`,
  then writes the identity file via the same temp+replace dance. Lance
  readers were unaffected (they connect directly to S3).

### Added

#### SQLite B-tree metadata sidecar (writer-side)
- **`shard.btreemeta` sidecar**: each finalized SQLite shard now publishes a small sibling artifact alongside `shard.db` containing every interior B-tree page plus every page belonging to the schema btree. A range-mode reader can fetch the sidecar once on shard open and pin those pages for the lifetime of the shard reader, collapsing the typical ~3 round trips per point lookup down to ~1. The sidecar is ~0.5–2% of the main DB for typical shards.
- **`SqliteFactory.emit_btree_metadata`** and **`SqliteVecFactory.emit_btree_metadata`**: bool field, default `True`. Opt out via `emit_btree_metadata=False` for write-cost regimes that never use the range-read VFS.
- **Manifest hook**: writers automatically record `manifest.custom["sqlite_btreemeta"]` (`{format_version, page_size, filename}`) when the toggle is on, so reader-side discovery is one snapshot-level lookup with no per-shard probing.
- **Page identification via `dbstat`**: extraction uses SQLite's own `dbstat` virtual table (via APSW) — no hand-rolled file-format parsing — and reads page bytes by direct file I/O at deterministic offsets.
- **zstd-compressed, indexed body** (format v3): the sidecar's body is zstd-compressed via the `zstandard` package (now in the `[sqlite-range]` extra) and carries a `(pageno, offset)` index sorted by pageno. The index lets a reader either pin the full body in memory or decompress to disk and `pread` individual pages with only an in-memory `pageno → offset` map (~8 bytes/page) — useful for memory-constrained or very-many-shard readers. Btree pages compress ~12× — a 100 k-row shard's sidecar drops from ~262 KB uncompressed to ~21 KB on the wire (~0.2% of the source DB). Magic and version stay uncompressed so readers validate before decompressing.
- **Graceful degradation**: best-effort emission. Missing APSW, missing `zstandard`, missing `dbstat`, an unsafe journal mode (e.g. WAL), or any extraction/upload failure logs at debug/transient severity and skips the sidecar; the main `shard.db` PUT proceeds unchanged. See [`docs/architecture/sqlite-btree-sidecar.md`](docs/architecture/sqlite-btree-sidecar.md).
- **Journal-mode guard**: extraction refuses to run on `journal_mode=wal|wal2` files. WAL can park committed pages in a `-wal` sidecar that the extractor's direct-file-I/O path would miss; rather than emit silently-incomplete sidecars if a future writer change enables WAL, the guard fails fast with `reason=unsupported_journal_mode`.
- **`shardyfusion.sqlite_adapter.extract_btree_metadata`** and **`BtreeMetaUnavailableError`**: new public symbols for users that want to extract sidecars out-of-band.

#### Performance & Public API
- **`shardyfusion.routing.make_hash_router(num_dbs, algorithm, *, key_type=None)`**: factory that returns a `(key) -> db_id` closure with hash-algorithm and (optionally) key-type dispatch resolved once. The optional `key_type` (`"int"`, `"str"`, `"bytes"`) selects a type-specialised digest path that skips per-call `isinstance` branching in `canonical_bytes`. Wired into all writer paths (Spark/Dask/Ray/Python single + parallel) and the reader's `SnapshotRouter._build_route_one`.
- **`shardyfusion.cel.compile_cel_cached`** is now a public name (previously `_compile_cel_cached`); the LRU cache size has been bumped from 16 to 64. The old private name remains as a back-compat alias.
- **`CompiledCel.columns_list`** property: cached column-name list, used by `evaluate_cel_arrow_batch` to avoid rebuilding the list per batch.

### Changed
- **Hash routing hot path**: writers and the reader now build a single resolved closure via `make_hash_router(...)` instead of dispatching on the algorithm enum and key type per call. The `xxh3_64` digest is bound once per partition/batch.
- **`canonical_bytes(key)`**: uses `type(key) is bytes` fast-path (no copy) and avoids redundant isinstance checks.
- **CEL compile sites** in per-batch / per-partition writer closures now go through `compile_cel_cached` instead of raw `compile_cel`, ensuring at most one compile per `(expr, columns)` per worker process.
- **Python writer CEL routing** (`_write_single_process_cel` and the three `_parallel_writer.py` CEL paths): hoisted CEL compile + categorical-routing-values lookup out of the per-row loop. Per record now does only `columns_fn(record)` (when present) + a direct CEL evaluate; no per-row tuple sort or LRU lookup.
- **`SnapshotRouter` (CEL)**: eager-compiles the CEL expression in `__init__` and reuses the compiled instance from both `_build_route_one` and `route_with_context`, removing the per-call `if compiled is None` check and lazy-import.
- **`route_cel_batch`**: builds the categorical routing-values lookup once per batch instead of per row.

#### Writers
- **Spark writer** (`writer-spark`): PySpark DataFrame-based sharded writer with hash, range, and custom expression sharding strategies.
- **Dask writer** (`writer-dask`): Dask DataFrame-based sharded writer (hash and range sharding, no Java required).
- **Ray writer** (`writer-ray`): Ray Data Dataset-based sharded writer (hash and range sharding, no Java required).
- **Python writer** (`writer-python`): Pure-Python iterator-based writer with single-process and multi-process (`parallel=True`) modes.
- **Token-bucket rate limiter** (`max_writes_per_second`) for all writer paths.
- **Routing verification**: Runtime spot-check (`verify_routing=True`) comparing framework-assigned shard IDs against Python routing on a sample of written rows (Spark, Dask, and Ray writers).
- **Two-phase publish protocol**: Manifest written first, then `_CURRENT` pointer updated, with exponential-backoff retries for transient S3 errors.

#### Reader
- **`ShardedReader`** (non-thread-safe) and **`ConcurrentShardedReader`** (thread-safe with lock/pool modes) for service-side key lookups.
- **`ShardReaderHandle`** and **`ShardDetail`** types for direct shard-level access.
- **Shard-level APIs**: `shard_for_key()`, `shards_for_keys()`, `reader_for_key()`, `readers_for_keys()` for routing inspection and raw handle borrowing.
- **Metadata APIs**: `snapshot_info()` and `shard_details()` for inspecting the current snapshot without performing database reads.
- **Pool checkout timeout**: `pool_checkout_timeout` parameter (default 30s) for pool mode — raises `SlateDbApiError` when exhausted.
- **Reader parameter validation** at construction time (`max_workers`, `pool_checkout_timeout`).
- **Lock ordering validation** for `ConcurrentShardedReader` internal locks (debug builds only).

#### CLI
- **`shardy`** command-line tool with `get`, `multiget`, `info`, `shards`, `route`, `refresh`, and `exec` subcommands.
- **Interactive REPL** mode (no subcommand → `slate>` prompt).
- **stdin support** for `multiget -` (pipe keys from stdin).
- **`--version` flag**.

#### Infrastructure
- **`ManifestStore`** protocol — unified manifest read/write interface.
- **`DbAdapterFactory`** protocol for custom shard database backends.
- **Key encodings**: `u64be` (default, 8-byte) and `u32be` (4-byte) big-endian unsigned integer encodings.
- **JSON schema validation** for manifest and CURRENT pointer formats.
- **Metrics/observability**: `MetricsCollector` protocol with `MetricEvent` catalog for writer and reader lifecycle events.
- **SQLite shard access mode (`sqlite_mode`)** — new `ReaderConfig` field with three values: `"download"` (full-file fetch), `"range"` (S3 range-read VFS via `apsw` + `obstore`), and `"auto"` (default; per-snapshot decision based on shard size distribution).
- **`AdaptiveSqliteReaderFactory`** / **`AsyncAdaptiveSqliteReaderFactory`** — composing factories that resolve to a download or range sub-factory once per snapshot using `decide_access_mode()`. Cached by `manifest.required_build.run_id`; rebuilt on `refresh()` when the run_id rotates.
- **`AdaptiveSqliteVecReaderFactory`** / **`AsyncAdaptiveSqliteVecReaderFactory`** — same adaptive policy applied to unified KV+vector sqlite-vec snapshots.
- **`decide_access_mode()`** policy function in `shardyfusion.sqlite_adapter` — OR-semantics over `per_shard_threshold` (default 16 MiB) and `total_budget` (default 2 GiB).
- **`SqliteAccessPolicy`** Protocol + **`make_threshold_policy()`** helper for advanced users who want to override the built-in heuristic.
- **CLI flags**: `--sqlite-mode {download,range,auto}`, `--sqlite-auto-per-shard-bytes BYTES`, `--sqlite-auto-total-bytes BYTES` on the root `shardy` group. CLI flags override TOML; TOML supports the same keys (`sqlite_mode`, `sqlite_auto_per_shard_threshold_bytes`, `sqlite_auto_total_budget_bytes`) under `[reader]`.
- **`sqlite-adaptive`** / **`read-sqlite-adaptive`** extras: composes `sqlite` + `sqlite-range` so `AdaptiveSqliteReaderFactory` (the default reader mode) can resolve to either tier. **`sqlite-adaptive-async`** is the async counterpart (adds `aiobotocore`).
- **`cli-minimal`** extra: bare CLI binary (`click>=8.0` only). Combine with any backend extra (e.g. `pip install 'shardyfusion[cli-minimal,read-sqlite]'`) for a slim install. The kitchen-sink `cli` extra still bundles every read backend.

### Changed
- **Refactored writer public APIs to grouped configs, inputs, and options** (breaking). KV writers now use `write_hash_sharded(data, config, input, options=None)` and `write_cel_sharded(data, config, input, options=None)` across Python, Spark, Dask, and Ray. Single-database writers use `write_single_db(data, config, input, options=None)`. Python writers take `PythonRecordInput`; Spark/Dask/Ray KV writers take `ColumnWriteInput`; per-call execution knobs live in `PythonWriteOptions`, `SparkWriteOptions`, `DaskWriteOptions`, `RayWriteOptions`, or `SingleDbWriteOptions`.
- **Renamed vector writer entry points to `write_sharded` and removed vector writer scalar keyword arguments** (breaking). Standalone vector writes use `shardyfusion.vector.write_sharded(records, config, input=None, options=None)`. Distributed vector writes use `write_sharded(data, config, input, options=None)` from `shardyfusion.writer.spark.vector_writer`, `shardyfusion.writer.dask.vector_writer`, or `shardyfusion.writer.ray.vector_writer`; the framework packages also export `write_sharded` and a compatibility `write_vector_sharded` alias for that refactored function. Column mapping now lives in `VectorColumnInput`.
- **Split writer configuration into nested public config groups** (breaking). KV configs now group storage, output, manifest, KV adapter settings, retry, rate limits, observability, lifecycle, and optional vector settings via `WriterStorageConfig`, `WriterOutputConfig`, `WriterManifestConfig`, `KeyValueWriteConfig`, `WriterRetryConfig`, `KvWriteRateLimitConfig`, `WriterObservabilityConfig`, and `WriterLifecycleConfig`. Hash and CEL sharding settings live in `HashShardingConfig` and `CelShardingConfig`.
- **Separated KV and vector write rate-limit config**. KV writers use `KvWriteRateLimitConfig(max_writes_per_second, max_write_bytes_per_second)`. Vector writers use `VectorWriteRateLimitConfig(max_writes_per_second)`. `max_writes_per_second` is no longer a separate public parameter on vector writer functions.
- **Centralized config validation dispatch** via the `ValidatableConfig` Protocol and `validate_configs(...)` helper. Writer entry points now validate config, input, and options through one ordered validation call.
- **Shared distributed partition write runtime across Spark, Dask, and Ray**. Internal partition-write runtime state now uses common `PartitionWriteRuntime.from_public_config(...)` construction for fields used by every distributed engine, with `RetryingPartitionWriteRuntime` adding the Dask/Ray-only retry and per-partition sort settings.
- **Replaced boto3/aiobotocore with obstore as the unified S3 I/O layer** (`obstore>=0.4` is now a core dependency). All S3 I/O (sync and async) is delegated to obstore's Rust-backed `S3Store`. Retries are handled by the Rust layer with shardyfusion-tuned aggressive defaults (`max_retries=5`, `init_backoff=200ms`, `max_backoff=4s`, `retry_timeout=30s`). A single Python-level fallback retry (2s delay) is applied for PUT operations on `GenericError` only.
- **New `StorageBackend` and `AsyncStorageBackend` Protocols** in `shardyfusion.storage` — `ObstoreBackend`/`AsyncObstoreBackend` (default), `MemoryBackend`/`AsyncMemoryBackend` (test/development). `S3ManifestStore`, `S3RunRegistry`, `AsyncS3ManifestStore`, and all cleanup functions now accept a backend instance via constructor or parameter injection instead of constructing their own boto3/aiobotocore clients.
- **LanceDB factory signatures changed** — `LanceDbWriterFactory` and `LanceDbReaderFactory` no longer accept an `s3_client` parameter. They now accept `credential_provider` + `s3_connection_options` and build `storage_options` directly from config. This is a **breaking change** for anyone constructing LanceDB factories manually.
- **User-supplied `CredentialProvider` implementations must be picklable** — `WriterStorageConfig.credential_provider` on KV and vector writer configs is serialized by Spark/Dask/Ray workers and by the Python writer's multiprocessing spawn mode. Custom providers must implement `__getstate__`/`__setstate__` (or otherwise be compatible with `pickle.dumps`) so that worker processes can reconstruct the credential source. The built-in `EnvCredentialProvider` and `StaticCredentialProvider` are already picklable.
- **`AsyncShardedReader` default manifest store** is now `AsyncS3ManifestStore` backed by `AsyncObstoreBackend` (native async obstore). The `aiobotocore` dependency has been removed entirely.
- **Manifest `format_version` bumped to `4`** (was `2`/`3`). Readers strictly accept only `{4}`; older manifests are rejected with `ManifestParseError`. **No backward compatibility** — re-publish snapshots after upgrading.
- **Default S3 run-registry credential resolution now uses only writer-level S3 settings** (`config.storage.credential_provider` and `config.storage.s3_connection_options`). Manifest-level S3 settings (`WriterManifestConfig.credential_provider` / `WriterManifestConfig.s3_connection_options`) remain scoped to the default S3 manifest store and are no longer consulted for run-registry S3 access. **Migration**: if you previously configured S3 credentials only via `WriterManifestConfig`, copy `credential_provider` and `s3_connection_options` to `WriterStorageConfig`, or pass an explicit `run_registry=` instance. Otherwise the run registry will be silently disabled (the writer logs a failure and continues without a run record).
- **Removed `managed_run_record()`** — use `RunRecordLifecycle.start(...)` directly as the run-record context manager.
- **`RequiredShardMeta.db_bytes` is now mandatory** (`int`, non-optional). Synthetic empty shards (padded by `SnapshotRouter`) report `db_bytes=0`. Postgres-backed manifest stores must run `ALTER TABLE shardyfusion_manifests_shards ADD COLUMN db_bytes BIGINT NOT NULL DEFAULT 0` before upgrade.
- **Reader-factory protocols (`ShardReaderFactory`, `AsyncShardReaderFactory`, vector counterparts) now require a `manifest` keyword argument** on `__call__`. Custom factory implementations must accept `manifest: ParsedManifest`. Built-in factories (`SqliteReaderFactory`, `SqliteRangeReaderFactory`, `SlateDbReaderFactory`, `LanceDbReaderFactory`, `SqliteVecReaderFactory`, and async variants) all accept and forward this parameter.
- **`cli` extra now aggregates all read backends** (`read-slatedb`, `read-slatedb-async`, `read-sqlite`, `read-sqlite-range`, `sqlite-async`, `unified-slatedb-lancedb`, `unified-sqlite-vec`) so any published snapshot can be opened without additional installs. Users who previously relied on a slim `cli` install must now opt in via narrower extras (`shardyfusion[read-slatedb]` etc.) directly.
- **Renamed extras to name the storage backend explicitly** (breaking; pre-1.0). Every extra now advertises its backend in its name, matching the existing `-sqlite` convention. Old → new:
  - `read` → `read-slatedb`
  - `read-async` → `read-slatedb-async`
  - `writer-spark` → `writer-spark-slatedb`
  - `writer-python` → `writer-python-slatedb`
  - `writer-dask` → `writer-dask-slatedb`
  - `writer-ray` → `writer-ray-slatedb`
  - `unified-vector` → `unified-slatedb-lancedb`
  - `unified-vector-sqlite` → `unified-sqlite-vec`

### Removed
- **`vector` extra alias** — removed in favor of explicit backend names. Use `vector-lancedb` (LanceDB) or `vector-sqlite` (sqlite-vec) directly.

### Removed
- **`boto3` and `aiobotocore` dependencies** — completely removed from all extras and dependency groups. obstore is now the sole S3 client. No direct AWS SDK installation is required.
- **`create_s3_client()` helper** — removed from `shardyfusion.vector.writer` and all other locations. Use `shardyfusion.storage.create_s3_store()` to build an `obstore.store.S3Store` directly.
- **`get_bytes()` / `put_bytes()` / `list_prefixes()` free functions** — removed from top-level modules. All S3 I/O now goes through `StorageBackend`/`AsyncStorageBackend` Protocol methods (`get`, `put`, `delete`, `list`).
- **`ManifestBuilder`** protocol and **`YamlManifestBuilder`** class — `SqliteManifestBuilder` is now the sole manifest format, used internally by `S3ManifestStore`.
- **`parse_manifest()`** function — replaced by `parse_manifest_payload()` which only accepts SQLite manifests.
- **`WriterManifestConfig.manifest_builder`** parameter — manifest format is no longer pluggable via config.
- **`shardy schema`** CLI subcommand and its REPL counterpart — JSON Schemas remain accessible programmatically via `ParsedManifest.model_json_schema()` and `CurrentPointer.model_json_schema()`.

### Changed (breaking)
- **Replaced the old KV `WriteConfig` API with concrete grouped configs** — use `HashShardedWriteConfig` (with `HashShardingConfig`) for hash sharding, `CelShardedWriteConfig` (with `CelShardingConfig`) for CEL sharding, and `SingleDbWriteConfig` for single-database writes. `BaseShardedWriteConfig` is shared infrastructure and should not be instantiated directly.
- **Split `ShardingSpec` into `HashShardingSpec` and `CelShardingSpec` for manifest/internal use** — public writer sharding settings now live in `HashShardingConfig` and `CelShardingConfig`.
- **Removed unified KV `write_sharded` entry point** — all KV writer frameworks (Python, Spark, Dask, Ray) now expose per-strategy functions: `write_hash_sharded()` and `write_cel_sharded()`. Vector writers still expose `write_sharded()` from vector-specific modules.
- **`cel_sharding()` and `cel_sharding_by_columns()`** now return `CelShardingSpec` instead of the old unified `ShardingSpec`.
- **Removed `_LegacyWriteConfig` and `_LegacyShardingSpec`** — these backward-compatibility aliases have been removed with no deprecation cycle.

### Removed
- **`route_key()` from `_writer_core.py`** — writer-side routing now uses `route_hash()` or `route_cel()` directly. Reader `route_key()` methods remain unchanged.

### Changed
- **Internal sharding column is dropped before value encoding** in DataFrame-based writers (Spark, Dask, Ray). When using `json_cols(None)` (or `json_cols()` with no explicit columns), the internal `_shard_id` / `_vector_db_id` column is no longer included in the serialized stored value. The column name is configurable via `BaseShardedWriteConfig.shard_id_col` (KV writers, default `_shard_id`) and `VectorShardedWriteConfig.shard_id_col` (vector writers, default `_vector_db_id`). A `ConfigValidationError` is raised at write time if the configured name collides with a user data column.
