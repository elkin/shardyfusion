# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

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
- **Renamed vector writer entry points to `write_sharded` and removed vector writer scalar keyword arguments** (breaking). Standalone vector writes use `shardyfusion.vector.write_sharded(records, config, input=None, options=None)`. Distributed vector writes use `write_sharded(data, config, input, options=None)` from `shardyfusion.writer.spark.vector_writer`, `shardyfusion.writer.dask.vector_writer`, or `shardyfusion.writer.ray.vector_writer`; column mapping now lives in `VectorColumnInput`.
- **Split writer configuration into nested public config groups** (breaking). KV configs now group storage, output, manifest, KV adapter settings, retry, rate limits, observability, lifecycle, and optional vector settings via `WriterStorageConfig`, `WriterOutputConfig`, `WriterManifestConfig`, `KeyValueWriteConfig`, `WriterRetryConfig`, `KvWriteRateLimitConfig`, `WriterObservabilityConfig`, and `WriterLifecycleConfig`. Hash and CEL sharding settings live in `HashShardingConfig` and `CelShardingConfig`.
- **Separated KV and vector write rate-limit config**. KV writers use `KvWriteRateLimitConfig(max_writes_per_second, max_write_bytes_per_second)`. Vector writers use `VectorWriteRateLimitConfig(max_writes_per_second)`. `max_writes_per_second` is no longer a separate public parameter on vector writer functions.
- **Shared distributed partition write runtime across Spark, Dask, and Ray**. Internal partition-write runtime state now uses a common `PartitionWriteRuntime.from_public_config(...)` constructor that validates public config/input first and asserts internal contracts.
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
