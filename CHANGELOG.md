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
- **`slate-reader`** command-line tool with `get`, `multiget`, `info`, `shards`, `route`, `refresh`, and `exec` subcommands.
- **Interactive REPL** mode (no subcommand → `slate>` prompt).
- **stdin support** for `multiget -` (pipe keys from stdin).
- **`--version` flag**.

#### Infrastructure
- **`ManifestStore`** protocol — unified manifest read/write interface.
- **`ManifestBuilder`** protocol for custom manifest serialization formats.
- **`DbAdapterFactory`** protocol for custom shard database backends.
- **Key encodings**: `u64be` (default, 8-byte) and `u32be` (4-byte) big-endian unsigned integer encodings.
- **JSON schema validation** for manifest and CURRENT pointer formats.
- **Metrics/observability**: `MetricsCollector` protocol with `MetricEvent` catalog for writer and reader lifecycle events.
