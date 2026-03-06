# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- **Spark writer** (`writer-spark`): PySpark DataFrame-based sharded writer with hash, range, and custom expression sharding strategies.
- **Dask writer** (`writer-dask`): Dask DataFrame-based sharded writer (hash and range sharding, no Java required).
- **Ray writer** (`writer-ray`): Ray Data Dataset-based sharded writer (hash and range sharding, no Java required).
- **Python writer** (`writer-python`): Pure-Python iterator-based writer with single-process and multi-process (`parallel=True`) modes.
- **Reader** (`read`): `ShardedReader` (non-thread-safe) and `ConcurrentShardedReader` (thread-safe with lock/pool modes) for service-side key lookups.
- **CLI** (`cli`): `slate-reader` command-line tool with interactive REPL, batch YAML execution, and `get`/`multiget`/`info`/`refresh` subcommands.
- **Two-phase publish protocol**: Manifest written first, then `_CURRENT` pointer updated, with exponential-backoff retries for transient S3 errors.
- **Token-bucket rate limiter** (`max_writes_per_second`) for all writer paths.
- **JSON schema validation** for manifest and CURRENT pointer formats.
- **Pluggable interfaces**: `ManifestBuilder`, `ManifestPublisher`, `ManifestReader`, `DbAdapterFactory` protocols for custom implementations.
- **Key encodings**: `u64be` (default, 8-byte) and `u32be` (4-byte) big-endian unsigned integer encodings.
- **Metrics/observability**: `MetricsCollector` protocol with `MetricEvent` catalog for writer and reader lifecycle events.
- **Routing verification**: Runtime spot-check (`verify_routing=True`) comparing framework-assigned shard IDs against Python routing on a sample of written rows.
