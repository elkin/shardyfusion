# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`slatedb_spark_sharded` is a sharded snapshot writer/reader library for SlateDB. It provides:
- Writer-side Spark APIs to build `num_dbs` independent SlateDB shard databases
- Pure-Python iterator-based writer (no Spark/Java needed)
- Manifest + `_CURRENT` publishing protocol (default S3, pluggable interfaces)
- Reader-side routing helpers for service-side `get` and `multi_get`
- `slate-reader` CLI for interactive and batch lookups against published snapshots
- Token-bucket rate limiter (`max_writes_per_second`) for both writer paths

## Commands

### Install Dependencies

```bash
uv sync --extra read           # Reader-only (no Spark/Java required)
uv sync --extra writer         # Writer-side (includes Spark, requires Java)
uv sync --extra cli            # CLI only (click + pyyaml, no Spark/Java)
uv sync --all-extras --dev     # Full dev environment
```

### Lint, Format, Type Check

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check slatedb_spark_sharded --python-version 3.11 --error-on-warning
uv run pyright slatedb_spark_sharded
```

### Tests

```bash
# Targeted pytest (prefer this while developing)
uv run pytest -q tests/unit/cli
uv run pytest -q tests/unit/writer
uv run pytest -q tests/unit/read
uv run pytest -q tests/unit/shared
uv run pytest -q tests/integration/read
uv run pytest -q tests/integration/writer

# Tox labels
uv run tox -m quality       # lint, format, type, type-fallback, package, docs-check
uv run tox -m unit          # unit tests across py311-py314 and Spark 3.5/4
uv run tox -m integration   # integration tests across py311-py314
uv run tox -m e2e           # end-to-end tests against Garage S3

# Specific tox environments (examples — see `uv run tox -l` for full list)
uv run tox -e py311-all-spark35-unit
uv run tox -e py314-read-unit
uv run tox -e py311-read-integration
uv run tox -e py311-writer-spark4-integration
uv run tox -e py311-read-e2e

# Parallel tox (cap to avoid OOM)
uv run tox p -p 2
```

### Local Shortcuts (via justfile)

```bash
just sync          # uv sync --all-extras --dev
just fix           # ruff check --fix + ruff format (auto-fix)
just quality       # tox -m quality
just quality-p     # tox -m quality, parallel (-p 4)
just unit          # tox -m unit
just unit-p        # tox -m unit, parallel (-p 2)
just integration   # tox -m integration
just integration-p # tox -m integration, parallel (-p 2)
just ci            # quality → unit → integration in sequence
```

### Container Workflows (via justfile)

```bash
just docker-build    # build the CI image
just docker-shell    # interactive shell inside the container
just d-quality       # quality checks in container
just d-unit          # unit tests in container
just d-e2e           # e2e tests against Garage (via compose)
just d-ci            # quality → unit → integration in container
just d <command>     # arbitrary command in container

# Use Docker instead of Podman (default is podman)
CONTAINER_ENGINE=docker just docker-build
```

The container uses an isolated project venv at `/opt/slatedb-venv`, not the host `.venv`.

### Build & Docs

```bash
uv build
uv run mkdocs build --strict
```

### CI Pipeline (GitHub Actions)

On every push/PR, CI runs in order: **quality → package → unit → integration** (all parallel within each stage).

- **Quality**: lint + format + ty + pyright (Python 3.11 only)
- **Package**: `uv build` + wheel smoke-test import
- **Unit**: matrix of py3.11–3.14 × Spark 3.5/4 (read-unit, writer-unit, all-unit)
- **Integration**: matrix of py3.11–3.14 × Spark 3.5/4 (moto-backed S3)
- **E2E**: runs via `just d-e2e` (not in GitHub Actions — local/container only)

Java 17 (temurin) is used for all Spark-requiring CI jobs. Weekly scheduled build on Mondays at 06:00 UTC.

## Architecture

The library is split into four independent paths that share config, manifest models, and core logic:

**Writer path (Spark)** (requires PySpark + Java):
`writer/spark/writer.py` → `writer/spark/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`

**Writer path (Python)** (no Spark/Java needed):
`writer/python/writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`

**Reader path** (no Spark/Java needed):
`reader/reader.py` → `routing.py` → `manifest_readers.py`

**CLI path** (requires click + pyyaml):
`cli/app.py` → `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`

### Module Dependency Graph

```
Layer 0 — Core types & errors (no internal deps):
  errors.py, type_defs.py, sharding_types.py, ordering.py, logging.py

Layer 1 — Config & serialization:
  config.py (→ sharding_types)
  serde.py (→ sharding_types)
  _rate_limiter.py (standalone)

Layer 2 — Storage, publish, routing, manifest:
  storage.py (standalone)
  manifest.py (→ sharding_types)
  publish.py (→ manifest, storage)
  routing.py (→ manifest, serde, sharding_types, ordering)
  manifest_readers.py (→ manifest, storage)

Layer 3 — Writer core (shared by Spark + Python writers):
  _writer_core.py (→ config, manifest, publish, routing, serde, sharding_types, storage)

Layer 4 — Entry points:
  writer/spark/sharding.py (→ config, sharding_types)
  writer/spark/writer.py (→ _writer_core, writer/spark/sharding, config, serde, slatedb_adapter)
  writer/python/writer.py (→ _writer_core, _rate_limiter, config, serde, slatedb_adapter)
  reader/reader.py (→ manifest_readers, routing, sharding_types)
  cli/app.py → cli/config.py, cli/output.py, cli/interactive.py, cli/batch.py

Layer 5 — Adapters & testing:
  slatedb_adapter.py (→ storage, type_defs)
  testing.py (→ storage; fake/file-backed/real adapters for tests)
```

### Write Pipeline

**Spark writer** (`write_sharded_spark`):
1. Entry point in `writer/spark/writer.py`. Optionally applies Spark conf overrides via `SparkConfOverrideContext`.
2. `writer/spark/sharding.py` adds `_slatedb_db_id` column via Spark SQL expressions (hash, range, or custom), then converts the DataFrame to a pair RDD partitioned so partition index = db_id.
3. Each partition writes one shard to S3 at a temporary path (`_tmp/run_id=.../db=XXXXX/attempt=YY/`).
4. The driver collects results and selects deterministic winners (lowest attempt → task_attempt_id → URL).
5. A manifest artifact is built and published, then the `_CURRENT` pointer is updated.

**Python writer** (`write_sharded`):
1. Entry point in `writer/python/writer.py`. Accepts `Iterable[T]` with `key_fn`/`value_fn` callables.
2. Supports single-process (all adapters open simultaneously) or multi-process (`parallel=True`, one worker per shard via `multiprocessing.spawn`).
3. Optional rate limiting via `max_writes_per_second` (token-bucket in `_rate_limiter.py`).
4. Uses the same `_writer_core.py` functions for routing, winner selection, manifest building, and publishing.

### Read Pipeline

1. `SlateShardedReader` in `reader/reader.py` loads the `_CURRENT` pointer from S3 and dereferences the manifest.
2. Builds a `SnapshotRouter` from the manifest sharding metadata (mirrors write-time sharding logic).
3. `get(key)` / `multi_get(keys)` routes keys to shard IDs, then reads from the appropriate shard.
4. `refresh()` atomically swaps readers using reference counting for safe cleanup of in-flight operations.
5. Thread safety via `threading.Lock` (default) or `ThreadPoolExecutor` pool mode (`thread_safety` config).

### CLI (`slate-reader`)

Entry point: `cli/app.py:main` (registered in `pyproject.toml` as `slate-reader`).
Install with `uv sync --extra cli` (or `--all-extras`). Requires `click>=8.0` and `pyyaml>=6.0`.

Subcommands: `get KEY`, `multiget KEY [KEY ...]`, `info`, `refresh`, `exec --script FILE`.
No subcommand → enters interactive REPL (`cmd.Cmd` with `slate> ` prompt).

Key coercion: CLI keys are strings; when manifest uses integer encoding (`u64be`/`u32be`), keys are auto-coerced to `int` via `cli/config.py:coerce_cli_key()`.

| Module | Responsibility |
|---|---|
| `cli/app.py` | Click group, subcommands, reader construction, entry point |
| `cli/config.py` | TOML loading, CURRENT URL resolution, S3 client config, `coerce_cli_key`, `coerce_s3_option` |
| `cli/output.py` | Value encoding (`base64`/`hex`/`utf8`), result builders, `format_result`, `emit` |
| `cli/interactive.py` | `SlateReaderRepl` (`cmd.Cmd` subclass) |
| `cli/batch.py` | YAML script loading and execution |

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
| `ManifestPublisher` | `DefaultS3Publisher` | Where/how to publish manifests |
| `ManifestReader` | `DefaultS3ManifestReader` | How to load manifest/CURRENT |
| `DbAdapterFactory` | `SlateDbFactory` | How to open shard databases |

`ValueSpec` controls how DataFrame rows are serialized to bytes: `binary_col`, `json_cols`, or a callable encoder.

### Key Encodings

The `key_encoding` field on `WriteConfig` (default `"u64be"`) controls how keys are serialized to bytes in SlateDB:

- **`u64be`** (default): 8-byte big-endian unsigned integer. Keys in `[0, 2^64-1]`.
- **`u32be`**: 4-byte big-endian unsigned integer. Keys in `[0, 2^32-1]`. Cuts key storage in half.

Both encodings produce identical hash routing results for keys in `[0, 2^32-1]` because the routing layer zero-extends to 8-byte little-endian before hashing (matching Spark's `xxhash64(cast(key as long))`). The encoding is stored in the manifest and used by the reader/CLI for key coercion and lookup encoding.

### Sharding Strategies

- **Hash** (default): `pmod(xxhash64(cast(key as long)), num_dbs)` — requires integer key column
- **Range**: Explicit boundaries or computed via `approxQuantile` — supports int/float/string keys
- **Custom**: User-provided Spark SQL expression or column builder callable (Spark writer only)

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
1. `_verify_routing_agreement()` in `writer/spark/writer.py` — runtime spot-check comparing Spark SQL vs Python routing on a sample of written rows. Controlled by `verify_routing=True` (default).
2. `tests/unit/writer/test_routing_contract.py` — hypothesis property tests + Spark cross-checks covering ~200 edge-case keys × multiple `num_dbs` values.
3. Single implementation in `routing.py:_xxhash64_db_id()` imported by both writer paths (never reimplemented).

### Error Hierarchy

All errors inherit from `SlatedbSparkShardedError` which carries a `retryable: bool` flag:

| Error | Retryable | Trigger |
|---|---|---|
| `ConfigValidationError` | No | Invalid config (missing boundaries, unsupported strategy) |
| `ShardAssignmentError` | No | Rows assigned to invalid shard IDs |
| `ShardCoverageError` | No | Partition results didn't cover all expected shard IDs |
| `SlateDbApiError` | No | SlateDB binding unavailable or incompatible |
| `ManifestBuildError` | No | Manifest artifact creation failed |
| `ManifestParseError` | No | Manifest/CURRENT payload couldn't be parsed |
| `ReaderStateError` | No | Reader operation in invalid lifecycle state |
| `PublishManifestError` | Yes | Manifest publish failed (transient infra) |
| `PublishCurrentError` | Yes | CURRENT pointer publish failed (transient infra) |
| `S3TransientError` | Yes | Transient S3 error (throttle, 500, 503, timeout) |

## Public API Summary

Exported from `__init__.py` (conditional on installed extras):

**Always available:** `WriteConfig`, `ManifestOptions`, `OutputOptions`, `ShardingSpec`, `ShardingStrategy`, `KeyEncoding`, `ValueSpec`, `ManifestArtifact`, `ManifestBuilder`, `JsonManifestBuilder`, `ManifestPublisher`, `DefaultS3Publisher`, `ManifestReader`, `DefaultS3ManifestReader`, `FunctionManifestReader`, `SnapshotRouter`, `SlateShardedReader`, `SlateDbReaderFactory`, `SlateDbFactory`, `DbAdapter`, `DbAdapterFactory`, `BuildResult`, `BuildStats`, `BuildDurations`, `CurrentPointer`, `RequiredBuildMeta`, `RequiredShardMeta`, `ManifestShardingSpec`, `ShardReaderFactory`, `parse_json_manifest`, `ManifestParseError`, `ReaderStateError`, `FailureSeverity`

**With `writer` extra:** `write_sharded_spark`, `DataFrameCacheContext`, `SparkConfOverrideContext`

**With base deps (no extra):** `write_sharded` (Python writer — `xxhash` + `pydantic` for import; needs `boto3` at publish time unless using a custom `ManifestPublisher`)

## Testing Notes

- **`tests/unit/`** — fast, no Spark; use pytest-xdist parallelism (`-n 2`)
  - `tests/unit/shared/` — config, manifest, publish tests
  - `tests/unit/read/` — reader, manifest_readers tests
  - `tests/unit/writer/` — routing, serde, sharding, winner selection, python writer tests
  - `tests/unit/cli/` — CLI unit tests (Click CliRunner with mocked reader)
- **`tests/integration/`** — requires S3 emulation via `moto`; writer tests additionally require Spark + Java
- **`tests/e2e/`** — end-to-end tests against Garage S3 server via compose; run with `just d-e2e`
- **`tests/helpers/`** — shared test scenarios (`s3_test_scenarios.py`) used by both integration (moto) and e2e (Garage) suites
- `@pytest.mark.spark` marks tests requiring a local PySpark session
- `@pytest.mark.e2e` marks end-to-end tests requiring a container engine
- For behavior changes: add/adjust unit tests first, then integration tests where routing/publishing or Spark behavior is affected

### Testing Patterns

- **Contract tests** (`test_routing_contract.py`): hypothesis property tests verifying sharding invariant holds across random keys, plus Spark-vs-Python cross-checks with ~200 edge-case keys.
- **Shared scenarios** (`tests/helpers/s3_test_scenarios.py`): backend-agnostic test functions accepting an `LocalS3Service` dict — reused across moto (integration) and Garage (e2e).
- **Test adapters** (`testing.py`): `FakeSlateDbAdapter` (in-memory counter), `FileBackedSlateDbAdapter` (JSONL persistence), `RealSlateDbFileAdapter` (real SlateDB with `file://` URLs) — all implement `DbAdapter` protocol.

### E2E Tests

E2e tests run against a Garage S3 server via compose (`podman compose` or `docker compose`):

```bash
just d-e2e                           # Podman (default)
CONTAINER_ENGINE=docker just d-e2e   # Docker
```

The Garage image is built from `docker/garage-e2e.Dockerfile`. The test runner connects to Garage at `http://garage:3900` via compose networking — no ports are exposed to the host.

## Coding Conventions

- Python 3.11+ with full type hints (pyright + ruff enforced)
- Ruff rules: `E`, `F`, `I`; line length 88; **`E501` ignored** (long lines allowed)
- Keep Spark logic in writer-side modules; avoid Python UDFs when Spark built-ins are available
- Dataclasses with `slots=True` for performance-sensitive models
- Manifest models use Pydantic (`ConfigDict(use_enum_values=False)`) — enum names serialize, not values
- Commit subjects: imperative, Conventional Commit prefixes recommended (`fix:`, `feat:`, `chore:`, `test:`)

## Gotchas & Non-obvious Behavior

- **SlateDB import is deferred**: `slatedb_adapter.py` imports the `slatedb` module inside `__init__`, not at module load. Tests monkeypatch `sys.modules["slatedb"]` to provide fakes.
- **Python writer parallel mode uses `spawn`**: `multiprocessing.get_context("spawn")` is hardcoded. All objects passed to workers must be picklable. Min/max key tracking happens in the main process.
- **Rate limiter releases lock before sleeping**: `TokenBucket.acquire()` releases the lock during the sleep interval, allowing other threads to proceed without starvation.
- **Reader reference counting**: `refresh()` atomically swaps `_ReaderState` but defers closing old handles until `refcount` drops to zero. Never close handles that might have in-flight reads.
- **Garage requires path-style addressing**: E2E tests set `addressing_style: "path"` in `S3ClientConfig` because Garage doesn't support virtual-hosted-style.
- **Session-scoped test fixtures**: PySpark and S3 (moto/Garage) fixtures are session-scoped for performance. Tests share the same Spark session and S3 service.
- **Writer scenario imports are deferred**: `tests/helpers/s3_test_scenarios.py` imports writer modules inside function bodies so reader-only test collection doesn't fail.
- **`OutputOptions.local_root` defaults to `/tmp/slatedb-spark`**: Not per-user; may cause permission issues in shared environments.

## Environment Notes

- Spark-based writer flows require **Java 17** (`JAVA_HOME` or `PATH`). CI uses temurin distribution.
- `SPARK_LOCAL_IP=127.0.0.1` is set in tox and devcontainer to avoid hostname resolution issues.
- Reader-only and Python writer usage do not require Spark/Java.
- CLI requires `click>=8.0` and `pyyaml>=6.0` (`uv sync --extra cli`).
