# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`slatedb_spark_sharded` is a sharded snapshot writer/reader library for SlateDB. It provides:
- Writer-side Spark APIs to build `num_dbs` independent SlateDB shard databases
- Manifest + `_CURRENT` publishing protocol (default S3, pluggable interfaces)
- Reader-side routing helpers for service-side `get` and `multi_get`
- `slate-reader` CLI for interactive and batch lookups against published snapshots

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
uv run ty check slatedb_spark_sharded --python-version 3.10 --error-on-warning
uv run pyright slatedb_spark_sharded
```

### Tests

```bash
# Targeted pytest (prefer this while developing)
uv run pytest -q tests/unit/cli
uv run pytest -q tests/unit/writer
uv run pytest -q tests/integration/read

# Tox labels
uv run tox -m quality       # lint, format, type, type-fallback, package, docs-check
uv run tox -m unit          # unit tests across Python 3.10/3.11 and Spark 3.5/4
uv run tox -m integration   # integration tests

# Specific tox environments
uv run tox -e py311-all-spark35-unit
uv run tox -e py311-read-integration
uv run tox -e py311-writer-spark4-integration

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

# Named container recipes (mirrors local shortcuts)
just d-quality       # quality checks in container
just d-quality-p     # quality checks in container, parallel
just d-unit          # unit tests in container
just d-unit-p        # unit tests in container, parallel
just d-integration   # integration tests in container
just d-integration-p # integration tests in container, parallel
just d-e2e           # e2e tests against Garage (via compose)
just d-ci            # quality → unit → integration in container

# Arbitrary command in container
just d uv run tox -e py311-all-spark35-unit

# Use Docker instead of Podman (default is podman)
CONTAINER_ENGINE=docker just docker-build
CONTAINER_ENGINE=docker just d-quality
```

The container uses an isolated project venv at `/opt/slatedb-venv`, not the host `.venv`.

### Build & Docs

```bash
uv build
uv run mkdocs build --strict
```

## Architecture

The library is split into three independent paths that share config and manifest models:

**Writer path (Spark)** (requires PySpark + Java): `writer/spark/writer.py` → `writer/spark/sharding.py` → `serde.py` → `slatedb_adapter.py`

**Writer path (Python)** (no Spark/Java needed): `writer/python/writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`

**Reader path** (no Spark/Java needed): `reader/reader.py` → `routing.py` → `manifest_readers.py`

**CLI path** (requires click + pyyaml): `cli/app.py` → `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`

**Shared models/protocols**: `config.py`, `manifest.py`, `publish.py`, `storage.py`

### Write Pipeline

1. `write_sharded_spark(df, config, *, key_col, value_spec)` in `writer/spark/writer.py` is the entry point.
2. `writer/spark/sharding.py` adds a `_slatedb_db_id` column via Spark SQL expressions (hash, range, or custom), then converts the DataFrame to a pair RDD partitioned so partition index = db_id.
3. Each partition writes one shard to S3 at a temporary path (`_tmp/run_id=.../db=XXXXX/attempt=YY/`).
4. The driver collects results and selects deterministic winners (lowest attempt → task_attempt_id → URL).
5. A manifest artifact is built and published, then the `_CURRENT` pointer is updated.

### Read Pipeline

1. `SlateShardedReader` in `reader/reader.py` loads the `_CURRENT` pointer from S3 and dereferences the manifest.
2. Builds a `SnapshotRouter` from the manifest sharding metadata (mirrors write-time sharding logic).
3. `get(key)` / `multi_get(keys)` routes keys to shard IDs, then reads from the appropriate shard.
4. `refresh()` atomically swaps readers using reference counting for safe cleanup of in-flight operations.

### CLI (`slate-reader`)

The `slate-reader` entry point is registered in `pyproject.toml` (`project.scripts`).
Install with `uv sync --extra cli` (or `--all-extras`). Requires `click>=8.0` and `pyyaml>=6.0`.

#### Usage

```bash
# One-shot subcommands
slate-reader --current-url s3://bucket/prefix/_CURRENT get 42
slate-reader --current-url s3://bucket/prefix/_CURRENT multiget 1 2 3
slate-reader --current-url s3://bucket/prefix/_CURRENT info
slate-reader --current-url s3://bucket/prefix/_CURRENT refresh

# Batch script execution
slate-reader --current-url s3://bucket/prefix/_CURRENT exec --script batch.yaml --output results.jsonl

# Interactive REPL (entered when no subcommand is given)
slate-reader --current-url s3://bucket/prefix/_CURRENT
```

#### Global Options

| Option | Description |
|---|---|
| `--current-url URL` | S3 URL to the `_CURRENT` pointer (overrides env / config) |
| `--config PATH` | Path to `reader.toml` (overrides `SLATE_READER_CONFIG` env) |
| `--credentials PATH` | Path to `credentials.toml` (overrides `SLATE_READER_CREDENTIALS` env) |
| `--s3-option KEY=VALUE` | Override S3 connection option (repeatable) |
| `--output-format FORMAT` | Output format: `json`, `jsonl` (default), `table`, `text` |

#### Subcommands

| Subcommand | Arguments | Description |
|---|---|---|
| `get` | `KEY` | Look up a single key |
| `multiget` | `KEY [KEY ...]` | Look up multiple keys (space-separated) |
| `info` | — | Show manifest metadata (run_id, num_dbs, sharding strategy) |
| `refresh` | — | Reload `_CURRENT` and manifest |
| `exec` | `--script FILE [--output FILE]` | Execute a YAML batch script |

#### CURRENT URL Resolution

The CURRENT URL is resolved in priority order:
1. `--current-url` CLI option
2. `SLATE_READER_CURRENT` environment variable
3. `current_url` in the `[reader]` section of `reader.toml`

#### Key Coercion

CLI keys are always strings. When the manifest uses an integer key encoding (`u64be` or `u32be`), keys are automatically coerced from string to `int` before lookup. For other encodings (e.g. `utf8`), keys are passed as strings. The coercion logic lives in `cli/config.py:coerce_cli_key()`.

#### Configuration Files

**`reader.toml`** — searched in order: `./reader.toml`, `~/.config/slatefusion/reader.toml`, or via `SLATE_READER_CONFIG` env / `--config` flag.

```toml
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
local_root = "/tmp/slatefusion"
thread_safety = "lock"        # "lock" or "pool"
max_workers = 4
slate_env_file = "/path/to/env"
credentials_profile = "default"

[output]
format = "jsonl"              # json | jsonl | table | text
value_encoding = "base64"     # base64 | hex | utf8
null_repr = "null"
```

**`credentials.toml`** — searched in order: `./credentials.toml`, `~/.config/slatefusion/credentials.toml`, or via `SLATE_READER_CREDENTIALS` env / `--credentials` flag. Warns if file permissions are wider than `0600`.

```toml
[default]
endpoint_url = "http://localhost:9000"
region = "us-east-1"
access_key_id = "..."
secret_access_key = "..."
addressing_style = "path"
verify_ssl = true
connect_timeout = 10
read_timeout = 30
max_attempts = 3
```

#### Batch Scripts

YAML files with a `commands` list. Each command has an `op` field (`get`, `multiget`, `refresh`, `info`). Optional `on_error` at the top level: `"stop"` (default) or `"continue"`. Batch mode defaults output to `jsonl`.

```yaml
on_error: continue
commands:
  - op: get
    key: 42
  - op: multiget
    keys: [1, 2, 3]
  - op: info
  - op: refresh
```

#### Interactive REPL

When no subcommand is given, the CLI enters a `cmd.Cmd` REPL with a `slate> ` prompt. REPL commands: `get KEY`, `multiget KEY [KEY ...]`, `info`, `refresh`, `quit`/`exit`/`Ctrl-D`. Interactive mode defaults output to `json` (pretty) instead of `jsonl`.

#### CLI Module Architecture

| Module | Responsibility |
|---|---|
| `cli/app.py` | Click group, subcommands, reader construction, entry point |
| `cli/config.py` | TOML loading, CURRENT URL resolution, S3 client config, `coerce_cli_key`, `coerce_s3_option` |
| `cli/output.py` | Value encoding (`base64`/`hex`/`utf8`), result builders, `format_result`, `emit` |
| `cli/interactive.py` | `SlateReaderRepl` (`cmd.Cmd` subclass) |
| `cli/batch.py` | YAML script loading and execution |

### Key Abstractions

These are all Protocols, allowing user-provided implementations:

| Protocol | Default Implementation | Purpose |
|---|---|---|
| `ManifestBuilder` | `JsonManifestBuilder` | Manifest serialization format |
| `ManifestPublisher` | `DefaultS3Publisher` | Where/how to publish manifests |
| `ManifestReader` | `DefaultS3ManifestReader` | How to load manifest/CURRENT |

`ValueSpec` controls how DataFrame rows are serialized to bytes: `binary_col`, `json_cols`, or a callable encoder.

### Key Encodings

The `key_encoding` field on `WriteConfig` (default `"u64be"`) controls how keys are serialized to bytes in SlateDB:

- **`u64be`** (default): 8-byte big-endian unsigned integer. Supports keys in `[0, 2^64-1]`.
- **`u32be`**: 4-byte big-endian unsigned integer. Supports keys in `[0, 2^32-1]`. Cuts key storage in half for datasets with sequential keys up to ~4.3B.

Both encodings produce identical hash routing results for keys in `[0, 2^32-1]` because the routing layer zero-extends to 8-byte little-endian before hashing (matching Spark's `xxhash64(cast(key as long))`). The encoding is stored in the manifest and used by the reader/CLI for key coercion and lookup encoding.

### Sharding Strategies

- **Hash** (default): `pmod(xxhash64(cast(key as long)), num_dbs)` — requires integer key column
- **Range**: Explicit boundaries or computed via `approxQuantile` — supports int/float/string keys
- **Custom**: User-provided Spark SQL expression or column builder callable

The `SnapshotRouter` in `routing.py` mirrors the writer's sharding logic exactly for consistent key routing at read time.

## Testing Notes

- `tests/unit/` — fast, no Spark; use pytest-xdist parallelism
  - `tests/unit/cli/` — CLI unit tests (Click CliRunner with mocked reader, config/output helpers)
- `tests/integration/` — requires S3 emulation via `moto`; writer tests additionally require Spark + Java
- `tests/e2e/` — end-to-end tests against a real S3-compatible server (Garage via compose); run with `just d-e2e`
- `tests/helpers/` — shared test scenarios used by both integration (moto) and e2e (Garage) suites
- `@pytest.mark.spark` marks tests requiring a local PySpark session
- `@pytest.mark.e2e` marks end-to-end tests requiring a container engine
- For behavior changes: add/adjust unit tests first, then integration tests where routing/publishing or Spark behavior is affected

### E2E Tests

E2e tests run against a Garage S3 server via compose (`podman compose` or `docker compose`). The compose stack (`docker/compose-e2e.yaml`) starts a Garage service and a test runner on a shared network:

```bash
# Via just (recommended)
just d-e2e

# Use Docker instead of Podman
CONTAINER_ENGINE=docker just d-e2e
```

The Garage image is built from `docker/garage-e2e.Dockerfile` (Alpine wrapper around the official `dxflrs/garage` image). The test runner connects to Garage at `http://garage:3900` via compose networking — no ports are exposed to the host.

## Coding Conventions

- Python 3.11+ with full type hints (pyright + ruff enforced)
- Ruff rules: `E`, `F`, `I`; line length 88
- Keep Spark logic in writer-side modules; avoid Python UDFs when Spark built-ins are available
- Dataclasses with `slots=True` for performance-sensitive models
- Commit subjects: imperative, Conventional Commit prefixes recommended (`fix:`, `feat:`, `chore:`, `test:`)
