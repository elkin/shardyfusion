# Repository Guidelines

## Project Structure & Module Organization
- Source package: `shardyfusion/`
  - Writer path (Spark): `writer/spark/writer.py`, `writer/spark/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Writer path (Dask): `writer/dask/writer.py`, `writer/dask/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Writer path (Ray): `writer/ray/writer.py`, `writer/ray/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Writer path (Python): `writer/python/writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py` or `sqlite_adapter.py`
  - Reader path (sync): `reader/reader.py`, `reader/concurrent_reader.py`, `routing.py`, `manifest_store.py`, `db_manifest_store.py`
  - Reader path (async): `reader/async_reader.py`, `routing.py`, `async_manifest_store.py` (`AsyncS3ManifestStore`)
  - CLI path: `cli/app.py`, `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`
  - SQLite backend: `sqlite_adapter.py` (writer + download-and-cache reader + range-read VFS reader + async wrappers)
  - Vector search: `vector/types.py`, `vector/config.py`, `vector/sharding.py`, `vector/_merge.py`, `vector/writer.py`, `vector/reader.py`, `vector/adapters/usearch_adapter.py`, `vector/adapters/sqlite_vec_adapter.py` (vector-only shim)
  - Unified KV+vector: `sqlite_vec_adapter.py` (sqlite-vec unified adapter), `composite_adapter.py` (KV+vector composite), `reader/unified_reader.py` (UnifiedShardedReader)
  - Shared models/protocols: `config.py`, `credentials.py`, `manifest.py`, `manifest_store.py`, `async_manifest_store.py`, `run_registry.py`, `storage.py`, `testing.py`
  - Core utilities: `errors.py`, `type_defs.py`, `sharding_types.py`, `ordering.py`, `logging.py`, `serde.py`, `cel.py`, `_rate_limiter.py`, `metrics/`
- Tests: `tests/`
  - Unit: `tests/unit/shared`, `tests/unit/read`, `tests/unit/read_async`, `tests/unit/writer`, `tests/unit/backend/{slatedb,sqlite}`, `tests/unit/cel`, `tests/unit/metrics`, `tests/unit/cli`, `tests/unit/vector`
  - Integration: `tests/integration/read`, `tests/integration/writer`, `tests/integration/backend/sqlite`, `tests/integration/vector`
  - E2E: `tests/e2e/read`, `tests/e2e/writer` (Garage S3 via compose)
  - Shared helpers: `tests/helpers/s3_test_scenarios.py`, `tests/helpers/smoke_scenarios.py`, `tests/helpers/run_record_assertions.py`
- Docs: `docs/` with MkDocs config in `mkdocs.yml`
- Local container/dev setup: `docker/ci.Dockerfile`, `.devcontainer/devcontainer.json`

## Build, Test, and Development Commands
- Install dependencies:
  - Default reader (SlateDB): `uv sync --extra read`
  - Async reader (SlateDB + aiobotocore): `uv sync --extra read-async`
  - SQLite readers: `uv sync --extra read-sqlite`, `uv sync --extra read-sqlite-range`, `uv sync --extra sqlite-async`
  - Spark writer: `uv sync --extra writer-spark` or `uv sync --extra writer-spark-sqlite`
  - Python writer: `uv sync --extra writer-python` or `uv sync --extra writer-python-sqlite`
  - Dask writer: `uv sync --extra writer-dask` or `uv sync --extra writer-dask-sqlite`
  - Ray writer: `uv sync --extra writer-ray` or `uv sync --extra writer-ray-sqlite`
  - CLI-only: `uv sync --extra cli`
  - Observability extras: `uv sync --extra metrics-prometheus`, `uv sync --extra metrics-otel`
  - Vector search: `uv sync --extra vector` (USearch), `uv sync --extra vector-sqlite` (sqlite-vec), `uv sync --extra unified-vector` (KV+vector)
  - Full dev: `uv sync --all-extras --dev`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`
- Auto-fix: `just fix` (ruff check --fix + ruff format)
- Type check: `uv run pyright shardyfusion`
- Package build: `uv build`
- Tox labels:
  - `uv run tox -m quality` — lint, format, type, package, docs-check
  - `uv run tox -m unit` — unit tests across py311-py314 and Spark 3.5/4
  - `uv run tox -m integration` — integration tests across py311-py314
  - `uv run tox -m e2e` — end-to-end tests against Garage S3
- Just shortcuts (grouped by category):
  - Setup: `just setup` (bootstrap), `just doctor` (health check), `just clean`, `just clean-all`
  - Dev: `just sync`, `just fix`, `just docs`, `just docs-serve`
  - Test: `just quality`, `just unit`, `just integration`, `just ci`
- Container workflows via `justfile`:
  - Build image: `just d-build`
  - Run shell in container: `just d-shell`
  - Run any local-style command in container: `just d <command>`
  - Named recipes: `just d-quality`, `just d-unit`, `just d-integration`, `just d-e2e`, `just d-ci`, `just d-clean`
  - Use Docker instead of Podman: `CONTAINER_ENGINE=docker just d-build`
  - Container runs use an isolated uv project env at `/opt/shardyfusion-venv` (not host `.venv`).

## Coding Style & Naming Conventions
- Python 3.11+ with full type hints.
- Formatting/linting via Ruff (`E`, `F`, `I`; line length 88; `E501` ignored — long lines allowed).
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Dataclasses with `slots=True` for performance-sensitive models.
- Keep Spark logic in writer-side modules; avoid introducing Python UDFs when Spark built-ins are available.

## Testing Guidelines
- Framework: `pytest`; parallel unit execution uses `pytest-xdist`.
- Test files: `test_*.py`; keep reader and writer tests in their flavor directories.
- Prefer targeted runs while developing:
  - `uv run pytest -q tests/unit/writer`
  - `uv run pytest -q tests/unit/cli`
  - `uv run pytest -q tests/integration/read`
- Integration writer tests require Spark + Java and local S3 emulation (`moto`).
- E2E tests require a container engine and run via `just d-e2e`.
- Contract tests in `tests/unit/writer/core/test_routing_contract.py` verify writer-reader sharding invariant with hypothesis property tests.
- Shared test scenarios in `tests/helpers/s3_test_scenarios.py` are reused across moto and Garage backends.
- Test adapters in `testing.py`: `FakeSlateDbAdapter`, `FileBackedSlateDbAdapter`, `RealSlateDbFileAdapter`.
- For behavior changes, add/adjust unit tests first, then integration tests where routing/publishing or Spark behavior is affected.

## Commit & Pull Request Guidelines
- Follow concise, imperative commit subjects; Conventional Commit prefixes are used in history (`fix:`, `chore:`, `test:`, `feat:`) and recommended.
- Keep commits scoped (single concern) and include tests for functional changes.
- PRs should include:
  - What changed and why
  - Impacted area (`read`, `writer`, `cli`, or shared)
  - Commands run (e.g., `tox -m quality`, targeted/unit/integration tests)

## Environment & Configuration Notes
- Spark-based writer flows require Java (`JAVA_HOME` or `PATH`).
- Reader-only, Python writer, Dask writer, and Ray writer usage do not require Spark/Java.
- CLI requires `click>=8.0` and `pyyaml>=6.0` (`uv sync --extra cli`).
