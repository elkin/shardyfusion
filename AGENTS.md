# Repository Guidelines

## Project Structure & Module Organization
- Source package: `shardyfusion/` (31 modules)
  - Writer path (Spark): `writer/spark/writer.py`, `writer/spark/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`
  - Writer path (Dask): `writer/dask/writer.py`, `writer/dask/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`
  - Writer path (Ray): `writer/ray/writer.py`, `writer/ray/sharding.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`
  - Writer path (Python): `writer/python/writer.py` → `_writer_core.py` → `serde.py` → `slatedb_adapter.py`
  - Reader path: `reader/reader.py`, `routing.py`, `manifest_readers.py`
  - CLI path: `cli/app.py`, `cli/config.py`, `cli/output.py`, `cli/interactive.py`, `cli/batch.py`
  - Shared models/protocols: `config.py`, `manifest.py`, `publish.py`, `storage.py`, `testing.py`
  - Core utilities: `errors.py`, `type_defs.py`, `sharding_types.py`, `ordering.py`, `logging.py`, `serde.py`, `_rate_limiter.py`
- Tests: `tests/`
  - Unit: `tests/unit/shared`, `tests/unit/read`, `tests/unit/writer`, `tests/unit/cli`
  - Integration: `tests/integration/read`, `tests/integration/writer`
  - E2E: `tests/e2e/read`, `tests/e2e/writer` (Garage S3 via compose)
  - Shared helpers: `tests/helpers/s3_test_scenarios.py` (backend-agnostic test scenarios)
- Docs: `docs/` with MkDocs config in `mkdocs.yml`
- Local container/dev setup: `docker/ci.Dockerfile`, `.devcontainer/devcontainer.json`

## Build, Test, and Development Commands
- Install dependencies:
  - Reader-only: `uv sync --extra read`
  - Spark writer: `uv sync --extra writer-spark`
  - Python writer: `uv sync --extra writer-python`
  - Dask writer: `uv sync --extra writer-dask`
  - Ray writer: `uv sync --extra writer-ray`
  - CLI-only: `uv sync --extra cli`
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
- Just shortcuts: `just sync`, `just fix`, `just quality`, `just quality-p`, `just unit`, `just unit-p`, `just integration`, `just integration-p`, `just ci`
- Container workflows via `justfile`:
  - Build image: `just docker-build`
  - Run shell in container: `just docker-shell`
  - Run any local-style command in container: `just d <command>`
  - Named recipes: `just d-quality`, `just d-unit`, `just d-integration`, `just d-e2e`, `just d-ci`
  - Use Docker instead of Podman: `CONTAINER_ENGINE=docker just docker-build`
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
- Contract tests in `test_routing_contract.py` verify writer-reader sharding invariant with hypothesis property tests.
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
