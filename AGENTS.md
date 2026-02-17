# Repository Guidelines

## Project Structure & Module Organization
- Source package: `slatedb_spark_sharded/`
  - Writer path: `writer.py`, `sharding.py`, `serde.py`, `slatedb_adapter.py`
  - Reader path: `reader.py`, `routing.py`, `manifest_readers.py`
  - Shared models/protocols: `config.py`, `manifest.py`, `publish.py`, `storage.py`
- Tests: `tests/`
  - Unit: `tests/unit/shared`, `tests/unit/read`, `tests/unit/writer`
  - Integration: `tests/integration/read`, `tests/integration/writer`
- Docs: `docs/` with MkDocs config in `mkdocs.yml`
- Local container/dev setup: `docker/ci.Dockerfile`, `.devcontainer/devcontainer.json`

## Build, Test, and Development Commands
- Sync dev environment: `uv sync --all-extras --dev`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`
- Type checks:
  - `uv run ty check slatedb_spark_sharded --python-version 3.10 --error-on-warning`
  - `uv run pyright slatedb_spark_sharded`
- Package build: `uv build`
- Tox labels:
  - `uv run tox -m quality`
  - `uv run tox -m unit`
  - `uv run tox -m integration`

## Coding Style & Naming Conventions
- Python 3.10+ with full type hints.
- Formatting/linting via Ruff (`E`, `F`, `I`; line length 88).
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep Spark logic in writer-side modules; avoid introducing Python UDFs when Spark built-ins are available.

## Testing Guidelines
- Framework: `pytest`; parallel unit execution uses `pytest-xdist`.
- Test files: `test_*.py`; keep reader and writer tests in their flavor directories.
- Prefer targeted runs while developing:
  - `uv run pytest -q tests/unit/writer`
  - `uv run pytest -q tests/integration/read`
- For behavior changes, add/adjust unit tests first, then integration tests where routing/publishing or Spark behavior is affected.

## Commit & Pull Request Guidelines
- Follow concise, imperative commit subjects; Conventional Commit prefixes are used in history (`fix:`, `chore:`, `test:`) and recommended.
- Keep commits scoped (single concern) and include tests for functional changes.
- PRs should include:
  - What changed and why
  - Impacted area (`read`, `writer`, or shared)
  - Commands run (e.g., `tox -m quality`, targeted/unit/integration tests)

## Environment & Configuration Notes
- Spark-based writer flows require Java (`JAVA_HOME` or `PATH`).
- Reader-only usage does not require Spark/Java.
