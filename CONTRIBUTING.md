# Contributing to shardyfusion

Thank you for your interest in contributing to shardyfusion! This guide covers the development setup, coding standards, and contribution workflow.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Java 17+ (only for Spark writer tests)

### Install

```bash
# Full development install with all extras
uv sync --all-extras --dev
```

If you only need specific components:

```bash
uv sync --extra read --dev       # Reader only
uv sync --extra writer-python    # Python writer only
uv sync --extra writer-dask      # Dask writer only
uv sync --extra writer-ray       # Ray writer only
uv sync --extra writer-spark     # Spark writer (requires Java)
```

### Running Quality Checks

```bash
# Lint and format
uv run ruff check .
uv run ruff format --check .

# Type checking
uv run pyright shardyfusion

# Auto-fix lint and format
just fix
```

### Running Tests

```bash
# All unit tests
uv run pytest -q tests/unit/ -n 2

# Specific area
uv run pytest -q tests/unit/read/
uv run pytest -q tests/unit/writer/python/
uv run pytest -q tests/unit/cli/

# Integration tests (uses moto for S3)
uv run pytest -q tests/integration/

# Full tox matrix
uv run tox -m quality
uv run tox -m unit
uv run tox -m integration

# Full CI locally
just ci
```

### Containerized Development

```bash
just docker-build
just d uv run tox -m quality
just d uv run tox -m unit
```

## Code Style

- **Python 3.11+** with full type hints
- **ruff** for linting and formatting (E501 ignored — long lines allowed)
- **pyright** in standard mode for type checking
- Dataclasses with `slots=True` for performance-sensitive models
- Pydantic models use `ConfigDict(use_enum_values=False)` — enum names serialize, not values
- See `pyproject.toml` for full ruff/pyright configuration

## Testing Guidelines

- **Unit tests** (`tests/unit/`): Fast, no external dependencies. Use `pytest-xdist` for parallelism.
- **Integration tests** (`tests/integration/`): Use moto for S3. Spark tests require Java.
- **E2E tests** (`tests/e2e/`): Require Garage S3 via Docker Compose (`just d-e2e`).
- **Contract tests**: Hypothesis property tests for routing invariants in `tests/unit/writer/core/test_routing_contract.py`.
- For behavior changes: add/adjust unit tests first, then integration tests where routing/publishing is affected.

### Test Markers

- `@pytest.mark.spark` — requires PySpark
- `@pytest.mark.dask` — requires Dask extras
- `@pytest.mark.ray` — requires Ray extras
- `@pytest.mark.e2e` — end-to-end (container engine required)

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `just ci` or at minimum `just fix && uv run pyright shardyfusion && uv run pytest -q tests/unit/ -n 2`
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/) prefixes:
   - `feat:` — new feature
   - `fix:` — bug fix
   - `chore:` — maintenance
   - `test:` — test changes
   - `docs:` — documentation
5. Open a PR against `main`

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation including:

- Module dependency graph
- Write and read pipeline descriptions
- Critical invariants (xxhash64 sharding)
- Error hierarchy
- Key abstractions and protocols
