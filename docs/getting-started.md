# Getting Started

## Requirements

- Python 3.10+
- Java 17+ (for Spark integration tests)
- `uv` installed

## Install

```bash
# Reader-side only
uv sync --extra read

# Spark writer (includes PySpark, requires Java)
uv sync --extra writer-spark

# Python writer (no Spark/Java required)
uv sync --extra writer-python

# Dask writer (no Spark/Java required)
uv sync --extra writer-dask

# CLI only (no Spark/Java required)
uv sync --extra cli

# Full install
uv sync --all-extras
```

## Development Setup

```bash
uv sync --all-extras --dev
```

## Common Commands

```bash
# Lint and style
uv run ruff check .
uv run ruff format --check .

# Type check
uv run pyright slatedb_spark_sharded

# Run tests directly
uv run pytest -q

# Tox matrix
tox -e py311-all-spark35-unit
tox -e py311-read-integration
```

## Build Package

```bash
uv build
```
