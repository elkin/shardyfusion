# Getting Started

## Requirements

- Python 3.11+
- Java 17+ (for Spark integration tests)
- `uv` installed

## Install

```bash
# Reader-side only (default SlateDB backend)
uv sync --extra read

# Async reader (default SlateDB backend, includes aiobotocore)
uv sync --extra read-async

# Reader-side only with the SQLite backend
uv sync --extra read-sqlite

# Reader-side with SQLite range reads (APSW-backed)
uv sync --extra read-sqlite-range

# Async SQLite reader wrappers (download-and-cache + aiobotocore)
uv sync --extra sqlite-async

# Async SQLite range reads (combine async wrappers + APSW)
uv sync --extra sqlite-async --extra read-sqlite-range

# Spark writer (default SlateDB backend, requires Java)
uv sync --extra writer-spark

# Spark writer with the SQLite backend
uv sync --extra writer-spark-sqlite

# Python writer (default SlateDB backend, no Spark/Java required)
uv sync --extra writer-python

# Python writer with the SQLite backend
uv sync --extra writer-python-sqlite

# Dask writer (default SlateDB backend, no Spark/Java required)
uv sync --extra writer-dask

# Dask writer with the SQLite backend
uv sync --extra writer-dask-sqlite

# Ray writer (default SlateDB backend, no Spark/Java required)
uv sync --extra writer-ray

# Ray writer with the SQLite backend
uv sync --extra writer-ray-sqlite

# CLI only (default SlateDB reader stack, no Spark/Java required)
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
uv run pyright shardyfusion

# Run tests directly
uv run pytest -q

# Tox matrix
tox -e py311-pythonwriter-slatedb-unit
tox -e py311-read-slatedb-integration
tox -m unit
tox -m integration
```

## Build Package

```bash
uv build
```
