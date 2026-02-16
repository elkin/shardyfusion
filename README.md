# slatedb_spark_sharded

`slatedb_spark_sharded` is a sharded snapshot writer/reader library for SlateDB.

It provides:

- writer-side Spark APIs to build `num_dbs` independent SlateDB shard databases
- manifest + `_CURRENT` publishing protocol (default S3, pluggable interfaces)
- reader-side routing helpers for service-side `get` and `multi_get`

## Runtime Prerequisites

- Reader-only usage does not require Java.
- Writer usage (PySpark) requires a local Java runtime (JRE/JDK) available on `PATH`
  or via `JAVA_HOME`.
- Running Spark-based tests also requires Java.

## Installation

```bash
# Reader-side dependencies only (no Spark)
uv sync --extra read

# Writer-side dependencies (includes Spark)
uv sync --extra writer

# Full install
uv sync --all-extras
```

For development:

```bash
uv sync --all-extras --dev
```

## Minimal Writer Usage

```python
from slatedb_spark_sharded import SlateDbConfig, ValueSpec, write_sharded_slatedb

config = SlateDbConfig(
    num_dbs=8,
    s3_prefix="s3://bucket/prefix",
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)

result = write_sharded_slatedb(df, config)
```

To apply temporary Spark config for the write call:

```python
result = write_sharded_slatedb(
    df,
    config,
    spark_conf_overrides={"spark.speculation": "false"},
)
```

## Minimal Reader Usage

```python
from slatedb_spark_sharded import SlateShardedReader

reader = SlateShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/slatedb-reader",
)

value = reader.get(123)
batch = reader.multi_get([1, 2, 3])
reader.refresh()
reader.close()
```

## Development Workflow

### Lint and style

```bash
uv run ruff check .
uv run ruff format --check .
```

### Type checking

```bash
uv run ty check slatedb_spark_sharded --python-version 3.10 --error-on-warning
uv run pyright slatedb_spark_sharded
```

### Tests

```bash
# Direct pytest run
uv run pytest -q

# Tox quality/stage targets
uv run tox -e lint,format,type
uv run tox -e py311-all-spark35-unit
uv run tox -e py311-read-integration,py311-writer-spark4-integration
```

Parallel tox environments (cap env-level parallelism to avoid OOM):

```bash
uv run tox p -p 2
```

### Build package artifacts

```bash
uv build
```

## Release Process

1. Bump version:

```bash
uv version X.Y.Z
```

2. Commit + merge to `main`.
3. Tag and push:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

4. GitHub Actions `Release` workflow validates and publishes to PyPI via trusted publishing.

## Documentation

MkDocs site is published from `main` to GitHub Pages.

- docs build check runs on pull requests
- docs publish runs on pushes to `main`

Local docs build:

```bash
uv run mkdocs build --strict
```

See `docs/` and `docs/how-it-works.md` for architecture and operational details.
