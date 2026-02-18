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

Containerized local development run (Podman):

```bash
podman build -f docker/ci.Dockerfile -t slatedb-spark-sharded-ci .
podman run --rm -v "$PWD:/workspace" -w /workspace slatedb-spark-sharded-ci \
  /bin/bash -lc "uv sync --all-extras --dev && uv run tox -m quality && uv run tox -m unit && uv run tox -m integration"
```

The image includes both Python 3.11 and Python 3.10 so tox `py311-*` and
`py310-*` environments execute instead of being skipped.

Short container prefix via `just`:

```bash
just docker-build
just d uv run tox -m quality
just d uv run tox -m unit
just d uv run tox -m integration
```

`just d ...` runs the same command shape as local usage, but inside the container.
It uses container-only uv state and a container-only project venv path
(`UV_PROJECT_ENVIRONMENT=/opt/slatedb-venv`), so it does not reuse host `.venv`.

Container runtime defaults to `podman`; switch to Docker with:

```bash
CONTAINER_ENGINE=docker just docker-build
CONTAINER_ENGINE=docker just d uv run tox -m quality
```

Dev Container (VS Code):

1. Install the VS Code `Dev Containers` extension.
2. Open this repository in VS Code.
3. Run `Dev Containers: Reopen in Container`.

The Dev Container reuses `docker/ci.Dockerfile` and runs
`uv sync --all-extras --dev` automatically after container creation.

If you use Podman as the backend, expose a Docker-compatible socket
(`podman system service`) and point VS Code Dev Containers to it.

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
