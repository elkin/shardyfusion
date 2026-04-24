# Optional imports

shardyfusion is one package with many feature dimensions: four writer engines (Python, Spark, Dask, Ray), three KV backends (SlateDB, SQLite-download, SQLite-range-read), two vector backends (LanceDB, sqlite-vec), two metrics backends (Prometheus, OTel), two manifest stores (S3, Postgres), and a CLI. Forcing every install to pull all of these would be untenable.

The **optional-imports pattern** keeps the package importable with no extras installed, and gates feature availability on per-extra dependency groups.

## The pattern

1. Define an extra in `pyproject.toml` under `[project.optional-dependencies]`.
2. The module that depends on it is **imported lazily** — never at the top level of `shardyfusion/__init__.py`.
3. The lazy import is wrapped in a helper that raises a clear "install with `pip install shardyfusion[<extra>]`" message if the dependency is missing.

### Example: CEL

```python
# shardyfusion/_writer_core.py:113
def _get_cel_imports() -> tuple[Any, ...]:
    from shardyfusion.cel import compile_cel, route_cel_batch  # local import
    return compile_cel, route_cel_batch
```

`shardyfusion.cel` itself does:

```python
# shardyfusion/cel.py
def _import_cel() -> Any:
    try:
        from cel_expr_python.cel import NewEnv, Type
    except ImportError as e:
        raise ImportError("CEL routing requires `pip install shardyfusion[cel]`") from e
    return NewEnv, Type
```

The CEL package is [`cel-expr-python`](https://pypi.org/project/cel-expr-python/) — a fast Rust-backed CEL implementation, not the older pure-Python `celpy`.

### Example: vector adapters

`shardyfusion/vector/adapters/lancedb_adapter.py` imports `lancedb` only inside the constructor of the LanceDB factory. If `lancedb` is missing, the user gets `ImportError: ... pip install shardyfusion[vector]` — but the rest of shardyfusion (KV writers, readers) is untouched.

### Example: `UnifiedShardedReader`

`shardyfusion/__init__.py` exposes `UnifiedShardedReader` via `__getattr__`:

```python
def __getattr__(name):
    if name == "UnifiedShardedReader":
        from shardyfusion.reader.unified_reader import UnifiedShardedReader
        return UnifiedShardedReader
    raise AttributeError(name)
```

Importing `shardyfusion` does not pull `unified_reader`, which would in turn pull vector dependencies.

## Extras index

| Extra | What it enables | Notes |
|---|---|---|
| `slatedb` | SlateDB driver | Base building block; pulled in by reader/writer extras. |
| `sqlite` | SQLite driver | Base building block. |
| `sqlite-range` | APSW + range-read VFS | Base building block for range-read SQLite. |
| `read` | SlateDB sync reader | Default sync reader. |
| `read-async` | SlateDB async reader (aiobotocore) | Async reader. |
| `read-sqlite` | SQLite download-and-cache reader | Sync. |
| `read-sqlite-range` | SQLite range-read reader (APSW) | Sync. |
| `sqlite-async` | Async SQLite readers (download + range) | Async. |
| `writer-spark` | Spark writer (SlateDB) | Requires Java. |
| `writer-spark-sqlite` | Spark writer (SQLite) | Requires Java. |
| `writer-python` | Python writer (SlateDB) | Pure Python. |
| `writer-python-sqlite` | Python writer (SQLite) | Pure Python. |
| `writer-dask` | Dask writer (SlateDB) | |
| `writer-dask-sqlite` | Dask writer (SQLite) | |
| `writer-ray` | Ray writer (SlateDB) | |
| `writer-ray-sqlite` | Ray writer (SQLite) | |
| `cli` | `click>=8.0` | PyYAML is a base dep, not part of this extra. |
| `cel` | `cel-expr-python` | CEL routing. |
| `metrics-prometheus` | `prometheus_client` | Prometheus metrics backend. |
| `metrics-otel` | `opentelemetry` SDK | OTel metrics backend. |
| `vector-lancedb` | LanceDB vector backend | |
| `vector` | Alias for `vector-lancedb` | |
| `vector-sqlite` | sqlite-vec unified KV+vector | |
| `unified-vector` | Composite KV+vector wiring (LanceDB) | For `UnifiedShardedReader`. |
| `unified-vector-sqlite` | Composite KV+vector wiring (sqlite-vec) | |
| `all` | Convenience bundle | Does **not** include vector extras. |
| `test` | Test runner deps | Dev. |
| `quality` | Lint / typecheck deps | Dev. |
| `docs` | MkDocs + plugins | Dev. |

For the canonical list, see `pyproject.toml`. The validate-docs skill (`.claude/skills/validate-docs/`) cross-checks every extra documented in docs against the canonical list.

## Why not "just pip install everything"

- **Conflicting transitive deps**: LanceDB and pyspark have incompatible Arrow versions in some combinations.
- **Container image size**: minimal reader installs are ~50MB; pulling vector + Spark would push past 1GB.
- **Python version coverage**: shardyfusion targets Python 3.11–3.13 (`requires-python = ">=3.11,<3.14"`). Some optional dependencies have narrower support windows; gating them keeps the base install broadly compatible.

## Contributor rules

When adding a new optional dependency:

1. Add it to `[project.optional-dependencies]` under a meaningful extra.
2. Import it lazily — never at module top-level for any module imported by `shardyfusion/__init__.py`.
3. Wrap the import in a helper that raises `ImportError` with a `pip install shardyfusion[<extra>]` message.
4. Add it to the extras index in this page.
5. Add a use-case page that exercises the new extra.
6. Run `validate-docs`; it will refuse to pass if extras are out of sync.

## See also

- [`contributing/extras-and-dependencies.md`](../contributing/extras-and-dependencies.md) — operational guide.
- [`contributing/adding-an-adapter.md`](../contributing/adding-an-adapter.md) — worked example.
