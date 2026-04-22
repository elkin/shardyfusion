# Adding an adapter

This page walks through adding a new **KV adapter** end-to-end, using a hypothetical backend `foodb` as the example. The same pattern applies to vector adapters (substitute `vector/adapters/`) and composite adapters.

Before starting, read:

- [`architecture/adapters.md`](../architecture/adapters.md) — the protocol surface.
- [`architecture/writer-core.md`](../architecture/writer-core.md) — what the writer expects.
- [`extras-and-dependencies.md`](extras-and-dependencies.md) — gating the new dependency.

## Anatomy

A KV adapter has four pieces:

1. **`DbAdapterFactory`** — constructs per-shard adapters.
2. **`DbAdapter`** — write-side protocol (`write_batch`, `flush`, `checkpoint`, `close`, context-manager).
3. **`ShardReaderFactory`** — constructs per-shard readers.
4. **`ShardReader`** — read-side `get(key) -> bytes | None`.

Sync and async readers are distinct factories; you can ship sync first and add async later.

## 1. Decide the read strategy

Two patterns exist for non-native backends:

- **Download-and-cache**: fetch the entire shard file to local disk on first access, open with the native driver. Simple; works for any file-based backend. See `SqliteShardReader` (`sqlite_adapter.py:250`).
- **Range-read VFS**: implement a custom VFS over HTTP range requests. Complex; only worthwhile for large shards with sparse access. See `SqliteRangeShardReader` (`sqlite_adapter.py:456`).

Default to download-and-cache.

## 2. Add the dependency

Per [`extras-and-dependencies.md`](extras-and-dependencies.md):

```toml
# pyproject.toml
[project.optional-dependencies]
foodb = ["foodb-py>=1.0", "boto3>=1.28"]

[dependency-groups]
backend-foodb = ["foodb-py>=1.0", "boto3>=1.28"]
```

## 3. Implement the writer factory and adapter

Create `shardyfusion/foodb_adapter.py`:

```python
from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

from shardyfusion.errors import DbAdapterError


class FooDbAdapterError(DbAdapterError):
    """foodb-specific failures."""


def _import_foodb() -> Any:
    try:
        import foodb_py
    except ImportError as exc:
        raise ImportError(
            "foodb adapter requires `pip install shardyfusion[foodb]`"
        ) from exc
    return foodb_py


class FooDbFactory:
    """Factory for FooDbAdapter instances."""

    def __call__(self, *, db_url: str, local_dir: str) -> "FooDbAdapter":
        return FooDbAdapter(db_url=db_url, local_dir=local_dir)


class FooDbAdapter:
    """Per-shard FooDB writer adapter."""

    def __init__(self, *, db_url: str, local_dir: str) -> None:
        foodb = _import_foodb()
        self._db_url = db_url
        self._local_dir = Path(local_dir)
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._db = foodb.open(self._local_dir / "shard.foodb")

    def __enter__(self) -> "FooDbAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with suppress(Exception):
            self.close()

    def write_batch(self, pairs: list[tuple[bytes, bytes]]) -> None:
        try:
            self._db.put_many(pairs)
        except Exception as exc:
            raise FooDbAdapterError(f"write_batch failed: {exc}") from exc

    def flush(self) -> None:
        self._db.flush()

    def checkpoint(self) -> str | None:
        # Persist locally, upload to db_url, return the canonical URL.
        self._db.close()
        _upload_dir_to_s3(self._local_dir, self._db_url)
        return self._db_url

    def close(self) -> None:
        with suppress(Exception):
            self._db.close()
```

Notes:

- `__call__` signature is **keyword-only**: `(*, db_url, local_dir)`.
- `checkpoint()` returns the canonical URL stored in the manifest (used for winner-selection tiebreak).
- All backend-specific exceptions are wrapped in `FooDbAdapterError` (subclass of `DbAdapterError`).
- The lazy import lives in a helper, not at module top.

## 4. Implement the reader factory and reader

```python
class FooDbShardReader:
    def __init__(self, *, db_url: str, local_dir: str, ...) -> None:
        # Download db_url to local_dir on first construction
        ...

    def get(self, key: bytes) -> bytes | None:
        return self._db.get(key)

    def close(self) -> None:
        ...


class FooDbReaderFactory:
    def __call__(self, *, db_url: str, local_dir: str, **kwargs) -> FooDbShardReader:
        return FooDbShardReader(db_url=db_url, local_dir=local_dir, **kwargs)
```

`ShardedReader(s3_prefix=..., reader_factory=FooDbReaderFactory(), ...)` then routes shard reads through your adapter.

## 5. Wire metrics (optional but recommended)

If your backend can fail in interesting ways (retries, throttling), accept a `MetricsCollector | None` and emit events from the relevant `MetricEvent` (or extend the enum if a truly new event type is warranted — extending requires an ADR).

## 6. Tests

| Layer | What to test |
|---|---|
| Unit | Round-trip in-memory or with `tmp_path`. Mirror `tests/unit/backend/sqlite/`. |
| Unit | Routing-contract property test still passes (it doesn't touch your adapter directly, but confirms invariants). |
| Integration | Writer + reader against moto S3. Mirror `tests/integration/backend/sqlite/`. |
| E2E | Add to `tests/e2e/writer/` if the backend uploads to S3. |

Use the shared scenarios from `tests/helpers/s3_test_scenarios.py` so all backends exercise the same edge cases.

## 7. Tox

In `tox.ini`:

```ini
[testenv]
dependency_groups =
    foodb: backend-foodb
```

Add envs to `env_list` and the `unit` / `integration` labels:

```
py{311,312,313}-foodb-unit
py{311,312,313}-foodb-integration
```

Then `just ci-matrix` regenerates `.github/ci-matrix.json`.

## 8. Documentation

Per [`adding-a-use-case.md`](adding-a-use-case.md):

1. Add `docs/use-cases/kv-storage/build/python.md` (or the appropriate writer leaf page if the adapter supports an existing writer flavor).
2. Add a row to the use-case map in `docs/index.md`.
3. Update [`architecture/adapters.md`](../architecture/adapters.md) with a row in the built-in adapters table.
4. Update [`architecture/optional-imports.md`](../architecture/optional-imports.md) extras index.

## 9. Validate

```bash
just fix
just quality
just unit
just integration
uv run python .claude/skills/validate-docs/scripts/check_docs.py
just d-e2e
```

If green, open the PR.

## Public-export decision

By default, **do not** add your factory or adapter to `shardyfusion/__init__.py`'s `__all__`. SQLite and sqlite-vec adapters are imported from their own modules (`from shardyfusion.sqlite_adapter import SqliteFactory`). Only the SlateDB factory is in the top-level public surface, because it's the default backend.

If your backend is meant to be the new default, that's an ADR-level change.

## See also

- [`architecture/adapters.md`](../architecture/adapters.md) — the protocol.
- [`architecture/writer-core.md`](../architecture/writer-core.md) — how the writer drives the adapter.
- [`extras-and-dependencies.md`](extras-and-dependencies.md) — gating the import.
- [`adding-a-use-case.md`](adding-a-use-case.md) — documentation requirements.
