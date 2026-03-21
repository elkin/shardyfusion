"""Sharded SQL query reader — fan-out SQL across SQLite shard snapshots.

The :class:`ShardedSqlReader` downloads SQLite shard files from S3 and
opens them locally.  Callers can execute SQL against individual shards,
route a query by key, or fan-out across all (or a subset of) shards and
merge results.

Example::

    reader = ShardedSqlReader(
        s3_prefix="s3://bucket/prefix",
        local_root="/tmp/shardyfusion",
    )
    # Query a specific shard
    rows = reader.query_shard(0, "SELECT * FROM data WHERE age > ?", (30,))

    # Fan-out to all shards
    all_rows = reader.query_all("SELECT * FROM data WHERE status = ?", ("active",))

    # Query the shard owning a specific key
    rows = reader.query_key(42, "SELECT * FROM data WHERE user_id = ?", (42,))
"""

from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .credentials import CredentialProvider
from .errors import ReaderStateError
from .logging import get_logger, log_event
from .manifest import ManifestRef, RequiredBuildMeta, RequiredShardMeta
from .manifest_store import ManifestStore, S3ManifestStore
from .routing import SnapshotRouter
from .sqlite_adapter import _DB_FILENAME
from .sqlite_schema import SqliteSchema
from .storage import get_bytes
from .type_defs import S3ConnectionOptions

_logger = get_logger(__name__)

Row = dict[str, Any]
"""A single result row as a column-name → value dict."""


def _row_to_dict(row: sqlite3.Row) -> Row:
    return dict(row)


# ---------------------------------------------------------------------------
# ShardedSqlReader
# ---------------------------------------------------------------------------


class ShardedSqlReader:
    """Fan-out SQL queries across a sharded SQLite snapshot.

    On construction, downloads all non-empty shard SQLite files from S3
    and opens read-only connections.  Queries are then served entirely
    from local disk.
    """

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: ManifestStore | None = None,
        current_name: str = "_CURRENT",
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        max_download_workers: int = 8,
        mmap_size: int = 268435456,
    ) -> None:
        self._s3_prefix = s3_prefix
        self._local_root = local_root
        self._mmap_size = mmap_size
        self._closed = False

        store = manifest_store or S3ManifestStore(
            s3_prefix=s3_prefix,
            current_name=current_name,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
        ref = store.load_current()
        if ref is None:
            raise ReaderStateError("No _CURRENT pointer found")

        parsed = store.load_manifest(ref.ref)
        self._build: RequiredBuildMeta = parsed.required_build
        self._shards: list[RequiredShardMeta] = parsed.shards
        self._manifest_ref: ManifestRef = ref
        self._router = SnapshotRouter(self._build, self._shards)
        self._custom = parsed.custom

        # Extract SQLite schema from manifest custom fields (if present)
        schema_dict = self._custom.get("sqlite_schema")
        self._schema: SqliteSchema | None = (
            SqliteSchema.from_dict(schema_dict) if schema_dict else None
        )

        # Download and open shard connections in parallel
        self._conns: dict[int, sqlite3.Connection] = {}
        non_empty = [s for s in self._router.shards if s.db_url is not None]
        with ThreadPoolExecutor(max_workers=max_download_workers) as pool:
            futures = {pool.submit(self._open_shard, s): s.db_id for s in non_empty}
            for f in as_completed(futures):
                db_id = futures[f]
                self._conns[db_id] = f.result()

        log_event(
            "sqlite_sql_reader_opened",
            logger=_logger,
            s3_prefix=s3_prefix,
            num_shards=len(self._conns),
        )

    def _open_shard(self, shard: RequiredShardMeta) -> sqlite3.Connection:
        local_dir = Path(self._local_root) / f"shard={shard.db_id:05d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        db_path = local_dir / _DB_FILENAME

        if not db_path.exists():
            assert shard.db_url is not None
            s3_key = f"{shard.db_url.rstrip('/')}/{_DB_FILENAME}"
            data = get_bytes(s3_key)
            db_path.write_bytes(data)

        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA mmap_size = {self._mmap_size}")
        conn.execute("PRAGMA cache_size = -8000")
        return conn

    # -- Metadata --

    @property
    def num_shards(self) -> int:
        return self._build.num_dbs

    @property
    def schema(self) -> SqliteSchema | None:
        return self._schema

    @property
    def manifest_ref(self) -> ManifestRef:
        return self._manifest_ref

    @property
    def shard_ids(self) -> list[int]:
        """Return db_ids of non-empty shards with open connections."""
        return sorted(self._conns.keys())

    # -- Single-shard queries --

    def query_shard(
        self,
        shard_id: int,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[Row]:
        """Execute SQL against a specific shard.  Returns dicts."""
        self._check_open()
        conn = self._conns.get(shard_id)
        if conn is None:
            return []  # empty shard
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]

    def query_key(
        self,
        key: int | str | bytes,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[Row]:
        """Route to the shard owning *key*, then execute SQL there."""
        shard_id = self._router.route_one(key)
        return self.query_shard(shard_id, sql, params)

    # -- Fan-out queries --

    def query_all(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        limit: int | None = None,
    ) -> list[Row]:
        """Fan-out SQL to ALL non-empty shards.  Results are concatenated."""
        return self.query_shards(self.shard_ids, sql, params, limit=limit)

    def query_shards(
        self,
        shard_ids: list[int],
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        limit: int | None = None,
    ) -> list[Row]:
        """Fan-out SQL to specific shards.  Results are concatenated."""
        self._check_open()
        results: list[Row] = []
        for sid in shard_ids:
            conn = self._conns.get(sid)
            if conn is None:
                continue
            rows = conn.execute(sql, params).fetchall()
            results.extend(_row_to_dict(r) for r in rows)
            if limit is not None and len(results) >= limit:
                return results[:limit]
        return results

    # -- Concurrent fan-out --

    def query_all_concurrent(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        limit: int | None = None,
        max_workers: int = 4,
    ) -> list[Row]:
        """Fan-out SQL to all shards using a thread pool."""
        self._check_open()
        results: list[Row] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.query_shard, sid, sql, params): sid
                for sid in self.shard_ids
            }
            for f in as_completed(futures):
                results.extend(f.result())
                if limit is not None and len(results) >= limit:
                    break
        return results[:limit] if limit else results

    # -- Pandas integration --

    def query_all_df(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        limit: int | None = None,
    ) -> Any:
        """Fan-out SQL to all shards and return a pandas DataFrame."""
        import pandas as pd  # pyright: ignore[reportMissingImports]

        rows = self.query_all(sql, params, limit=limit)
        return pd.DataFrame(rows)

    def query_shard_df(
        self,
        shard_id: int,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> Any:
        """Execute SQL on one shard and return a pandas DataFrame."""
        import pandas as pd  # pyright: ignore[reportMissingImports]

        rows = self.query_shard(shard_id, sql, params)
        return pd.DataFrame(rows)

    # -- Lifecycle --

    def close(self) -> None:
        if self._closed:
            return
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
        self._closed = True

    def __enter__(self) -> ShardedSqlReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ReaderStateError("ShardedSqlReader is closed")


# ---------------------------------------------------------------------------
# ConcurrentShardedSqlReader
# ---------------------------------------------------------------------------


class ConcurrentShardedSqlReader:
    """Thread-safe SQL reader with a persistent thread pool for fan-out.

    Unlike :class:`ShardedSqlReader` which creates short-lived thread pools
    per query, this reader maintains a long-lived pool.
    """

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        max_workers: int = 4,
        manifest_store: ManifestStore | None = None,
        current_name: str = "_CURRENT",
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        max_download_workers: int = 8,
        mmap_size: int = 268435456,
    ) -> None:
        self._inner = ShardedSqlReader(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=manifest_store,
            current_name=current_name,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
            max_download_workers=max_download_workers,
            mmap_size=mmap_size,
        )
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._closed = False

    @property
    def num_shards(self) -> int:
        return self._inner.num_shards

    @property
    def schema(self) -> SqliteSchema | None:
        return self._inner.schema

    def query_shard(
        self, shard_id: int, sql: str, params: tuple[Any, ...] = ()
    ) -> list[Row]:
        return self._inner.query_shard(shard_id, sql, params)

    def query_key(
        self, key: int | str | bytes, sql: str, params: tuple[Any, ...] = ()
    ) -> list[Row]:
        return self._inner.query_key(key, sql, params)

    def query_all(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        limit: int | None = None,
    ) -> list[Row]:
        results: list[Row] = []
        futures = {
            self._pool.submit(self._inner.query_shard, sid, sql, params): sid
            for sid in self._inner.shard_ids
        }
        for f in as_completed(futures):
            results.extend(f.result())
            if limit is not None and len(results) >= limit:
                break
        return results[:limit] if limit else results

    def close(self) -> None:
        if self._closed:
            return
        self._pool.shutdown(wait=False)
        self._inner.close()
        self._closed = True

    def __enter__(self) -> ConcurrentShardedSqlReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# AsyncShardedSqlReader
# ---------------------------------------------------------------------------


class AsyncShardedSqlReader:
    """Async SQL reader using ``asyncio.to_thread`` for SQLite operations."""

    def __init__(self, inner: ShardedSqlReader) -> None:
        self._inner = inner

    @classmethod
    async def open(
        cls,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: ManifestStore | None = None,
        current_name: str = "_CURRENT",
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        max_download_workers: int = 8,
        mmap_size: int = 268435456,
    ) -> AsyncShardedSqlReader:
        inner = await asyncio.to_thread(
            lambda: ShardedSqlReader(
                s3_prefix=s3_prefix,
                local_root=local_root,
                manifest_store=manifest_store,
                current_name=current_name,
                credential_provider=credential_provider,
                s3_connection_options=s3_connection_options,
                max_download_workers=max_download_workers,
                mmap_size=mmap_size,
            )
        )
        return cls(inner)

    @property
    def num_shards(self) -> int:
        return self._inner.num_shards

    @property
    def schema(self) -> SqliteSchema | None:
        return self._inner.schema

    async def query_shard(
        self, shard_id: int, sql: str, params: tuple[Any, ...] = ()
    ) -> list[Row]:
        return await asyncio.to_thread(self._inner.query_shard, shard_id, sql, params)

    async def query_key(
        self, key: int | str | bytes, sql: str, params: tuple[Any, ...] = ()
    ) -> list[Row]:
        return await asyncio.to_thread(self._inner.query_key, key, sql, params)

    async def query_all(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        limit: int | None = None,
    ) -> list[Row]:
        """Fan-out to all shards using asyncio tasks (each wraps to_thread)."""
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(self._inner.query_shard, sid, sql, params)
            )
            for sid in self._inner.shard_ids
        ]
        results: list[Row] = []
        for coro in asyncio.as_completed(tasks):
            shard_rows = await coro
            results.extend(shard_rows)
            if limit is not None and len(results) >= limit:
                for t in tasks:
                    t.cancel()
                break
        return results[:limit] if limit else results

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)

    async def __aenter__(self) -> AsyncShardedSqlReader:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
