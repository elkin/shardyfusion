"""SQLite shard adapter — write locally, upload once, read from S3.

KV adapter (``SqliteFactory`` / ``SqliteAdapter``): satisfies the
``DbAdapter`` / ``ShardReader`` protocols.  Stores data in a
``WITHOUT ROWID`` ``kv(k BLOB PK, v BLOB)`` table.

Lifecycle: build a local SQLite DB → upload the single ``.db`` file to
S3 on close → reader downloads (or range-reads) the file.

Reader tiers:

* **Download-and-cache** (default): one S3 GET, then all lookups are local.
* **Range-read VFS** (optional, requires ``apsw``): translates SQLite page
  reads into S3 ``Range`` requests with an LRU page cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import types
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .errors import ShardyfusionError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .storage import get_bytes, put_bytes

_logger = get_logger(__name__)

_DB_FILENAME = "shard.db"
_DB_IDENTITY_FILENAME = "shard.identity.json"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SqliteAdapterError(ShardyfusionError):
    """SQLite adapter error (non-retryable)."""

    retryable = False


# ---------------------------------------------------------------------------
# Writer: KV adapter (Layer 1)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteFactory:
    """Picklable factory that builds local SQLite KV shards."""

    page_size: int = 4096
    cache_size_pages: int = -2000  # negative = KiB, so ~8 MB

    def __call__(self, *, db_url: str, local_dir: Path) -> SqliteAdapter:
        return SqliteAdapter(
            db_url=db_url,
            local_dir=local_dir,
            page_size=self.page_size,
            cache_size_pages=self.cache_size_pages,
        )


class SqliteAdapter:
    """KV write-path adapter: builds SQLite DB locally, uploads to S3 on close.

    Satisfies the :class:`~shardyfusion.slatedb_adapter.DbAdapter` protocol.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        page_size: int = 4096,
        cache_size_pages: int = -2000,
    ) -> None:
        self._db_url = db_url
        self._local_dir = local_dir
        self._db_path = local_dir / _DB_FILENAME
        self._uploaded = False
        self._closed = False
        self._checkpointed = False

        local_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        conn.execute(f"PRAGMA page_size = {page_size}")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute(f"PRAGMA cache_size = {cache_size_pages}")
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv "
            "(k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID"
        )
        conn.execute("BEGIN")
        self._conn: sqlite3.Connection | None = conn

        log_event(
            "sqlite_adapter_opened",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        if self._conn is None:
            raise SqliteAdapterError("Adapter already closed")
        if self._checkpointed:
            raise SqliteAdapterError("Cannot write after checkpoint")
        self._conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", pairs)

    def flush(self) -> None:
        pass  # no-op: local file, no WAL

    def checkpoint(self) -> str | None:
        if self._conn is None:
            raise SqliteAdapterError("Adapter already closed")
        if self._checkpointed:
            raise SqliteAdapterError("Adapter already checkpointed")
        self._conn.execute("COMMIT")
        self._conn.execute("PRAGMA optimize")
        self._conn.close()
        self._conn = None
        self._checkpointed = True

        file_hash = hashlib.sha256(self._db_path.read_bytes()).hexdigest()
        log_event(
            "sqlite_adapter_checkpointed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=self._db_url,
            checkpoint_id=file_hash,
        )
        return file_hash

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._conn is not None:
                try:
                    self._conn.execute("COMMIT")
                except sqlite3.OperationalError:
                    pass
                self._conn.close()
                self._conn = None

            if self._db_path.exists() and not self._uploaded:
                s3_key = f"{self._db_url.rstrip('/')}/{_DB_FILENAME}"
                put_bytes(
                    s3_key,
                    self._db_path.read_bytes(),
                    content_type="application/x-sqlite3",
                )
                self._uploaded = True
                log_event(
                    "sqlite_adapter_uploaded",
                    level=logging.DEBUG,
                    logger=_logger,
                    db_url=self._db_url,
                )
        except Exception as exc:
            log_failure(
                "sqlite_adapter_close_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise
        finally:
            self._closed = True
            log_event(
                "sqlite_adapter_closed",
                level=logging.DEBUG,
                logger=_logger,
                db_url=self._db_url,
            )


# ---------------------------------------------------------------------------
# Reader: download-and-cache (Tier 1)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteReaderFactory:
    """Picklable factory for the download-and-cache SQLite shard reader."""

    mmap_size: int = 268435456  # 256 MB

    def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> SqliteShardReader:
        return SqliteShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            mmap_size=self.mmap_size,
        )


class SqliteShardReader:
    """Download-and-cache reader: one S3 GET, then pure local lookups.

    Satisfies the :class:`~shardyfusion.type_defs.ShardReader` protocol.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        mmap_size: int = 268435456,
    ) -> None:
        self._db_url = db_url
        local_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = local_dir / _DB_FILENAME
        self._identity_path = local_dir / _DB_IDENTITY_FILENAME
        self._checkpoint_id = checkpoint_id

        if not self._is_cached_snapshot_current():
            s3_key = f"{db_url.rstrip('/')}/{_DB_FILENAME}"
            data = get_bytes(s3_key)
            self._db_path.write_bytes(data)
            self._write_cached_snapshot_identity()

        conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA mmap_size = {mmap_size}")
        conn.execute("PRAGMA cache_size = -8000")
        self._conn: sqlite3.Connection | None = conn

    def _expected_snapshot_identity(self) -> dict[str, str | None]:
        return {
            "db_url": self._db_url,
            "checkpoint_id": self._checkpoint_id,
        }

    def _is_cached_snapshot_current(self) -> bool:
        if not self._db_path.exists() or not self._identity_path.exists():
            return False

        try:
            cached_identity = json.loads(self._identity_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False

        return cached_identity == self._expected_snapshot_identity()

    def _write_cached_snapshot_identity(self) -> None:
        self._identity_path.write_text(
            json.dumps(self._expected_snapshot_identity(), sort_keys=True)
        )

    # -- ShardReader protocol --

    def get(self, key: bytes) -> bytes | None:
        if self._conn is None:
            raise SqliteAdapterError("Reader already closed")
        row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Reader: async wrapper (download-and-cache)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncSqliteReaderFactory:
    """Async factory for the download-and-cache SQLite shard reader."""

    mmap_size: int = 268435456

    async def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> AsyncSqliteShardReader:
        inner = await asyncio.to_thread(
            SqliteShardReader,
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            mmap_size=self.mmap_size,
        )
        return AsyncSqliteShardReader(inner)


class AsyncSqliteShardReader:
    """Async wrapper around :class:`SqliteShardReader`.

    Delegates blocking ``sqlite3`` calls to a thread executor via
    :func:`asyncio.to_thread`.

    This wrapper does not guard against concurrent ``close()`` and
    ``get()`` calls.  Callers must ensure that ``close()`` is not
    invoked while ``get()`` operations are still in flight.  When used
    via :class:`~shardyfusion.reader.async_reader.AsyncShardedReader`,
    the reader's borrow-count mechanism provides this guarantee
    automatically.
    """

    def __init__(self, inner: SqliteShardReader) -> None:
        self._inner = inner

    async def get(self, key: bytes) -> bytes | None:
        return await asyncio.to_thread(self._inner.get, key)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)


# ---------------------------------------------------------------------------
# Reader: S3 range-read VFS (Tier 2, requires apsw)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteRangeReaderFactory:
    """Picklable factory for the range-read VFS reader.  Requires ``apsw``."""

    page_cache_pages: int = 1024  # ~4 MB at 4 KB/page

    def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> SqliteRangeShardReader:
        return SqliteRangeShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            page_cache_pages=self.page_cache_pages,
        )


class _S3ReadOnlyFile:
    """Read-only virtual file backed by S3 range requests with LRU cache.

    This is the core I/O layer used by the APSW VFS implementation.
    """

    def __init__(
        self,
        *,
        bucket: str,
        key: str,
        page_cache_pages: int = 1024,
    ) -> None:
        from .storage import create_s3_client

        self._client = create_s3_client()
        self._bucket = bucket
        self._key = key
        self._page_cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self._page_cache_pages = page_cache_pages

        head = self._client.head_object(Bucket=bucket, Key=key)
        self._size: int = head["ContentLength"]

    @property
    def size(self) -> int:
        return self._size

    def read(self, offset: int, amount: int) -> bytes:
        cache_key = (offset, amount)
        cached = self._page_cache.get(cache_key)
        if cached is not None:
            self._page_cache.move_to_end(cache_key)
            return cached

        end = min(offset + amount - 1, self._size - 1)
        resp = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=f"bytes={offset}-{end}",
        )
        data = resp["Body"].read()

        # LRU eviction
        if len(self._page_cache) >= self._page_cache_pages:
            self._page_cache.popitem(last=False)
        self._page_cache[cache_key] = data

        return data


class SqliteRangeShardReader:
    """Range-read reader: fetches only the SQLite pages needed via S3 Range requests.

    Point lookups on a shard with 1M keys require ~3-4 page reads (B-tree
    depth).  An LRU page cache reduces repeated page fetches to zero for
    warm paths.

    Requires ``apsw`` (``pip install apsw``).  Satisfies the
    :class:`~shardyfusion.type_defs.ShardReader` protocol.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        page_cache_pages: int = 1024,
    ) -> None:
        try:
            import apsw  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise SqliteAdapterError(
                "apsw is required for the range-read VFS reader. "
                "Install via: pip install apsw"
            ) from exc

        from .storage import parse_s3_url

        self._db_url = db_url
        s3_key_full = f"{db_url.rstrip('/')}/{_DB_FILENAME}"
        bucket, key = parse_s3_url(s3_key_full)
        self._s3_file = _S3ReadOnlyFile(
            bucket=bucket, key=key, page_cache_pages=page_cache_pages
        )

        # Register a unique VFS name for this reader instance
        import uuid

        vfs_name = f"s3range_{uuid.uuid4().hex}"
        self._vfs = _create_apsw_vfs(vfs_name, self._s3_file)
        self._conn = apsw.Connection(
            f"file:{_DB_FILENAME}?mode=ro",
            flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
            vfs=vfs_name,
        )

    def get(self, key: bytes) -> bytes | None:
        conn = self._conn
        if conn is None:
            raise SqliteAdapterError("Reader is closed")
        cursor = conn.cursor()
        cursor.execute("SELECT v FROM kv WHERE k = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        if hasattr(self, "_conn") and self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
        if hasattr(self, "_vfs") and self._vfs is not None:
            self._vfs.unregister()
            self._vfs = None


def _create_apsw_vfs(vfs_name: str, s3_file: _S3ReadOnlyFile) -> Any:
    """Create and register an APSW VFS backed by S3 range reads.

    Defines VFS and VFSFile subclasses at call time because ``apsw`` is
    an optional dependency that may not be available at module load.
    Returns the registered VFS instance (caller must keep it alive while
    the connection is open, and call ``unregister()`` on close).
    """
    import apsw  # pyright: ignore[reportMissingImports]

    class S3RangeVFS(apsw.VFS):
        def __init__(self) -> None:
            super().__init__(vfs_name, "")

        def xOpen(
            self, name: str | apsw.URIFilename | None, flags: list[int]
        ) -> S3RangeVFSFile:
            return S3RangeVFSFile("", flags)

    class S3RangeVFSFile(apsw.VFSFile):
        def __init__(self, _name: str, _flags: list[int]) -> None:
            # Do NOT call super().__init__() — there is no underlying
            # base file to open; all I/O goes through S3 range reads.
            pass

        def xRead(self, amount: int, offset: int) -> bytes:
            data = s3_file.read(offset, amount)
            # SQLite expects exactly ``amount`` bytes; pad with zeros
            # if the read is short (e.g. at end of file).
            if len(data) < amount:
                data += b"\x00" * (amount - len(data))
            return data

        def xFileSize(self) -> int:
            return s3_file.size

        def xClose(self) -> None:
            pass

        def xLock(self, level: int) -> None:
            pass

        def xUnlock(self, level: int) -> None:
            pass

        def xCheckReservedLock(self) -> bool:
            return False

        def xFileControl(self, op: int, ptr: int) -> bool:
            return False

        def xSectorSize(self) -> int:
            return 4096

        def xDeviceCharacteristics(self) -> int:
            return 0

    return S3RangeVFS()


# ---------------------------------------------------------------------------
# Reader: async range-read wrapper
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncSqliteRangeReaderFactory:
    """Async factory for the range-read VFS reader."""

    page_cache_pages: int = 1024

    async def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> AsyncSqliteRangeShardReader:
        inner = await asyncio.to_thread(
            SqliteRangeShardReader,
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            page_cache_pages=self.page_cache_pages,
        )
        return AsyncSqliteRangeShardReader(inner)


class AsyncSqliteRangeShardReader:
    """Async wrapper around :class:`SqliteRangeShardReader`.

    This wrapper does not guard against concurrent ``close()`` and
    ``get()`` calls.  Callers must ensure that ``close()`` is not
    invoked while ``get()`` operations are still in flight.
    """

    def __init__(self, inner: SqliteRangeShardReader) -> None:
        self._inner = inner

    async def get(self, key: bytes) -> bytes | None:
        return await asyncio.to_thread(self._inner.get, key)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)
