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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, runtime_checkable

from ._sqlite_vfs import S3ReadOnlyFile, S3VfsError, create_apsw_vfs
from .credentials import CredentialProvider, S3Credentials
from .errors import ShardyfusionError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .storage import ObstoreBackend, create_s3_store, parse_s3_url
from .type_defs import Manifest, S3ConnectionOptions

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
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(self, *, db_url: str, local_dir: Path) -> SqliteAdapter:
        return SqliteAdapter(
            db_url=db_url,
            local_dir=local_dir,
            page_size=self.page_size,
            cache_size_pages=self.cache_size_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
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
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        page_size = int(page_size)
        cache_size_pages = int(cache_size_pages)

        self._db_url = db_url
        self._local_dir = local_dir
        self._db_path = local_dir / _DB_FILENAME
        self._uploaded = False
        self._closed = False
        self._checkpointed = False
        self._db_bytes = 0
        self._s3_conn_opts = s3_connection_options
        self._s3_creds: S3Credentials | None = (
            credential_provider.resolve() if credential_provider else None
        )

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

        with open(self._db_path, "rb") as f:
            file_hash = hashlib.file_digest(f, "sha256").hexdigest()
        self._db_bytes = self._db_path.stat().st_size
        log_event(
            "sqlite_adapter_checkpointed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=self._db_url,
            checkpoint_id=file_hash,
        )
        return file_hash

    def db_bytes(self) -> int:
        return self._db_bytes

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
                bucket, _ = parse_s3_url(s3_key)
                store = create_s3_store(
                    bucket=bucket,
                    credentials=self._s3_creds,
                    connection_options=self._s3_conn_opts,
                )
                backend = ObstoreBackend(store)
                backend.put(
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
        else:
            self._closed = True
        finally:
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
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> SqliteShardReader:
        del manifest  # unused in concrete factory
        return SqliteShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            mmap_size=self.mmap_size,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
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
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._db_url = db_url
        local_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = local_dir / _DB_FILENAME
        self._identity_path = local_dir / _DB_IDENTITY_FILENAME
        self._checkpoint_id = checkpoint_id

        if not self._is_cached_snapshot_current():
            s3_key = f"{db_url.rstrip('/')}/{_DB_FILENAME}"
            bucket, _ = parse_s3_url(s3_key)
            creds = credential_provider.resolve() if credential_provider else None
            store = create_s3_store(
                bucket=bucket,
                credentials=creds,
                connection_options=s3_connection_options,
            )
            backend = ObstoreBackend(store)
            data = backend.get(s3_key)
            self._db_path.write_bytes(data)
            self._write_cached_snapshot_identity()

        mmap_size = int(mmap_size)
        conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
        )
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
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncSqliteShardReader:
        del manifest  # unused in concrete factory
        inner = await asyncio.to_thread(
            SqliteShardReader,
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            mmap_size=self.mmap_size,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
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

    page_cache_pages: int = 1024  # ~4 MB at 4 KB/page; 0 disables caching
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> SqliteRangeShardReader:
        del manifest  # unused in concrete factory
        return SqliteRangeShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            page_cache_pages=self.page_cache_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )


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
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        try:
            import apsw  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise SqliteAdapterError(
                "apsw and obstore are required for the range-read VFS reader. "
                "Install via: pip install 'shardyfusion[sqlite-range]' "
                "or 'shardyfusion[sqlite-adaptive]' for the adaptive reader."
            ) from exc

        from .storage import parse_s3_url

        self._db_url = db_url
        self._conn: Any | None = None
        self._vfs: Any | None = None

        s3_key_full = f"{db_url.rstrip('/')}/{_DB_FILENAME}"
        bucket, key = parse_s3_url(s3_key_full)
        creds = credential_provider.resolve() if credential_provider else None

        try:
            self._s3_file = S3ReadOnlyFile(
                bucket=bucket,
                key=key,
                page_cache_pages=page_cache_pages,
                s3_connection_options=s3_connection_options,
                s3_credentials=creds,
            )
        except S3VfsError as exc:
            raise SqliteAdapterError(str(exc)) from exc

        # Register a unique VFS name for this reader instance
        import uuid

        vfs_name = f"s3range_{uuid.uuid4().hex}"
        self._vfs = create_apsw_vfs(vfs_name, self._s3_file)
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
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._vfs is not None:
            self._vfs.unregister()
            self._vfs = None


# ---------------------------------------------------------------------------
# Reader: async range-read wrapper
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncSqliteRangeReaderFactory:
    """Async factory for the range-read VFS reader."""

    page_cache_pages: int = 1024  # 0 disables caching
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncSqliteRangeShardReader:
        del manifest  # unused in concrete factory
        inner = await asyncio.to_thread(
            SqliteRangeShardReader,
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            page_cache_pages=self.page_cache_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
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


# ---------------------------------------------------------------------------
# Access-mode policy + factory selectors
# ---------------------------------------------------------------------------


SqliteAccessMode = Literal["download", "range", "auto"]
"""User-facing SQLite shard access mode.

"download" downloads each shard file once and serves all lookups locally
(cheap RAM/disk, expensive cold start).  "range" opens shards via the
APSW range-read VFS, fetching only the SQLite pages needed (cheap cold
start, more S3 GETs).  "auto" picks one of the two per snapshot based on
the published ``db_bytes`` distribution; see :func:`decide_access_mode`.
"""


# Default thresholds for ``auto`` resolution (used by readers when no
# explicit value is provided).
DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES = 16 * 1024 * 1024  # 16 MiB
DEFAULT_AUTO_TOTAL_BUDGET_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def decide_access_mode(
    *,
    db_bytes_per_shard: Sequence[int],
    per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
    total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
) -> Literal["download", "range"]:
    """Decide between ``download`` and ``range`` for a snapshot.

    Returns ``"range"`` when *any* shard's size is at or above
    ``per_shard_threshold``, or when the *cumulative* size is at or above
    ``total_budget``; otherwise ``"download"``. Both comparisons use ``>=``
    (a shard exactly equal to the threshold tips the decision toward
    ``range``).

    The two thresholds protect distinct cost dimensions:

    * ``per_shard_threshold`` — guards against a single oversized shard
      blowing local disk on cold start.
    * ``total_budget`` — guards against many small shards summing to a huge
      working set when the reader holds them all simultaneously.

    Args:
        db_bytes_per_shard: Sequence of shard sizes in bytes (typically from
            ``RequiredShardMeta.db_bytes`` for every shard in the manifest).
            Empty input → ``"download"`` (no data to motivate ranged reads).
        per_shard_threshold: Per-shard size at or above which we switch to
            range-read. Defaults to 16 MiB.
        total_budget: Cumulative size at or above which we switch to
            range-read. Defaults to 2 GiB.
    """

    if not db_bytes_per_shard:
        return "download"
    if max(db_bytes_per_shard) >= per_shard_threshold:
        return "range"
    if sum(db_bytes_per_shard) >= total_budget:
        return "range"
    return "download"


@runtime_checkable
class SqliteAccessPolicy(Protocol):
    """Library-only hook for advanced ``auto`` overrides.

    Implementations receive the snapshot's per-shard byte sizes and return
    ``"download"`` or ``"range"``.  Used by reader state builders when the
    user supplies a custom policy instead of the default thresholds.
    """

    def decide(
        self, db_bytes_per_shard: Sequence[int]
    ) -> Literal["download", "range"]: ...


@dataclass(slots=True)
class _ThresholdPolicy:
    """Default :class:`SqliteAccessPolicy` backed by :func:`decide_access_mode`."""

    per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES
    total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES

    def decide(self, db_bytes_per_shard: Sequence[int]) -> Literal["download", "range"]:
        return decide_access_mode(
            db_bytes_per_shard=db_bytes_per_shard,
            per_shard_threshold=self.per_shard_threshold,
            total_budget=self.total_budget,
        )


def make_threshold_policy(
    *,
    per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
    total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
) -> SqliteAccessPolicy:
    """Construct a default threshold-based :class:`SqliteAccessPolicy`."""

    return _ThresholdPolicy(
        per_shard_threshold=per_shard_threshold,
        total_budget=total_budget,
    )


def make_sqlite_reader_factory(
    *,
    mode: Literal["download", "range"],
    mmap_size: int = 268435456,
    page_cache_pages: int = 1024,
    s3_connection_options: S3ConnectionOptions | None = None,
    credential_provider: CredentialProvider | None = None,
) -> SqliteReaderFactory | SqliteRangeReaderFactory:
    """Build a sync SQLite reader factory for a *concrete* access mode.

    ``mode`` must be ``"download"`` or ``"range"``; ``"auto"`` is resolved
    by the reader state builder before it calls this helper.
    """

    if mode == "download":
        return SqliteReaderFactory(
            mmap_size=mmap_size,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    if mode == "range":
        return SqliteRangeReaderFactory(
            page_cache_pages=page_cache_pages,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    raise SqliteAdapterError(
        f"make_sqlite_reader_factory: unsupported mode {mode!r}; "
        "expected 'download' or 'range' (resolve 'auto' before calling)"
    )


def make_async_sqlite_reader_factory(
    *,
    mode: Literal["download", "range"],
    mmap_size: int = 268435456,
    page_cache_pages: int = 1024,
    s3_connection_options: S3ConnectionOptions | None = None,
    credential_provider: CredentialProvider | None = None,
) -> AsyncSqliteReaderFactory | AsyncSqliteRangeReaderFactory:
    """Build an async SQLite reader factory for a *concrete* access mode."""

    if mode == "download":
        return AsyncSqliteReaderFactory(
            mmap_size=mmap_size,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    if mode == "range":
        return AsyncSqliteRangeReaderFactory(
            page_cache_pages=page_cache_pages,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    raise SqliteAdapterError(
        f"make_async_sqlite_reader_factory: unsupported mode {mode!r}; "
        "expected 'download' or 'range' (resolve 'auto' before calling)"
    )


# ---------------------------------------------------------------------------
# Adaptive (auto) factories
# ---------------------------------------------------------------------------


class AdaptiveSqliteReaderFactory:
    """Sync SQLite reader factory that auto-selects ``download`` vs ``range``.

    On first invocation for a given snapshot (keyed by
    ``manifest.required_build.run_id``) the factory inspects every shard's
    ``db_bytes`` and asks the configured :class:`SqliteAccessPolicy` for a
    decision.  The resulting concrete factory
    (:class:`SqliteReaderFactory` or :class:`SqliteRangeReaderFactory`) is
    cached and reused for all subsequent shards in the same snapshot.

    The cache is a single slot: when a new ``run_id`` arrives the previous
    factory is replaced.  This matches the reader lifecycle —
    ``ShardedReader.refresh()`` rebuilds all shard readers, so any retired
    sub-factory is dropped naturally.

    Thread-safety
    -------------
    The single-slot cache (``_cached_run_id`` / ``_cached_factory``) is **not**
    internally synchronised.  Concurrent calls into ``_resolve_factory`` from
    multiple threads against different snapshots could race and leave the
    cache in a momentarily inconsistent state (no corruption, but a redundant
    factory build).

    In practice this is safe because every supported caller serialises
    refresh-time state construction:

    * :class:`ConcurrentShardedReader` holds ``_refresh_lock`` while building
      a new ``_ReaderState`` (which is when ``__call__`` invocations happen).
    * :class:`AsyncShardedReader` serialises ``_build_state`` via an
      ``asyncio.Lock``.

    Callers wiring ``AdaptiveSqliteReaderFactory`` into custom reader plumbing
    must preserve this single-writer-during-refresh invariant.
    """

    __slots__ = (
        "_policy",
        "_mmap_size",
        "_page_cache_pages",
        "_s3_connection_options",
        "_credential_provider",
        "_cached_run_id",
        "_cached_factory",
    )

    def __init__(
        self,
        *,
        policy: SqliteAccessPolicy | None = None,
        per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
        total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
        mmap_size: int = 268435456,
        page_cache_pages: int = 1024,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._policy: SqliteAccessPolicy = policy or _ThresholdPolicy(
            per_shard_threshold=per_shard_threshold,
            total_budget=total_budget,
        )
        self._mmap_size = mmap_size
        self._page_cache_pages = page_cache_pages
        self._s3_connection_options = s3_connection_options
        self._credential_provider = credential_provider
        self._cached_run_id: str | None = None
        self._cached_factory: SqliteReaderFactory | SqliteRangeReaderFactory | None = (
            None
        )

    def _resolve_factory(
        self, manifest: Manifest
    ) -> SqliteReaderFactory | SqliteRangeReaderFactory:
        run_id = manifest.required_build.run_id
        if run_id == self._cached_run_id and self._cached_factory is not None:
            return self._cached_factory
        sizes = [shard.db_bytes for shard in manifest.shards]
        mode = self._policy.decide(sizes)
        factory = make_sqlite_reader_factory(
            mode=mode,
            mmap_size=self._mmap_size,
            page_cache_pages=self._page_cache_pages,
            s3_connection_options=self._s3_connection_options,
            credential_provider=self._credential_provider,
        )
        self._cached_run_id = run_id
        self._cached_factory = factory
        return factory

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> SqliteShardReader | SqliteRangeShardReader:
        factory = self._resolve_factory(manifest)
        return factory(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
        )


class AsyncAdaptiveSqliteReaderFactory:
    """Async counterpart of :class:`AdaptiveSqliteReaderFactory`.

    Shares the same single-slot cache contract: callers must serialise
    snapshot-state construction (``AsyncShardedReader`` does so via its
    refresh ``asyncio.Lock``).
    """

    __slots__ = (
        "_policy",
        "_mmap_size",
        "_page_cache_pages",
        "_s3_connection_options",
        "_credential_provider",
        "_cached_run_id",
        "_cached_factory",
    )

    def __init__(
        self,
        *,
        policy: SqliteAccessPolicy | None = None,
        per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
        total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
        mmap_size: int = 268435456,
        page_cache_pages: int = 1024,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._policy: SqliteAccessPolicy = policy or _ThresholdPolicy(
            per_shard_threshold=per_shard_threshold,
            total_budget=total_budget,
        )
        self._mmap_size = mmap_size
        self._page_cache_pages = page_cache_pages
        self._s3_connection_options = s3_connection_options
        self._credential_provider = credential_provider
        self._cached_run_id: str | None = None
        self._cached_factory: (
            AsyncSqliteReaderFactory | AsyncSqliteRangeReaderFactory | None
        ) = None

    def _resolve_factory(
        self, manifest: Manifest
    ) -> AsyncSqliteReaderFactory | AsyncSqliteRangeReaderFactory:
        run_id = manifest.required_build.run_id
        if run_id == self._cached_run_id and self._cached_factory is not None:
            return self._cached_factory
        sizes = [shard.db_bytes for shard in manifest.shards]
        mode = self._policy.decide(sizes)
        factory = make_async_sqlite_reader_factory(
            mode=mode,
            mmap_size=self._mmap_size,
            page_cache_pages=self._page_cache_pages,
            s3_connection_options=self._s3_connection_options,
            credential_provider=self._credential_provider,
        )
        self._cached_run_id = run_id
        self._cached_factory = factory
        return factory

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncSqliteShardReader | AsyncSqliteRangeShardReader:
        factory = self._resolve_factory(manifest)
        return await factory(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
        )
