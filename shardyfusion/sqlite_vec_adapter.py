"""SQLite-vec unified adapter — KV + vector search in one DB file.

Extends the SQLite adapter pattern to embed both a KV table and a
sqlite-vec virtual table in the same ``shard.db`` file.  This gives
a single S3 object per shard for both point lookups and ANN search.

Writer:
    ``SqliteVecAdapter`` / ``SqliteVecFactory`` — builds a local SQLite DB
    with ``kv`` table (KV data) and ``vec_index`` virtual table (vectors),
    uploads once on close.

Reader:
    ``SqliteVecShardReader`` / ``SqliteVecReaderFactory`` — downloads the
    DB file and supports both ``get(key)`` and ``search(query, top_k)``.

Requires the ``vector-sqlite`` extra (``sqlite-vec``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from .config import VectorSpec
from .credentials import CredentialProvider, S3Credentials
from .errors import ShardyfusionError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .storage import create_s3_client, get_bytes, put_bytes
from .type_defs import S3ConnectionOptions
from .vector.types import SearchResult

_logger = get_logger(__name__)

_DB_FILENAME = "shard.db"
_DB_IDENTITY_FILENAME = "shard.identity.json"

_SQLITE_VEC_IMPORT_ERROR = (
    "Unified KV+vector with SQLite requires the 'vector-sqlite' extra. "
    "Install it with: pip install shardyfusion[vector-sqlite]"
)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into the given connection."""
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_SQLITE_VEC_IMPORT_ERROR) from exc

    sqlite_vec.load(conn)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SqliteVecAdapterError(ShardyfusionError):
    """SQLite-vec adapter error (non-retryable)."""

    retryable = False


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteVecFactory:
    """Factory that builds unified KV + vector SQLite shards."""

    vector_spec: VectorSpec
    page_size: int = 4096
    cache_size_pages: int = -2000
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(self, *, db_url: str, local_dir: Path) -> SqliteVecAdapter:
        return SqliteVecAdapter(
            db_url=db_url,
            local_dir=local_dir,
            vector_spec=self.vector_spec,
            page_size=self.page_size,
            cache_size_pages=self.cache_size_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )


class SqliteVecAdapter:
    """Unified KV + vector write adapter: single SQLite DB per shard.

    Creates a ``kv`` table for KV data and a ``vec_index`` virtual table
    for vector embeddings.  Both are uploaded in a single ``.db`` file.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        vector_spec: VectorSpec,
        page_size: int = 4096,
        cache_size_pages: int = -2000,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._db_url = db_url
        self._local_dir = local_dir
        self._db_path = local_dir / _DB_FILENAME
        self._vector_spec = vector_spec
        self._uploaded = False
        self._closed = False
        self._checkpointed = False
        self._s3_conn_opts = s3_connection_options
        self._s3_creds: S3Credentials | None = (
            credential_provider.resolve() if credential_provider else None
        )

        local_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), isolation_level=None)

        # Load sqlite-vec extension
        _load_sqlite_vec(conn)

        # SQLite pragmas for write performance
        page_size = int(page_size)
        cache_size_pages = int(cache_size_pages)
        conn.execute(f"PRAGMA page_size = {page_size}")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute(f"PRAGMA cache_size = {cache_size_pages}")
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        conn.execute("PRAGMA temp_store = MEMORY")

        # KV table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv "
            "(k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID"
        )

        # Vector index table (sqlite-vec)
        dim = vector_spec.dim
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index "
            f"USING vec0(embedding float[{dim}])"
        )

        # Payload table for vector metadata
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_payloads "
            "(rowid INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )

        # ID mapping table for non-integer vector IDs
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_id_map "
            "(internal_id INTEGER PRIMARY KEY, original_id TEXT NOT NULL)"
        )

        conn.execute("BEGIN")
        self._conn: sqlite3.Connection | None = conn
        self._next_vec_id = 0

        log_event(
            "sqlite_vec_adapter_opened",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
            dim=dim,
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

    # -- KV operations (DbAdapter protocol) --

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        """Write KV pairs to the kv table."""
        if self._conn is None:
            raise SqliteVecAdapterError("Adapter already closed")
        if self._checkpointed:
            raise SqliteVecAdapterError("Cannot write after checkpoint")
        self._conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", pairs)

    # -- Vector operations --

    def write_vector_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write vectors to the vec_index table."""
        if self._conn is None:
            raise SqliteVecAdapterError("Adapter already closed")
        if self._checkpointed:
            raise SqliteVecAdapterError("Cannot write after checkpoint")

        reserved_int_ids = {
            int(ids[i])
            for i in range(len(ids))
            if isinstance(ids[i], (int, np.integer))
        }
        next_row_id = self._next_vec_id
        vec_rows: list[tuple[int, bytes]] = []
        payload_rows: list[tuple[int, str]] = []
        id_map_rows: list[tuple[int, str]] = []

        for i in range(len(ids)):
            raw_id = ids[i]
            if isinstance(raw_id, (int, np.integer)):
                row_id = int(raw_id)
                if row_id + 1 > next_row_id:
                    next_row_id = row_id + 1
            else:
                while next_row_id in reserved_int_ids:
                    next_row_id += 1
                row_id = next_row_id
                next_row_id += 1
                id_map_rows.append((row_id, str(raw_id)))

            vec_rows.append((row_id, vectors[i].astype(np.float32).tobytes()))
            if payloads is not None and payloads[i] is not None:
                payload_rows.append((row_id, json.dumps(payloads[i], default=str)))

        if id_map_rows:
            self._conn.executemany(
                "INSERT INTO vec_id_map (internal_id, original_id) VALUES (?, ?)",
                id_map_rows,
            )
        self._conn.executemany(
            "INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)",
            vec_rows,
        )
        if payload_rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vec_payloads (rowid, payload) VALUES (?, ?)",
                payload_rows,
            )

        self._next_vec_id = next_row_id

    # -- Lifecycle --

    def flush(self) -> None:
        pass  # no-op: local file, no WAL

    def checkpoint(self) -> str | None:
        if self._conn is None:
            raise SqliteVecAdapterError("Adapter already closed")
        if self._checkpointed:
            raise SqliteVecAdapterError("Adapter already checkpointed")
        self._conn.execute("COMMIT")
        self._conn.execute("PRAGMA optimize")
        self._conn.close()
        self._conn = None
        self._checkpointed = True

        with open(self._db_path, "rb") as f:
            file_hash = hashlib.file_digest(f, "sha256").hexdigest()
        log_event(
            "sqlite_vec_adapter_checkpointed",
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
                client = (
                    create_s3_client(self._s3_creds, self._s3_conn_opts)
                    if self._s3_creds or self._s3_conn_opts
                    else None
                )
                put_bytes(
                    s3_key,
                    self._db_path.read_bytes(),
                    content_type="application/x-sqlite3",
                    s3_client=client,
                )
                self._uploaded = True
                log_event(
                    "sqlite_vec_adapter_uploaded",
                    level=logging.DEBUG,
                    logger=_logger,
                    db_url=self._db_url,
                )
        except Exception as exc:
            log_failure(
                "sqlite_vec_adapter_close_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise
        else:
            self._closed = True


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteVecReaderFactory:
    """Factory that creates unified KV + vector shard readers."""

    mmap_size: int = 268_435_456  # 256 MB
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None = None,
    ) -> SqliteVecShardReader:
        return SqliteVecShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            mmap_size=self.mmap_size,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )


class SqliteVecShardReader:
    """Unified shard reader: KV point lookups + vector search on one DB.

    Downloads the ``.db`` file from S3 (same pattern as
    ``SqliteShardReader``), then supports both ``get()`` and ``search()``.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None = None,
        mmap_size: int = 268_435_456,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._db_url = db_url
        self._local_dir = local_dir
        local_dir.mkdir(parents=True, exist_ok=True)
        db_path = local_dir / _DB_FILENAME
        identity_path = local_dir / _DB_IDENTITY_FILENAME

        # Check cache validity
        if not _is_cached_snapshot_current(identity_path, db_url, checkpoint_id):
            s3_key = f"{db_url.rstrip('/')}/{_DB_FILENAME}"
            creds = credential_provider.resolve() if credential_provider else None
            client = (
                create_s3_client(creds, s3_connection_options)
                if creds or s3_connection_options
                else None
            )
            data = get_bytes(s3_key, s3_client=client)
            db_path.write_bytes(data)
            identity_path.write_text(
                json.dumps({"db_url": db_url, "checkpoint_id": checkpoint_id})
            )

        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        _load_sqlite_vec(conn)
        conn.execute(f"PRAGMA mmap_size = {mmap_size}")
        self._conn: sqlite3.Connection | None = conn

    def get(self, key: bytes) -> bytes | None:
        """KV point lookup."""
        if self._conn is None:
            raise SqliteVecAdapterError("Reader already closed")
        row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        ef: int = 50,
    ) -> list[SearchResult]:
        """Vector similarity search via sqlite-vec."""
        if self._conn is None:
            raise SqliteVecAdapterError("Reader already closed")

        query_bytes = query.astype(np.float32).tobytes()
        rows = self._conn.execute(
            "SELECT rowid, distance FROM vec_index "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (query_bytes, top_k),
        ).fetchall()

        results: list[SearchResult] = []
        for row_id, distance in rows:
            # Resolve original ID if an id_map entry exists
            original_id: int | str = row_id
            try:
                id_row = self._conn.execute(
                    "SELECT original_id FROM vec_id_map WHERE internal_id = ?",
                    (row_id,),
                ).fetchone()
                if id_row is not None:
                    original_id = id_row[0]
            except sqlite3.OperationalError:
                pass  # table doesn't exist (old format)

            payload: dict[str, Any] | None = None
            payload_row = self._conn.execute(
                "SELECT payload FROM vec_payloads WHERE rowid = ?",
                (row_id,),
            ).fetchone()
            if payload_row is not None:
                payload = json.loads(payload_row[0])

            results.append(
                SearchResult(
                    id=original_id,
                    score=float(distance),
                    payload=payload,
                )
            )
        return results

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_cached_snapshot_current(
    identity_path: Path,
    db_url: str,
    checkpoint_id: str | None,
) -> bool:
    """Check if the local cached DB matches the expected identity."""
    if not identity_path.exists():
        return False
    try:
        identity = json.loads(identity_path.read_text())
        return (
            identity.get("db_url") == db_url
            and identity.get("checkpoint_id") == checkpoint_id
        )
    except (json.JSONDecodeError, OSError):
        return False
