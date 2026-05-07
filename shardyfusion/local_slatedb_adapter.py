"""Local-first SlateDB adapter: writes to a ``file://`` object store, uploads to S3
on close.

This adapter mirrors :class:`~shardyfusion.slatedb_adapter.DefaultSlateDbAdapter` in
its write/flush/seal surface but targets a local ``file://`` object-store path so that
every :meth:`write_batch` and :meth:`flush` stays on the local filesystem.  When the
shard's :class:`~shardyfusion._shard_writer.write_shard_core` context manager exits,
:meth:`close` shuts down the local SlateDB instance and then uploads the entire
materialised store — every file under ``local_dir/_slatedb_store`` — to the shard's
canonical S3 ``db_url`` via the same ``obstore`` + ``ObstoreBackend`` path used by
:class:`~shardyfusion.sqlite_adapter.SqliteAdapter`.

The local store directory is ``local_dir / "_slatedb_store"``.  The writer core's
:func:`~shardyfusion._shard_writer.make_shard_local_dir` already ensures per-attempt
isolation, so two attempts never share a local directory.

Compared to the direct-to-S3 :class:`DefaultSlateDbAdapter`:
* Per-batch network I/O is eliminated — all writes stay local.
* The S3 upload is a single bulk copy at :meth:`close` time.
* :meth:`db_bytes` returns the actual on-disk size of the local store.
* Write-side throughput under backpressure is decoupled from S3 latency.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

from ._async_bridge import call_sync
from ._settings import SlateDbSettings
from ._slatedb_symbols import (
    get_db_builder_class,
    get_flush_options_classes,
    get_object_store_class,
    get_settings_class,
    get_write_batch_class,
)
from .credentials import (
    CredentialProvider,
    S3Credentials,
    apply_env_file,
    resolve_env_file,
)
from .errors import BridgeTimeoutError, DbAdapterError, ShardWriteError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .slatedb_adapter import SlateDbBridgeTimeouts, _is_transient_slatedb_error
from .storage import ObstoreBackend, create_s3_store, parse_s3_url
from .type_defs import S3ConnectionOptions

if TYPE_CHECKING:
    from slatedb.uniffi import Db, Settings

__all__ = [
    "LocalSlateDbAdapter",
    "LocalSlateDbFactory",
]

_logger = get_logger(__name__)

_STORE_DIR_NAME = "_slatedb_store"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LocalSlateDbFactory:
    """Picklable factory that builds local-then-upload SlateDB KV shards.

    ``settings`` is the typed :class:`SlateDbSettings` dataclass.
    ``bridge_timeouts`` controls optional per-operation timeouts on the
    sync→async bridge; defaults to all-unbounded for backward
    compatibility.
    ``credential_provider`` supplies S3 credentials for the upload step
    (resolved eagerly in the adapter).
    ``s3_connection_options`` configures the obstore S3 store for upload.
    """

    env_file: str | None = None
    settings: SlateDbSettings | None = None
    credential_provider: CredentialProvider | None = None
    bridge_timeouts: SlateDbBridgeTimeouts = field(
        default_factory=SlateDbBridgeTimeouts
    )
    s3_connection_options: S3ConnectionOptions | None = None

    def __call__(self, *, db_url: str, local_dir: Path) -> LocalSlateDbAdapter:
        with resolve_env_file(self.env_file, self.credential_provider) as env_path:
            return LocalSlateDbAdapter(
                db_url=db_url,
                local_dir=local_dir,
                env_file=env_path,
                settings=self.settings,
                bridge_timeouts=self.bridge_timeouts,
                s3_connection_options=self.s3_connection_options,
                credential_provider=self.credential_provider,
            )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LocalSlateDbAdapter:
    """Local-first SlateDB adapter: writes to ``file://``, uploads to S3 on close.

    Satisfies the :class:`~shardyfusion._adapter.DbAdapter` protocol.

    Unlike :class:`~shardyfusion.slatedb_adapter.DefaultSlateDbAdapter`, where
    ``local_dir`` is preserved-but-unused (the uniffi bindings stream
    everything to S3 directly), here ``local_dir`` *is* used: the SlateDB
    instance is opened against ``local_dir / "_slatedb_store"`` and every
    :meth:`write_batch` and :meth:`flush` materialises onto the local
    filesystem.  :meth:`close` then performs a two-phase finalisation —
    ``db.shutdown()`` followed by a bulk upload of the materialised store
    to ``db_url``.  Both phases are idempotent across repeated
    :meth:`close` calls (see :meth:`close` for the exact contract).
    """

    __slots__ = (
        "_db_url",
        "_local_dir",
        "_store_dir",
        "_db",
        "_settings",
        "_timeouts",
        "_s3_conn_opts",
        "_s3_creds",
        "_uploaded",
        "_closed",
    )

    _db: Db

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        env_file: str | None,
        settings: SlateDbSettings | None,
        bridge_timeouts: SlateDbBridgeTimeouts | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._db_url = db_url
        self._local_dir = local_dir
        self._settings = settings
        self._timeouts = bridge_timeouts or SlateDbBridgeTimeouts()
        self._s3_conn_opts = s3_connection_options
        self._s3_creds: S3Credentials | None = (
            credential_provider.resolve() if credential_provider else None
        )
        self._uploaded = False
        self._closed = False

        # Resolve the local store directory and build the file:// URL.
        local_dir.mkdir(parents=True, exist_ok=True)
        self._store_dir = local_dir / _STORE_DIR_NAME
        self._store_dir.mkdir(parents=True, exist_ok=True)
        file_url = f"file://{self._store_dir}"

        db_builder_cls = get_db_builder_class()
        object_store_cls = get_object_store_class()

        try:
            with apply_env_file(env_file):
                store = object_store_cls.resolve(file_url)
                builder = db_builder_cls("", store)
                if settings is not None:
                    settings_obj = self._build_settings(settings)
                    builder.with_settings(settings_obj)
                self._db = call_sync(builder.build(), timeout=self._timeouts.open_s)
        except BridgeTimeoutError:
            raise
        except Exception as exc:
            if _is_transient_slatedb_error(exc):
                raise ShardWriteError(
                    f"Transient local SlateDB open failure at {file_url!r}: {exc}"
                ) from exc
            raise DbAdapterError(
                f"Failed to open local SlateDB at {file_url!r} via slatedb.uniffi"
            ) from exc

        log_event(
            "local_slatedb_adapter_opened",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
            file_url=file_url,
        )

    @staticmethod
    def _build_settings(settings: SlateDbSettings) -> Settings:
        settings_cls = get_settings_class()
        obj = settings_cls.default()
        settings.apply(obj)
        return obj

    # ------------------------------------------------------------------
    # DbAdapter protocol
    # ------------------------------------------------------------------

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
        batch = list(pairs)
        if not batch:
            return

        write_batch_cls = get_write_batch_class()
        wb = write_batch_cls()
        for key, value in batch:
            wb.put(key, value)
        call_sync(self._db.write(wb), timeout=self._timeouts.write_s)

    def flush(self) -> None:
        flush_options_cls, flush_type_cls = get_flush_options_classes()
        options = flush_options_cls(flush_type=flush_type_cls.WAL)
        call_sync(
            self._db.flush_with_options(options),
            timeout=self._timeouts.flush_s,
        )
        log_event(
            "local_slatedb_adapter_flushed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=self._db_url,
        )

    def db_bytes(self) -> int:
        return _dir_size(self._store_dir)

    def seal(self) -> None:
        return None

    def close(self) -> None:
        """Shut down the local SlateDB instance and upload the store to S3.

        Two-phase finalisation:

        1. ``db.shutdown()`` finalises manifests, WALs, and in-flight
           compactions on the local filesystem.  If this raises, the
           adapter is marked closed (``_closed=True``) and the upload is
           skipped — there is no point uploading an unfinalised store and
           a subsequent :meth:`close` would only re-raise on the
           already-shut-down handle.
        2. ``_upload_store`` bulk-copies every file under the materialised
           store to ``db_url``.  If this raises, the adapter remains
           ``_closed=True`` but ``_uploaded=False``, so a subsequent
           :meth:`close` skips the (already-completed) shutdown and
           retries only the upload — useful for transient S3 failures.

        Repeated calls after a successful close are no-ops.
        """
        if self._closed:
            return
        try:
            # 1. Shut down the local SlateDB instance — this finalises
            #    manifests, WALs, and any in-flight compactions on disk.
            call_sync(self._db.shutdown(), timeout=self._timeouts.shutdown_s)
        except Exception as exc:
            log_failure(
                "local_slatedb_adapter_shutdown_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                db_url=self._db_url,
            )
            self._closed = True
            raise

        # Mark closed immediately after shutdown so that a subsequent close()
        # call does not attempt to shut down again, even if the upload fails.
        self._closed = True

        # 2. Upload the materialised store to S3.
        if not self._uploaded:
            try:
                _upload_store(
                    store_dir=self._store_dir,
                    db_url=self._db_url,
                    s3_connection_options=self._s3_conn_opts,
                    s3_credentials=self._s3_creds,
                )
            except Exception as exc:
                log_failure(
                    "local_slatedb_adapter_upload_failed",
                    severity=FailureSeverity.ERROR,
                    logger=_logger,
                    error=exc,
                    db_url=self._db_url,
                )
                raise
            self._uploaded = True
            log_event(
                "local_slatedb_adapter_uploaded",
                level=logging.DEBUG,
                logger=_logger,
                db_url=self._db_url,
            )

        log_event(
            "local_slatedb_adapter_closed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=self._db_url,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dir_size(directory: Path) -> int:
    """Sum the on-disk sizes of every regular file under *directory*."""
    total = 0
    for entry in directory.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _upload_store(
    *,
    store_dir: Path,
    db_url: str,
    s3_connection_options: S3ConnectionOptions | None,
    s3_credentials: S3Credentials | None,
) -> None:
    """Walk *store_dir* and upload every file to S3 under *db_url*.

    Each file's path relative to *store_dir* becomes the object key
    appended to the shard's ``db_url`` prefix.  For example, a local file
    ``<store_dir>/manifest/000001.manifest`` uploaded for
    ``db_url = s3://bucket/prefix/shards/db=00000/attempt=00`` becomes
    ``s3://bucket/prefix/shards/db=00000/attempt=00/manifest/000001.manifest``.
    """
    bucket, key_prefix = parse_s3_url(db_url)
    key_prefix = key_prefix.rstrip("/")

    store = create_s3_store(
        bucket=bucket,
        credentials=s3_credentials,
        connection_options=s3_connection_options,
    )
    backend = ObstoreBackend(store)

    # Uploads are sequential and blocking — fine for typical SlateDB shard
    # sizes (a few small manifest/WAL files); revisit with parallelism if
    # shards grow large enough that upload latency dominates close().
    for file_path in store_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(store_dir).as_posix()
        s3_key = f"{key_prefix}/{rel_path}"
        s3_url = f"s3://{bucket}/{s3_key}"
        backend.put(
            s3_url,
            file_path.read_bytes(),
            content_type="application/octet-stream",
        )
