"""SlateDB-specific writer adapter built on the slatedb.uniffi async bindings.

shardyfusion's writer call sites are synchronous (Spark UDFs, Dask
worker functions, the multiprocessing Python writer, …) but
slatedb 0.12 only exposes an async API on ``Db`` / ``WriteBatch``.
This adapter bridges the gap by submitting every async call through
:func:`shardyfusion._async_bridge.call_sync`, which is built on the
process-global event loop owned by :mod:`shardyfusion._slatedb_runtime`
and adds optional per-call timeouts and Ctrl-C cancellation safety.

Per-operation timeouts default to ``None`` (unbounded) to preserve the
pre-typed-bridge behavior. Configure :class:`SlateDbBridgeTimeouts` and
attach it to :class:`SlateDbFactory` to opt in to bounded operations;
on timeout the in-flight coroutine is cancelled on the bridge loop and
:class:`shardyfusion.errors.BridgeTimeoutError` (retryable) is raised.

Key shape changes versus pre-0.12:

* The legacy ``SlateDB(path, url=..., env_file=..., settings=...)``
  constructor is gone. We open via
  ``DbBuilder("", ObjectStore.resolve(db_url))`` and apply settings
  through :class:`shardyfusion.SlateDbSettings`. ``db_url`` is the
  full per-shard URL — :func:`slatedb.Db.resolve_object_store` parses
  the URL's path component into a ``PrefixStore`` root.
* ``Db.close()`` was renamed to :meth:`Db.shutdown`.
* Flushing requires :class:`FlushOptions(flush_type=FlushType.WAL)`
  — there is no zero-arg ``flush_with_options("wal")`` shortcut.
* Checkpoints are no longer created adapter-side. The :class:`DbAdapter`
  protocol's old ``checkpoint()`` method is gone; writers generate an
  opaque ``uuid.uuid4().hex`` after :meth:`close` succeeds.

The backend-neutral :class:`DbAdapter` and :class:`DbAdapterFactory`
protocols live in :mod:`shardyfusion._adapter`. See
``docs/architecture/index.md`` for the rationale (single-writer
invariant + S3 strong consistency + serial publish removes the need
for SlateDB-managed checkpoint pinning).
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
from .credentials import CredentialProvider, apply_env_file, resolve_env_file
from .errors import BridgeTimeoutError, DbAdapterError, ShardWriteError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .type_defs import S3ConnectionOptions

if TYPE_CHECKING:
    from slatedb.uniffi import Db, Settings

__all__ = [
    "DefaultSlateDbAdapter",
    "SlateDbBridgeTimeouts",
    "SlateDbFactory",
]

_logger = get_logger(__name__)


def _is_transient_slatedb_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a slatedb ``Error.Unavailable``.

    ``Error.Unavailable`` is the binding's "temporary unavailability"
    variant — typically an upstream object-store hiccup, throttle, or
    transient network failure. Treating it as retryable preserves the
    behaviour of the underlying retry loop, which only retries
    ``ShardyfusionError`` subclasses with ``retryable=True``.

    Other ``slatedb.uniffi`` ``Error`` variants (``Invalid``, ``Data``,
    ``Internal``, ``Closed``, ``Transaction``) indicate programmer or
    structural failures that retrying will not fix, and are wrapped as
    non-retryable :class:`DbAdapterError` by the caller.
    """
    try:
        from slatedb.uniffi import Error  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - runtime dependent
        return False
    unavailable = getattr(Error, "Unavailable", None)
    return unavailable is not None and isinstance(exc, unavailable)


@dataclass(slots=True, frozen=True)
class SlateDbBridgeTimeouts:
    """Per-operation wall-clock timeouts for the sync\u2192async bridge.

    Each field is an optional float in seconds. ``None`` (default)
    means "block forever", matching the legacy
    :func:`shardyfusion._slatedb_runtime.run_coro` behavior. Setting a
    timeout causes the coroutine to be cancelled on the bridge loop and
    :class:`shardyfusion.errors.BridgeTimeoutError` (retryable) to be
    raised when the deadline elapses.

    Recommended starting points if you want bounded operations:

    * ``open_s``: 60 \u2014 catches credential or DNS hangs at startup.
    * ``write_s``: 30 \u2014 typical S3 PUT under backpressure.
    * ``flush_s``: 60 \u2014 WAL flush waits for an object-store write.
    * ``shutdown_s``: 300 \u2014 final compaction / WAL drain.

    All four default to ``None`` so adopting the typed bridge does not
    change behavior for existing deployments.
    """

    open_s: float | None = None
    write_s: float | None = None
    flush_s: float | None = None
    shutdown_s: float | None = None


@dataclass(slots=True)
class SlateDbFactory:
    """Picklable factory that captures SlateDB-specific config.

    ``settings`` is the typed :class:`SlateDbSettings` dataclass.
    ``bridge_timeouts`` controls optional per-operation timeouts on the
    sync\u2192async bridge; defaults to all-unbounded for backward
    compatibility.

    ``s3_connection_options`` mirrors the field on
    :class:`~shardyfusion.local_slatedb_adapter.LocalSlateDbFactory` and
    :class:`~shardyfusion.reader._types.SlateDbReaderFactory`: when set,
    transport overrides are translated into the standard ``AWS_*`` env
    vars honored by Apache ``object_store`` for the duration of the
    ``ObjectStore.resolve()`` call inside the adapter.  Required for
    path-style S3-compatible stores such as Garage / MinIO / Ceph.
    """

    env_file: str | None = None
    settings: SlateDbSettings | None = None
    credential_provider: CredentialProvider | None = None
    bridge_timeouts: SlateDbBridgeTimeouts = field(
        default_factory=SlateDbBridgeTimeouts
    )
    s3_connection_options: S3ConnectionOptions | None = None

    def __call__(self, *, db_url: str, local_dir: Path) -> DefaultSlateDbAdapter:
        with resolve_env_file(
            self.env_file,
            self.credential_provider,
            connection_options=self.s3_connection_options,
        ) as env_path:
            return DefaultSlateDbAdapter(
                local_dir=local_dir,
                db_url=db_url,
                env_file=env_path,
                settings=self.settings,
                bridge_timeouts=self.bridge_timeouts,
            )


class DefaultSlateDbAdapter:
    """Adapter using the slatedb.uniffi ``Db`` + ``WriteBatch`` surface.

    ``local_dir`` is preserved on the dataclass for symmetry with the
    other backend adapters but is *not* used by the uniffi bindings —
    SlateDB writes go straight to the object store identified by
    ``db_url``. We accept it to avoid rippling factory signature
    changes through every framework writer.
    """

    __slots__ = ("_db_url", "_local_dir", "_db", "_settings", "_timeouts")

    _db: Db

    def __init__(
        self,
        *,
        local_dir: Path,
        db_url: str,
        env_file: str | None,
        settings: SlateDbSettings | None,
        bridge_timeouts: SlateDbBridgeTimeouts | None = None,
    ) -> None:
        self._db_url = db_url
        self._local_dir = local_dir
        self._settings = settings
        self._timeouts = bridge_timeouts or SlateDbBridgeTimeouts()

        db_builder_cls = get_db_builder_class()
        object_store_cls = get_object_store_class()

        # Load the env file into os.environ for the duration of the
        # store/builder construction so ObjectStore.resolve(url) picks
        # up AWS_*/AZURE_*/GOOGLE_* values from it. The handle holds
        # references to the resolved store after this block exits, so
        # the temporary env mutation does not need to persist.
        try:
            with apply_env_file(env_file):
                store = object_store_cls.resolve(db_url)
                builder = db_builder_cls("", store)
                if settings is not None:
                    settings_obj = self._build_settings(settings)
                    builder.with_settings(settings_obj)
                self._db = call_sync(builder.build(), timeout=self._timeouts.open_s)
        except BridgeTimeoutError:
            # BridgeTimeoutError is retryable; re-raise so the shard
            # retry path can honour exc.retryable. Wrapping it in a
            # generic DbAdapterError would silently downgrade to
            # retryable=False and short-circuit retries.
            raise
        except Exception as exc:
            # slatedb's ``Error.Unavailable`` signals transient
            # object-store unavailability (network hiccup, S3 503/throttle,
            # missing dependency that will come back). Surface it as a
            # retryable ``ShardWriteError`` so ``_run_attempts_with_retry``
            # honours its ``retryable`` flag instead of giving up after a
            # single attempt. Everything else (bad URL, invalid config,
            # corrupt manifest, internal binding error) stays as
            # non-retryable ``DbAdapterError``.
            if _is_transient_slatedb_error(exc):
                raise ShardWriteError(
                    f"Transient SlateDB open failure at {db_url!r}: {exc}"
                ) from exc
            raise DbAdapterError(
                f"Failed to open SlateDB at {db_url!r} via slatedb.uniffi"
            ) from exc

        log_event(
            "shard_adapter_opened",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
        )

    @staticmethod
    def _build_settings(settings: SlateDbSettings) -> Settings:
        settings_cls = get_settings_class()
        obj = settings_cls.default()
        settings.apply(obj)
        return obj

    @property
    def db(self) -> Db:
        return self._db

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
        # WriteBatch put/delete are sync in the uniffi bindings; only
        # Db.write itself is async.
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
            "shard_adapter_flushed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=self._db_url,
        )

    def db_bytes(self) -> int:
        # SlateDB shards live in object storage; the writer side has no
        # cheap way to total their materialized size. Auto-SQLite access
        # policies do not consult SlateDB sizes.
        return 0

    def seal(self) -> None:
        # SlateDB has nothing to finalize beyond flush() — the data is
        # already durable in object storage at that point.
        return None

    def close(self) -> None:
        try:
            call_sync(self._db.shutdown(), timeout=self._timeouts.shutdown_s)
        except Exception as exc:
            log_failure(
                "slatedb_adapter_close_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise
        log_event(
            "shard_adapter_closed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=self._db_url,
        )
