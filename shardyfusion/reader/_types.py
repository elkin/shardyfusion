"""Data types and factories for sharded readers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from shardyfusion._async_bridge import call_sync, gather_sync
from shardyfusion._slatedb_symbols import get_key_range_class, get_reader_symbols
from shardyfusion.credentials import (
    CredentialProvider,
    apply_env_file,
    resolve_env_file,
)
from shardyfusion.errors import ConfigValidationError, DbAdapterError
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_failure,
)
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import (
    KeyEncoding,
    ShardHashAlgorithm,
    ShardingStrategy,
)
from shardyfusion.type_defs import (
    Manifest,
    ShardReader,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Null shard reader for empty shards (db_url=None)
# ---------------------------------------------------------------------------


class _NullShardReader:
    """No-op reader for empty shards — always returns ``None``."""

    def get(self, key: bytes) -> bytes | None:
        return None

    def close(self) -> None:
        pass


@dataclass(slots=True, frozen=True)
class SnapshotInfo:
    """Read-only snapshot of manifest metadata."""

    run_id: str
    num_dbs: int
    sharding: ShardingStrategy
    created_at: datetime
    manifest_ref: str
    row_count: int = 0
    key_encoding: KeyEncoding = KeyEncoding.U64BE
    hash_algorithm: ShardHashAlgorithm = ShardHashAlgorithm.XXH3_64


@dataclass(slots=True, frozen=True)
class ShardDetail:
    """Per-shard metadata exposed by ``shard_details()``."""

    db_id: int
    row_count: int
    min_key: int | float | str | bytes | None
    max_key: int | float | str | bytes | None
    db_url: str | None


@dataclass(slots=True, frozen=True)
class ReaderHealth:
    """Diagnostic snapshot of reader state."""

    status: Literal["healthy", "degraded", "unhealthy"]
    manifest_ref: str
    manifest_age: timedelta
    num_shards: int
    is_closed: bool


# ---------------------------------------------------------------------------
# SlateDB shard reader (uniffi)
# ---------------------------------------------------------------------------


class _SlateDbReaderHandle:
    """Sync ``ShardReader`` over a uniffi async ``DbReader``.

    All async calls are funneled through the shardyfusion process-global
    asyncio loop (:mod:`shardyfusion._slatedb_runtime`) so callers stay
    on the synchronous :class:`~shardyfusion.type_defs.ShardReader`
    contract. ``checkpoint_id`` is accepted by the factory only for
    interface symmetry with other backends; uniffi 0.12 does not expose
    checkpoint pinning, and shardyfusion's single-writer + serial
    publish invariants make it unnecessary against S3 strong consistency.
    """

    __slots__ = ("_reader", "_iterator_chunk_size", "_closed")

    def __init__(self, reader: Any, *, iterator_chunk_size: int) -> None:
        self._reader = reader
        self._iterator_chunk_size = iterator_chunk_size
        self._closed = False

    def get(self, key: bytes) -> bytes | None:
        if self._closed:
            return None
        return call_sync(self._reader.get(bytes(key)))

    def scan_iter(
        self, start: bytes | None = None, end: bytes | None = None
    ) -> Iterator[tuple[bytes, bytes]]:
        """Yield ``(key, value)`` pairs in ``[start, end)``.

        The underlying uniffi iterator is async; we drive it from the
        shardyfusion runtime loop in chunks of ``iterator_chunk_size``
        to amortize the per-call sync→async bridge cost (a single hop is
        ~15–40 µs, so per-row hops would dominate scans by ~30×).
        """
        if self._closed:
            return
        key_range_cls = get_key_range_class()
        key_range = key_range_cls(
            start=bytes(start) if start is not None else None,
            start_inclusive=True,
            end=bytes(end) if end is not None else None,
            end_inclusive=False,
        )

        async def _open_iterator() -> Any:
            return await self._reader.scan(key_range)

        iterator = call_sync(_open_iterator())
        chunk = self._iterator_chunk_size
        # Probe once outside the loop; an inline ``try/except
        # AttributeError`` would also swallow AttributeErrors raised
        # *inside* a future ``next_batch`` body.
        has_next_batch = hasattr(iterator, "next_batch")

        async def _drain() -> list[tuple[bytes, bytes]]:
            if has_next_batch:
                batch = await iterator.next_batch(chunk)
                return [(bytes(kv.key), bytes(kv.value)) for kv in batch]
            out: list[tuple[bytes, bytes]] = []
            for _ in range(chunk):
                item = await iterator.next()
                if item is None:
                    break
                out.append((bytes(item.key), bytes(item.value)))
            return out

        try:
            while True:
                batch = call_sync(_drain())
                if not batch:
                    return
                yield from batch
        finally:
            # Release the iterator on exhaustion, exception, or
            # ``GeneratorExit``. ``hasattr`` keeps the call forward-
            # compatible with future uniffi versions that add ``aclose``.
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                try:
                    call_sync(aclose())
                except Exception as exc:
                    log_failure(
                        "scan_iter_aclose_failed",
                        severity=FailureSeverity.TRANSIENT,
                        logger=_logger,
                        error=exc,
                    )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # uniffi DbReader exposes async ``shutdown()`` that releases the
        # underlying Tokio/object-store resources. Without this call the
        # reader (and its background prefetch tasks) would survive until
        # Python finalization, which is unacceptable for long-running
        # services that periodically refresh snapshots. Swallow shutdown
        # errors — the handle is already marked closed and subsequent
        # ``get``/``scan_iter`` calls short-circuit on ``self._closed``.
        shutdown = getattr(self._reader, "shutdown", None)
        if shutdown is None:
            return
        try:
            call_sync(shutdown())
        except Exception:
            # Defensive: do not let a slatedb shutdown hiccup break a
            # caller's ``with reader:`` cleanup path. The handle is
            # already closed from the caller's perspective.
            pass


@dataclass(slots=True)
class SlateDbReaderFactory:
    """Picklable reader factory that captures SlateDB-specific config.

    ``iterator_chunk_size`` controls how many rows the sync iterator
    bridge fetches per async hop — see :class:`_SlateDbReaderHandle`.
    Must be positive; ``0`` would silently produce empty scans because
    the per-chunk drain loop runs ``range(chunk_size)`` rows at a time.
    """

    env_file: str | None = None
    credential_provider: CredentialProvider | None = None
    iterator_chunk_size: int = 1024

    def __post_init__(self) -> None:
        if self.iterator_chunk_size <= 0:
            raise ConfigValidationError(
                f"iterator_chunk_size must be positive, got {self.iterator_chunk_size}"
            )

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> ShardReader:
        del local_dir, checkpoint_id, manifest  # not used by uniffi reader
        try:
            db_reader_builder_cls, _, object_store_cls = get_reader_symbols()
        except DbAdapterError as exc:  # pragma: no cover - runtime dependent
            raise DbAdapterError(
                "slatedb package is required for reading shards"
            ) from exc

        with resolve_env_file(self.env_file, self.credential_provider) as env_path:
            with apply_env_file(env_path):
                store = object_store_cls.resolve(db_url)
                builder = db_reader_builder_cls("", store)
                reader = call_sync(builder.build())

        return _SlateDbReaderHandle(
            reader, iterator_chunk_size=self.iterator_chunk_size
        )

    def open_many(
        self,
        specs: list[tuple[str, Manifest]],
    ) -> list[ShardReader]:
        """Open multiple shard readers concurrently in one bridge hop.

        Equivalent to ``[self(db_url=u, ..., manifest=m) for u, m in specs]``
        but submits all ``DbReaderBuilder.build()`` coroutines to the
        bridge loop in a single :func:`asyncio.gather` so the slatedb
        Tokio runtime can drive them in parallel. For an N-shard reader
        this turns N sequential builds into one round-trip; concretely,
        4 shards \u00d7 ~50ms build time goes from ~200ms wall-clock to
        ~50ms.

        Returns the readers in the same order as ``specs``. Raises the
        first exception encountered (or :class:`BridgeTimeoutError` if
        the gather is bounded via a future overload).
        """
        if not specs:
            return []
        try:
            db_reader_builder_cls, _, object_store_cls = get_reader_symbols()
        except DbAdapterError as exc:  # pragma: no cover - runtime dependent
            raise DbAdapterError(
                "slatedb package is required for reading shards"
            ) from exc

        # ObjectStore.resolve and DbReaderBuilder construction are sync
        # \u2014 do them up front so the bridge hop only carries the async
        # build() coroutines. The env file context covers all of them.
        with resolve_env_file(self.env_file, self.credential_provider) as env_path:
            with apply_env_file(env_path):
                builders = []
                for db_url, _manifest in specs:
                    store = object_store_cls.resolve(db_url)
                    builders.append(db_reader_builder_cls("", store))
                raw_readers = gather_sync([b.build() for b in builders])

        return [
            _SlateDbReaderHandle(r, iterator_chunk_size=self.iterator_chunk_size)
            for r in raw_readers
        ]


def _shard_details_from_router(router: SnapshotRouter) -> list[ShardDetail]:
    """Extract per-shard metadata from a router's shard list."""
    return [
        ShardDetail(
            db_id=s.db_id,
            row_count=s.row_count,
            min_key=s.min_key,
            max_key=s.max_key,
            db_url=s.db_url,
        )
        for s in router.shards
    ]
