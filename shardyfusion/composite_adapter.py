"""Composite adapter — KV + vector sidecar in one lifecycle.

Wraps a ``DbAdapter`` (KV storage, e.g. SlateDB) alongside a
``VectorIndexWriter`` (e.g. LanceDB HNSW) so that both share
the same shard prefix and are managed as a single unit.

Writer side:
    ``CompositeFactory`` / ``CompositeAdapter`` — delegates ``write_batch``
    to the KV adapter and ``write_vector_batch`` to the vector writer.
    ``checkpoint()`` and ``close()`` call both in sequence.

Reader side:
    ``CompositeReaderFactory`` / ``CompositeShardReader`` — combines
    a KV ``ShardReader`` with a ``VectorShardReader`` for unified
    ``get()`` + ``search()`` on one shard.
"""

from __future__ import annotations

import logging
import types as _types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from ._adapter import DbAdapter, DbAdapterFactory
from .config import VectorSpec, vector_metric_to_str
from .errors import ShardyfusionError
from .logging import get_logger, log_event
from .type_defs import (
    AsyncShardReader,
    AsyncShardReaderFactory,
    Manifest,
    ShardReader,
    ShardReaderFactory,
)
from .vector.config import VectorIndexConfig
from .vector.types import (
    AsyncVectorShardReader,
    AsyncVectorShardReaderFactory,
    DistanceMetric,
    SearchResult,
    VectorIndexWriter,
    VectorIndexWriterFactory,
    VectorShardReader,
    VectorShardReaderFactory,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CompositeAdapterError(ShardyfusionError):
    """Composite adapter lifecycle error (non-retryable)."""

    retryable = False


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompositeFactory:
    """Factory that creates unified KV + vector sidecar adapters.

    Args:
        kv_factory: Factory for the KV adapter (e.g. ``SlateDbFactory``).
        vector_factory: Factory for the vector writer (e.g. ``LanceDBWriterFactory``).
        vector_spec: Vector configuration from a KV writer config.
    """

    kv_factory: DbAdapterFactory
    vector_factory: VectorIndexWriterFactory
    vector_spec: VectorSpec

    def vec_payload_bytes_in_kv_db(self) -> int:
        """Per-row embedding payload stored in the inner kv factory's .db.

        Returns 0 unconditionally.  :class:`CompositeAdapter.write_vector_batch`
        routes embeddings exclusively to ``vector_factory`` (the sidecar);
        the inner ``kv_factory`` only receives ``write_batch`` (KV pairs)
        — so even when the inner happens to be vector-aware
        (e.g. :class:`SqliteVecFactory`), its own ``vec_index`` table is
        never populated under this composite.  Delegating to the inner
        would over-budget the page-size picker for an empty cell.
        """
        return 0

    def __call__(self, *, db_url: str, local_dir: Path) -> CompositeAdapter:
        index_config = VectorIndexConfig(
            dim=self.vector_spec.dim,
            metric=DistanceMetric(vector_metric_to_str(self.vector_spec.metric)),
            index_type=self.vector_spec.index_type,
            index_params=self.vector_spec.index_params,
            quantization=self.vector_spec.quantization,
        )

        kv_adapter = self.kv_factory(db_url=db_url, local_dir=local_dir)

        # Vector writer gets a subdirectory to avoid file collisions
        vector_dir = local_dir / "vector"
        vector_dir.mkdir(parents=True, exist_ok=True)
        vector_writer = self.vector_factory(
            db_url=f"{db_url.rstrip('/')}/vector",
            local_dir=vector_dir,
            index_config=index_config,
        )

        return CompositeAdapter(
            kv_adapter=kv_adapter,
            vector_writer=vector_writer,
        )


class CompositeAdapter:
    """Unified KV + vector write adapter with sidecar vector index.

    Delegates KV operations to the wrapped ``DbAdapter`` and vector
    operations to the wrapped ``VectorIndexWriter``.  Lifecycle methods
    (flush, checkpoint, close) call both in sequence.
    """

    def __init__(
        self,
        *,
        kv_adapter: DbAdapter,
        vector_writer: VectorIndexWriter,
    ) -> None:
        self._kv = kv_adapter
        self._vec = vector_writer
        self._closed = False
        # Set when ``seal()`` raises in ``_kv.seal()`` so ``close()`` can
        # skip the vector sidecar upload — LanceDB.close() uploads
        # unconditionally (no _sealed gate), and shipping the sidecar
        # without a matching kv .db would leave an orphan in S3.
        self._seal_failed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _types.TracebackType | None,
    ) -> None:
        self.close()

    # -- KV operations (DbAdapter protocol) --

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        """Write KV pairs to the underlying KV adapter."""
        if self._closed:
            raise CompositeAdapterError("Adapter already closed")
        self._kv.write_batch(pairs)

    # -- Vector operations --

    def write_vector_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write vectors to the sidecar vector index."""
        if self._closed:
            raise CompositeAdapterError("Adapter already closed")
        self._vec.add_batch(ids, vectors, payloads)

    # -- Lifecycle --

    def flush(self) -> None:
        self._kv.flush()
        self._vec.flush()

    def seal(self) -> None:
        """Seal both halves. The composite has no separate identity:
        the writer stamps a single UUID covering KV + vector together
        (see :func:`shardyfusion._checkpoint_id.generate_checkpoint_id`).

        If ``_kv.seal()`` raises, ``_vec.seal()`` is skipped AND
        ``self._seal_failed`` is set so the subsequent ``close()`` skips
        the vector sidecar upload too — otherwise LanceDB's
        unconditional upload would leave an orphan in S3.
        """
        try:
            self._kv.seal()
        except Exception:
            self._seal_failed = True
            raise
        self._vec.seal()

    def db_bytes(self) -> int:
        return int(self._kv.db_bytes()) + int(self._vec.db_bytes())

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._seal_failed:
                # Intentionally do NOT call self._vec.close().  LanceDB's
                # close() uploads the local .lance dataset unconditionally
                # (no _sealed gate — see lancedb_adapter.py), so calling
                # it after a failed kv seal would publish a sidecar with
                # no matching kv .db and leave an orphan in S3.  The
                # local LanceDB handle is dropped on process exit; the
                # cost is acceptable for the failed-shard path.
                log_event(
                    "composite_adapter_skip_vec_after_kv_seal_failure",
                    level=logging.WARNING,
                    logger=_logger,
                )
            else:
                self._vec.close()
        finally:
            try:
                # ``_kv.close()`` will raise from its skip-upload branch
                # when seal previously failed (see SqliteAdapter.close());
                # we let that propagate so the writer learns the shard
                # was not published.
                self._kv.close()
            finally:
                self._closed = True
        log_event(
            "composite_adapter_closed",
            level=logging.DEBUG,
            logger=_logger,
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompositeReaderFactory:
    """Factory that creates unified KV + vector shard readers.

    Args:
        kv_factory: Factory for KV shard readers (e.g. ``SlateDbReaderFactory``).
        vector_factory: Factory for vector shard readers (e.g. ``LanceDBReaderFactory``).
        vector_spec: Vector configuration.
    """

    kv_factory: ShardReaderFactory
    vector_factory: VectorShardReaderFactory
    vector_spec: VectorSpec

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None = None,
        manifest: Manifest,
    ) -> CompositeShardReader:
        index_config = VectorIndexConfig(
            dim=self.vector_spec.dim,
            metric=DistanceMetric(vector_metric_to_str(self.vector_spec.metric)),
            index_type=self.vector_spec.index_type,
            index_params=self.vector_spec.index_params,
            quantization=self.vector_spec.quantization,
        )

        kv_reader = self.kv_factory(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
        )

        vector_dir = local_dir / "vector"
        vector_dir.mkdir(parents=True, exist_ok=True)
        vector_reader = self.vector_factory(
            db_url=f"{db_url.rstrip('/')}/vector",
            local_dir=vector_dir,
            index_config=index_config,
            manifest=manifest,
        )

        return CompositeShardReader(
            kv_reader=kv_reader,
            vector_reader=vector_reader,
        )


class CompositeShardReader:
    """Unified shard reader: KV point lookups + vector search.

    Wraps a KV ``ShardReader`` and a ``VectorShardReader`` to support
    both ``get()`` and ``search()`` on the same shard.
    """

    def __init__(
        self,
        *,
        kv_reader: ShardReader,
        vector_reader: VectorShardReader,
    ) -> None:
        self._kv = kv_reader
        self._vec = vector_reader

    def get(self, key: bytes) -> bytes | None:
        """KV point lookup."""
        return self._kv.get(key)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Vector similarity search."""
        return self._vec.search(query, top_k)

    def close(self) -> None:
        try:
            self._vec.close()
        finally:
            self._kv.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(slots=True)
class AsyncCompositeReaderFactory:
    """Factory that creates async unified KV + vector shard readers.

    Args:
        kv_factory: Factory for KV async shard readers.
        vector_factory: Factory for vector async shard readers.
        vector_spec: Vector configuration.
    """

    kv_factory: AsyncShardReaderFactory
    vector_factory: AsyncVectorShardReaderFactory
    vector_spec: VectorSpec

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None = None,
        manifest: Manifest,
    ) -> AsyncCompositeShardReader:
        index_config = VectorIndexConfig(
            dim=self.vector_spec.dim,
            metric=DistanceMetric(vector_metric_to_str(self.vector_spec.metric)),
            index_type=self.vector_spec.index_type,
            index_params=self.vector_spec.index_params,
            quantization=self.vector_spec.quantization,
        )

        kv_reader = await self.kv_factory(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
        )

        vector_dir = local_dir / "vector"
        vector_dir.mkdir(parents=True, exist_ok=True)
        vector_reader = await self.vector_factory(
            db_url=f"{db_url.rstrip('/')}/vector",
            local_dir=vector_dir,
            index_config=index_config,
            manifest=manifest,
        )

        return AsyncCompositeShardReader(
            kv_reader=kv_reader,
            vector_reader=vector_reader,
        )


class AsyncCompositeShardReader:
    """Async unified shard reader: KV point lookups + vector search.

    Wraps an async KV ``AsyncShardReader`` and an ``AsyncVectorShardReader`` to support
    both ``get()`` and ``search()`` on the same shard.
    """

    def __init__(
        self,
        *,
        kv_reader: AsyncShardReader,
        vector_reader: AsyncVectorShardReader,
    ) -> None:
        self._kv = kv_reader
        self._vec = vector_reader

    async def get(self, key: bytes) -> bytes | None:
        """KV point lookup."""
        return await self._kv.get(key)

    async def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Vector similarity search."""
        return await self._vec.search(query, top_k)

    async def close(self) -> None:
        try:
            await self._vec.close()
        finally:
            await self._kv.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
