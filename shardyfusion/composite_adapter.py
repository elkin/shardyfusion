"""Composite adapter — KV + vector sidecar in one lifecycle.

Wraps a ``DbAdapter`` (KV storage, e.g. SlateDB) alongside a
``VectorIndexWriter`` (e.g. USearch HNSW) so that both share
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

from .config import VectorSpec, vector_metric_to_str
from .errors import ShardyfusionError
from .logging import get_logger, log_event
from .slatedb_adapter import DbAdapter, DbAdapterFactory
from .vector.types import SearchResult, VectorIndexWriter, VectorIndexWriterFactory

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
        vector_factory: Factory for the vector writer (e.g. ``USearchWriterFactory``).
        vector_spec: Vector configuration from ``WriteConfig``.
    """

    kv_factory: DbAdapterFactory
    vector_factory: VectorIndexWriterFactory
    vector_spec: VectorSpec

    def __call__(self, *, db_url: str, local_dir: Path) -> CompositeAdapter:
        from .vector.config import VectorIndexConfig
        from .vector.types import DistanceMetric

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

    def checkpoint(self) -> str | None:
        kv_ckpt = self._kv.checkpoint()
        self._vec.checkpoint()
        # Return KV checkpoint ID as the primary identifier
        return kv_ckpt

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._vec.close()
        finally:
            try:
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
        vector_factory: Factory for vector shard readers (e.g. ``USearchReaderFactory``).
        vector_spec: Vector configuration.
    """

    kv_factory: Any  # ShardReaderFactory protocol
    vector_factory: Any  # VectorShardReaderFactory protocol
    vector_spec: VectorSpec

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None = None,
    ) -> CompositeShardReader:
        from .vector.config import VectorIndexConfig
        from .vector.types import DistanceMetric

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
        )

        vector_dir = local_dir / "vector"
        vector_dir.mkdir(parents=True, exist_ok=True)
        vector_reader = self.vector_factory(
            db_url=f"{db_url.rstrip('/')}/vector",
            local_dir=vector_dir,
            index_config=index_config,
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
        kv_reader: Any,  # ShardReader protocol (get + close)
        vector_reader: Any,  # VectorShardReader protocol (search + close)
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
        ef: int = 50,
    ) -> list[SearchResult]:
        """Vector similarity search."""
        return self._vec.search(query, top_k, ef=ef)

    def close(self) -> None:
        try:
            self._vec.close()
        finally:
            self._kv.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
