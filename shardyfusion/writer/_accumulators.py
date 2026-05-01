"""Composable shard batch accumulators for KV and unified KV+vector writes."""

from __future__ import annotations

import time
from typing import Any, Protocol

from .._rate_limiter import RateLimiter
from .._writer_core import update_min_max
from ..errors import ShardWriteError
from ..logging import get_logger
from ..metrics._events import MetricEvent
from ..metrics._protocol import MetricsCollector
from ..type_defs import KeyInput

_logger = get_logger(__name__)

VectorWriteTuple = tuple[int | str, Any, dict[str, Any] | None]


def _flush_batch(
    batch: list[tuple[bytes, bytes]],
    adapter: object,
    *,
    ops_limiter: RateLimiter | None,
    bytes_limiter: RateLimiter | None,
    metrics_collector: MetricsCollector | None,
    started: float,
) -> None:
    """Flush one accumulated batch through rate limiters and adapter."""
    if ops_limiter is not None:
        ops_limiter.acquire(len(batch))
    if bytes_limiter is not None:
        bytes_limiter.acquire(sum(len(k) + len(v) for k, v in batch))
    adapter.write_batch(batch)  # type: ignore[union-attr]
    if metrics_collector is not None:
        metrics_collector.emit(
            MetricEvent.BATCH_WRITTEN,
            {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "batch_size": len(batch),
            },
        )
    batch.clear()


def _flush_vector_batch(
    vector_ids: list[int | str],
    vector_data: list[Any],
    vector_payloads: list[dict[str, Any] | None],
    adapter: object,
) -> None:
    """Flush accumulated vector batch through the adapter."""
    if not vector_ids:
        return
    vector_writer = getattr(adapter, "write_vector_batch", None)
    if not callable(vector_writer):
        raise ShardWriteError("Adapter does not support vector writes in unified mode")
    import numpy as np  # type: ignore

    ids_arr = np.array(vector_ids)
    vectors_arr = np.array(vector_data, dtype=np.float32)
    vector_writer(ids_arr, vectors_arr, vector_payloads)
    vector_ids.clear()
    vector_data.clear()
    vector_payloads.clear()


class ShardBatchAccumulator(Protocol):
    """Protocol for pluggable shard batch accumulators."""

    def add(
        self,
        key_value: object,
        key_bytes: bytes,
        value_bytes: bytes,
        vector_item: VectorWriteTuple | None,
    ) -> None:
        """Add one row to the accumulator."""
        ...

    def maybe_flush(
        self,
        adapter: object,
        *,
        batch_size: int,
        ops_limiter: RateLimiter | None,
        bytes_limiter: RateLimiter | None,
        metrics_collector: MetricsCollector | None,
        started: float,
    ) -> None:
        """Flush batches if they have reached batch_size."""
        ...

    def flush_remaining(
        self,
        adapter: object,
        *,
        ops_limiter: RateLimiter | None,
        bytes_limiter: RateLimiter | None,
        metrics_collector: MetricsCollector | None,
        started: float,
    ) -> None:
        """Flush any remaining buffered batches unconditionally."""
        ...

    @property
    def row_count(self) -> int:
        """Number of rows accumulated so far."""
        ...

    @property
    def min_key(self) -> KeyInput | None:
        """Minimum key seen so far."""
        ...

    @property
    def max_key(self) -> KeyInput | None:
        """Maximum key seen so far."""
        ...


class KvAccumulator:
    """Accumulator for pure KV writes (no vector batching)."""

    __slots__ = ("_batch", "_row_count", "_min_key", "_max_key")

    def __init__(self) -> None:
        self._batch: list[tuple[bytes, bytes]] = []
        self._row_count = 0
        self._min_key: KeyInput | None = None
        self._max_key: KeyInput | None = None

    def add(
        self,
        key_value: object,
        key_bytes: bytes,
        value_bytes: bytes,
        vector_item: VectorWriteTuple | None,
    ) -> None:
        self._batch.append((key_bytes, value_bytes))
        self._row_count += 1
        self._min_key, self._max_key = update_min_max(
            self._min_key,
            self._max_key,
            key_value,  # type: ignore[arg-type]
        )

    def maybe_flush(
        self,
        adapter: object,
        *,
        batch_size: int,
        ops_limiter: RateLimiter | None,
        bytes_limiter: RateLimiter | None,
        metrics_collector: MetricsCollector | None,
        started: float,
    ) -> None:
        if len(self._batch) >= batch_size:
            _flush_batch(
                self._batch,
                adapter,
                ops_limiter=ops_limiter,
                bytes_limiter=bytes_limiter,
                metrics_collector=metrics_collector,
                started=started,
            )

    def flush_remaining(
        self,
        adapter: object,
        *,
        ops_limiter: RateLimiter | None,
        bytes_limiter: RateLimiter | None,
        metrics_collector: MetricsCollector | None,
        started: float,
    ) -> None:
        if self._batch:
            _flush_batch(
                self._batch,
                adapter,
                ops_limiter=ops_limiter,
                bytes_limiter=bytes_limiter,
                metrics_collector=metrics_collector,
                started=started,
            )

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def min_key(self) -> KeyInput | None:
        return self._min_key

    @property
    def max_key(self) -> KeyInput | None:
        return self._max_key


class UnifiedAccumulator:
    """Accumulator for unified KV + vector writes."""

    __slots__ = (
        "_batch",
        "_vector_ids",
        "_vector_data",
        "_vector_payloads",
        "_row_count",
        "_min_key",
        "_max_key",
    )

    def __init__(self) -> None:
        self._batch: list[tuple[bytes, bytes]] = []
        self._vector_ids: list[int | str] = []
        self._vector_data: list[Any] = []
        self._vector_payloads: list[dict[str, Any] | None] = []
        self._row_count = 0
        self._min_key: KeyInput | None = None
        self._max_key: KeyInput | None = None

    def add(
        self,
        key_value: object,
        key_bytes: bytes,
        value_bytes: bytes,
        vector_item: VectorWriteTuple | None,
    ) -> None:
        self._batch.append((key_bytes, value_bytes))
        self._row_count += 1
        self._min_key, self._max_key = update_min_max(
            self._min_key,
            self._max_key,
            key_value,  # type: ignore[arg-type]
        )
        if vector_item is not None:
            vec_id, vec_data, vec_payload = vector_item
            self._vector_ids.append(vec_id)
            self._vector_data.append(vec_data)
            self._vector_payloads.append(vec_payload)

    def maybe_flush(
        self,
        adapter: object,
        *,
        batch_size: int,
        ops_limiter: RateLimiter | None,
        bytes_limiter: RateLimiter | None,
        metrics_collector: MetricsCollector | None,
        started: float,
    ) -> None:
        if len(self._batch) >= batch_size:
            _flush_batch(
                self._batch,
                adapter,
                ops_limiter=ops_limiter,
                bytes_limiter=bytes_limiter,
                metrics_collector=metrics_collector,
                started=started,
            )
            _flush_vector_batch(
                self._vector_ids,
                self._vector_data,
                self._vector_payloads,
                adapter,
            )

    def flush_remaining(
        self,
        adapter: object,
        *,
        ops_limiter: RateLimiter | None,
        bytes_limiter: RateLimiter | None,
        metrics_collector: MetricsCollector | None,
        started: float,
    ) -> None:
        if self._batch:
            _flush_batch(
                self._batch,
                adapter,
                ops_limiter=ops_limiter,
                bytes_limiter=bytes_limiter,
                metrics_collector=metrics_collector,
                started=started,
            )
        _flush_vector_batch(
            self._vector_ids,
            self._vector_data,
            self._vector_payloads,
            adapter,
        )

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def min_key(self) -> KeyInput | None:
        return self._min_key

    @property
    def max_key(self) -> KeyInput | None:
        return self._max_key
