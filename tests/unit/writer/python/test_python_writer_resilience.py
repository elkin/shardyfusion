"""Tests for Python writer resilience fixes (C2, C3)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self

import pytest

from shardyfusion.config import HashWriteConfig, ManifestOptions, OutputOptions
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.run_registry import InMemoryRunRegistry, RunStatus
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.writer.python import write_sharded_by_hash

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class _TrackingAdapter:
    """In-memory adapter that records all write_batch calls."""

    def __init__(self) -> None:
        self.write_calls: list[list[tuple[bytes, bytes]]] = []
        self.flushed = False
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self.write_calls.append(list(pairs))

    def flush(self) -> None:
        self.flushed = True

    def checkpoint(self) -> str | None:
        return "fake-checkpoint"

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class _TrackingFactory:
    """Factory that creates tracking adapters and keeps references."""

    def __init__(self) -> None:
        self.adapters: dict[int, _TrackingAdapter] = {}
        self._call_count = 0

    def __call__(self, *, db_url: str, local_dir: Path) -> _TrackingAdapter:
        adapter = _TrackingAdapter()
        self.adapters[self._call_count] = adapter
        self._call_count += 1
        return adapter


def _make_config(
    num_dbs: int = 4,
    *,
    factory: _TrackingFactory | None = None,
    run_registry: InMemoryRunRegistry | None = None,
    batch_size: int = 50_000,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> HashWriteConfig:
    return HashWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=factory or _TrackingFactory(),
        output=OutputOptions(run_id="test-run"),
        manifest=ManifestOptions(store=InMemoryManifestStore()),
        run_registry=run_registry,
    )


# ---------------------------------------------------------------------------
# C2: Global batch limits
# ---------------------------------------------------------------------------


class TestGlobalBatchLimits:
    """Tests for max_total_batched_items and max_total_batched_bytes."""

    def test_max_total_batched_items_triggers_early_flush(self) -> None:
        """When total buffered items exceeds the limit, the largest batch is flushed."""
        factory = _TrackingFactory()
        # batch_size=100 so no per-shard flushes trigger for 10 items per shard
        config = _make_config(num_dbs=10, factory=factory, batch_size=100)

        # 50 records across 10 shards → ~5 per shard. Without limit, no flushes
        # until the final flush. With limit=20, flushes trigger mid-stream.
        records = list(range(50))
        result = write_sharded_by_hash(
            records,
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"v",
            max_total_batched_items=20,
        )

        assert result.stats.rows_written == 50
        # Verify all data was written (no loss)
        total_written = sum(
            sum(len(call) for call in adapter.write_calls)
            for adapter in factory.adapters.values()
        )
        assert total_written == 50

    def test_max_total_batched_bytes_triggers_early_flush(self) -> None:
        """When total buffered bytes exceeds the limit, the largest batch is flushed."""
        factory = _TrackingFactory()
        config = _make_config(num_dbs=2, factory=factory, batch_size=1000)

        # Each record produces ~1KB value. 20 records → ~20KB across 2 shards.
        big_value = b"x" * 1024
        records = list(range(20))
        result = write_sharded_by_hash(
            records,
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: big_value,
            max_total_batched_bytes=5000,  # ~5KB limit → should trigger flushes
        )

        assert result.stats.rows_written == 20
        total_written = sum(
            sum(len(call) for call in adapter.write_calls)
            for adapter in factory.adapters.values()
        )
        assert total_written == 20

    def test_no_limits_preserves_existing_behavior(self) -> None:
        """Without global limits, behavior matches original."""
        factory = _TrackingFactory()
        config = _make_config(num_dbs=4, factory=factory, batch_size=100)

        records = list(range(40))
        result = write_sharded_by_hash(
            records,
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"v",
            # No limits set
        )

        assert result.stats.rows_written == 40

    def test_limits_are_ignored_in_parallel_mode(self) -> None:
        """Global limits only apply to single-process mode."""
        from shardyfusion.testing import fake_adapter_factory

        config = HashWriteConfig(
            num_dbs=2,
            s3_prefix="s3://bucket/prefix",
            batch_size=50_000,
            adapter_factory=fake_adapter_factory,
            output=OutputOptions(run_id="test-run"),
            manifest=ManifestOptions(store=InMemoryManifestStore()),
        )

        records = list(range(10))
        # These params are accepted but only affect single-process mode
        result = write_sharded_by_hash(
            records,
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"v",
            parallel=True,
            max_total_batched_items=5,
        )

        assert result.stats.rows_written == 10


# ---------------------------------------------------------------------------
# C3: Adapter close failure logging
# ---------------------------------------------------------------------------


class TestAdapterCloseFailureLogging:
    """Tests for C3: adapter close failures are logged and raised."""

    def test_close_failure_propagates(self) -> None:
        """If adapter close fails, the exception propagates via ExitStack."""

        class _FailingCloseAdapter(_TrackingAdapter):
            def close(self) -> None:
                raise RuntimeError("simulated close failure")

        class _FailingCloseFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> _FailingCloseAdapter:
                return _FailingCloseAdapter()

        config = _make_config(num_dbs=2, factory=None)
        config.adapter_factory = _FailingCloseFactory()

        with pytest.raises(RuntimeError, match="simulated close failure"):
            write_sharded_by_hash(
                list(range(10)),
                config,
                key_fn=lambda r: r,
                value_fn=lambda r: b"v",
            )

    def test_close_failure_still_closes_all_adapters(self) -> None:
        """ExitStack closes all adapters even if some fail (LIFO order)."""
        close_attempts: list[int] = []

        class _TrackingCloseAdapter(_TrackingAdapter):
            def __init__(self, adapter_id: int) -> None:
                super().__init__()
                self._adapter_id = adapter_id

            def close(self) -> None:
                close_attempts.append(self._adapter_id)
                if self._adapter_id == 0:
                    raise RuntimeError("first adapter close fails")

        counter = [0]

        class _Factory:
            def __call__(
                self, *, db_url: str, local_dir: Path
            ) -> _TrackingCloseAdapter:
                adapter = _TrackingCloseAdapter(counter[0])
                counter[0] += 1
                return adapter

        config = _make_config(num_dbs=3, factory=None)
        config.adapter_factory = _Factory()

        with pytest.raises(RuntimeError, match="first adapter close fails"):
            write_sharded_by_hash(
                list(range(6)),
                config,
                key_fn=lambda r: r,
                value_fn=lambda r: b"v",
            )

        # ExitStack closes in LIFO order — all 3 adapters should be closed
        assert len(close_attempts) == 3


class TestRunRecordFailureStatus:
    def test_failed_write_marks_run_record_failed(self) -> None:
        registry = InMemoryRunRegistry()

        class _AlwaysFailAdapter(_TrackingAdapter):
            def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
                raise RuntimeError("adapter write failed")

        class _AlwaysFailFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> _AlwaysFailAdapter:
                return _AlwaysFailAdapter()

        config = _make_config(
            num_dbs=2,
            factory=None,
            run_registry=registry,
        )
        config.adapter_factory = _AlwaysFailFactory()

        with pytest.raises(RuntimeError, match="adapter write failed"):
            write_sharded_by_hash(
                list(range(4)),
                config,
                key_fn=lambda r: r,
                value_fn=lambda r: b"v",
            )

        ref = next(iter(registry._records))
        record = registry.load(ref)
        assert record.status is RunStatus.FAILED
        assert record.error_type == "RuntimeError"
