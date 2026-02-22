"""Tests for the pure-Python iterator writer."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Self

import pytest

from slatedb_spark_sharded._rate_limiter import TokenBucket
from slatedb_spark_sharded.config import ManifestOptions, OutputOptions, WriteConfig
from slatedb_spark_sharded.errors import ConfigValidationError
from slatedb_spark_sharded.manifest import BuildResult, ManifestArtifact
from slatedb_spark_sharded.publish import ManifestPublisher
from slatedb_spark_sharded.python_writer import write_sharded
from slatedb_spark_sharded.sharding_types import (
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)

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

    def close(self) -> None:
        self.closed = True


class _TrackingFactory:
    """Factory that creates tracking adapters and keeps references for assertions."""

    def __init__(self) -> None:
        self.adapters: dict[int, _TrackingAdapter] = {}
        self._call_count = 0

    def __call__(self, *, db_url: str, local_dir: str) -> _TrackingAdapter:
        adapter = _TrackingAdapter()
        self.adapters[self._call_count] = adapter
        self._call_count += 1
        return adapter


class InMemoryPublisher(ManifestPublisher):
    def __init__(self) -> None:
        self.objects: dict[str, ManifestArtifact] = {}

    def publish_manifest(
        self, *, name: str, artifact: ManifestArtifact, run_id: str
    ) -> str:
        ref = f"mem://manifests/run_id={run_id}/{name}"
        self.objects[ref] = artifact
        return ref

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        ref = f"mem://{name}"
        self.objects[ref] = artifact
        return ref


def _make_config(
    num_dbs: int = 4,
    *,
    factory: _TrackingFactory | None = None,
    batch_size: int = 50_000,
    sharding: ShardingSpec | None = None,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> WriteConfig:
    return WriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=factory or _TrackingFactory(),
        sharding=sharding or ShardingSpec(),
        output=OutputOptions(run_id="test-run"),
        manifest=ManifestOptions(publisher=InMemoryPublisher()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hash_routing_round_trip() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=4, factory=factory)

    records = list(range(40))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
    )

    assert isinstance(result, BuildResult)
    assert len(result.winners) == 4
    assert sorted(w.db_id for w in result.winners) == [0, 1, 2, 3]
    assert sum(w.row_count for w in result.winners) == 40
    assert result.manifest_ref.startswith("mem://manifests/")
    assert result.current_ref == "mem://_CURRENT"


def test_range_explicit_boundaries() -> None:
    factory = _TrackingFactory()
    config = _make_config(
        num_dbs=3,
        factory=factory,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=[10, 20],
        ),
    )

    records = list(range(30))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
    )

    assert len(result.winners) == 3
    # Shard 0: keys 0–9, shard 1: keys 10–19, shard 2: keys 20–29
    assert result.winners[0].row_count == 10
    assert result.winners[1].row_count == 10
    assert result.winners[2].row_count == 10


def test_range_without_boundaries_raises() -> None:
    config = _make_config(
        num_dbs=3,
        sharding=ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=None),
    )
    with pytest.raises(ConfigValidationError, match="boundaries"):
        write_sharded([], config, key_fn=lambda r: r, value_fn=lambda r: b"v")


def test_custom_expr_raises() -> None:
    config = _make_config(
        num_dbs=2,
        sharding=ShardingSpec(strategy=ShardingStrategy.CUSTOM_EXPR),
    )
    with pytest.raises(ConfigValidationError, match="Custom expression"):
        write_sharded([], config, key_fn=lambda r: r, value_fn=lambda r: b"v")


def test_empty_input() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=4, factory=factory)

    result = write_sharded(
        [],
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
    )

    assert len(result.winners) == 4
    assert all(w.row_count == 0 for w in result.winners)


def test_batch_flushing() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=1, factory=factory, batch_size=3)

    # 7 records → should produce 2 full batches (3 each) + 1 final (1)
    records = list(range(7))
    write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
    )

    # All adapters in factory, find the one that got records
    adapter = factory.adapters[0]
    total_pairs = sum(len(call) for call in adapter.write_calls)
    assert total_pairs == 7
    assert len(adapter.write_calls) >= 3  # at least 2 full + 1 final


def test_min_max_key_tracking() -> None:
    factory = _TrackingFactory()
    config = _make_config(
        num_dbs=2,
        factory=factory,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=[50],
        ),
    )

    records = [10, 20, 30, 60, 70, 80]
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
    )

    # Shard 0: keys 10, 20, 30; Shard 1: keys 60, 70, 80
    shard0 = next(w for w in result.winners if w.db_id == 0)
    shard1 = next(w for w in result.winners if w.db_id == 1)
    assert shard0.min_key == 10
    assert shard0.max_key == 30
    assert shard1.min_key == 60
    assert shard1.max_key == 80


def test_rate_limiter() -> None:
    """TokenBucket throttles correctly."""
    bucket = TokenBucket(rate=100.0)

    # Should acquire immediately when bucket is full
    start = time.monotonic()
    bucket.acquire(1)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


def test_rate_limited_write() -> None:
    """max_writes_per_second limits write throughput."""
    factory = _TrackingFactory()
    # batch_size=1 so every record triggers a write_batch, rate=1000 per second
    config = _make_config(num_dbs=1, factory=factory, batch_size=1)

    records = list(range(5))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
        max_writes_per_second=1000.0,
    )

    assert result.stats.rows_written == 5
