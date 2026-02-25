"""Tests for the pure-Python iterator writer."""

from __future__ import annotations

import pathlib
import time
from collections.abc import Iterable
from typing import Self

import pytest

from slatedb_spark_sharded._rate_limiter import TokenBucket
from slatedb_spark_sharded._writer_core import _route_key
from slatedb_spark_sharded.config import ManifestOptions, OutputOptions, WriteConfig
from slatedb_spark_sharded.errors import ConfigValidationError
from slatedb_spark_sharded.manifest import BuildResult, ManifestArtifact
from slatedb_spark_sharded.publish import ManifestPublisher
from slatedb_spark_sharded.serde import make_key_encoder
from slatedb_spark_sharded.sharding_types import (
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)
from slatedb_spark_sharded.slatedb_adapter import DbAdapterFactory
from slatedb_spark_sharded.testing import (
    fake_adapter_factory,
    file_backed_adapter_factory,
    file_backed_load_db,
)
from slatedb_spark_sharded.writer.python import write_sharded

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


# ---------------------------------------------------------------------------
# Parallel writer tests
# ---------------------------------------------------------------------------


def _make_parallel_config(
    tmp_path: pathlib.Path,
    adapter_factory: DbAdapterFactory,
    *,
    num_dbs: int = 4,
    batch_size: int = 50_000,
    sharding: ShardingSpec | None = None,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> WriteConfig:
    return WriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=adapter_factory,
        sharding=sharding or ShardingSpec(),
        output=OutputOptions(
            run_id="test-run",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(publisher=InMemoryPublisher()),
    )


def test_parallel_hash_routing_round_trip(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    factory = file_backed_adapter_factory(root_dir)
    config = _make_parallel_config(tmp_path, factory, num_dbs=4)

    records = list(range(40))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
        parallel=True,
    )

    assert isinstance(result, BuildResult)
    assert len(result.winners) == 4
    assert sorted(w.db_id for w in result.winners) == [0, 1, 2, 3]
    assert sum(w.row_count for w in result.winners) == 40
    assert result.manifest_ref.startswith("mem://manifests/")
    assert result.current_ref == "mem://_CURRENT"

    # Verify data integrity via file-backed reads
    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)
    for r in records:
        key_bytes = make_key_encoder(config.key_encoding)(r)
        assert key_bytes in all_kv
        assert all_kv[key_bytes] == f"v{r}".encode()


def test_parallel_range_explicit_boundaries(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    factory = file_backed_adapter_factory(root_dir)
    config = _make_parallel_config(
        tmp_path,
        factory,
        num_dbs=3,
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
        parallel=True,
    )

    assert len(result.winners) == 3
    assert result.winners[0].row_count == 10
    assert result.winners[1].row_count == 10
    assert result.winners[2].row_count == 10


def test_parallel_empty_input(tmp_path: pathlib.Path) -> None:
    config = _make_parallel_config(tmp_path, fake_adapter_factory, num_dbs=4)

    result = write_sharded(
        [],
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
        parallel=True,
    )

    assert len(result.winners) == 4
    assert all(w.row_count == 0 for w in result.winners)


def test_parallel_min_max_key_tracking(tmp_path: pathlib.Path) -> None:
    config = _make_parallel_config(
        tmp_path,
        fake_adapter_factory,
        num_dbs=2,
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
        parallel=True,
    )

    shard0 = next(w for w in result.winners if w.db_id == 0)
    shard1 = next(w for w in result.winners if w.db_id == 1)
    assert shard0.min_key == 10
    assert shard0.max_key == 30
    assert shard1.min_key == 60
    assert shard1.max_key == 80


def test_parallel_batch_flushing(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    factory = file_backed_adapter_factory(root_dir)
    config = _make_parallel_config(tmp_path, factory, num_dbs=1, batch_size=3)

    records = list(range(7))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
        parallel=True,
    )

    assert result.stats.rows_written == 7

    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)
    assert len(all_kv) == 7


def test_parallel_data_integrity(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    factory = file_backed_adapter_factory(root_dir)
    config = _make_parallel_config(tmp_path, factory, num_dbs=4)

    records = list(range(100))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"value-{r}".encode(),
        parallel=True,
    )

    assert result.stats.rows_written == 100

    # Collect all written data across all shards
    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)

    assert len(all_kv) == 100
    for r in records:
        key_bytes = make_key_encoder(config.key_encoding)(r)
        assert key_bytes in all_kv
        assert all_kv[key_bytes] == f"value-{r}".encode()

    # Verify each key is in the correct shard
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        for key_bytes in shard_data:
            key_int = int.from_bytes(key_bytes, byteorder="big", signed=False)
            expected_db_id = _route_key(
                key_int,
                num_dbs=config.num_dbs,
                sharding=config.sharding,
                key_encoding=config.key_encoding,
            )
            assert expected_db_id == winner.db_id


def test_parallel_single_shard(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    factory = file_backed_adapter_factory(root_dir)
    config = _make_parallel_config(tmp_path, factory, num_dbs=1)

    records = list(range(20))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
        parallel=True,
    )

    assert len(result.winners) == 1
    assert result.winners[0].db_id == 0
    assert result.winners[0].row_count == 20

    shard_data = file_backed_load_db(root_dir, result.winners[0].db_url)
    assert len(shard_data) == 20


def test_parallel_uneven_distribution(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    factory = file_backed_adapter_factory(root_dir)
    config = _make_parallel_config(tmp_path, factory, num_dbs=8)

    records = list(range(5))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
        parallel=True,
    )

    assert len(result.winners) == 8
    assert sum(w.row_count for w in result.winners) == 5

    # Some shards should be empty
    empty_shards = [w for w in result.winners if w.row_count == 0]
    assert len(empty_shards) >= 1

    # Verify all 5 records are recoverable
    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)
    assert len(all_kv) == 5


def test_parallel_rate_limited_write(tmp_path: pathlib.Path) -> None:
    config = _make_parallel_config(tmp_path, fake_adapter_factory, num_dbs=2)

    records = list(range(10))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
        parallel=True,
        max_writes_per_second=1000.0,
    )

    assert result.stats.rows_written == 10
