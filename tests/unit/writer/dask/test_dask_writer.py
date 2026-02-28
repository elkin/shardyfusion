"""Tests for the Dask writer."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from typing import Self

import pandas as pd
import pytest

dd = pytest.importorskip("dask.dataframe")
import dask  # noqa: E402

from slatedb_spark_sharded._writer_core import route_key  # noqa: E402
from slatedb_spark_sharded.config import (  # noqa: E402
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from slatedb_spark_sharded.errors import ConfigValidationError  # noqa: E402
from slatedb_spark_sharded.manifest import BuildResult  # noqa: E402
from slatedb_spark_sharded.serde import ValueSpec, make_key_encoder  # noqa: E402
from slatedb_spark_sharded.sharding_types import (  # noqa: E402
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)
from slatedb_spark_sharded.testing import (  # noqa: E402
    file_backed_adapter_factory,
    file_backed_load_db,
)
from slatedb_spark_sharded.writer.dask import write_sharded_dask  # noqa: E402
from tests.helpers.tracking import InMemoryPublisher, RecordingTokenBucket  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _synchronous_scheduler():
    """Force synchronous Dask scheduler for deterministic test behavior."""
    with dask.config.set(scheduler="synchronous"):
        yield


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

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
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


def _make_file_backed_config(
    tmp_path: pathlib.Path,
    *,
    num_dbs: int = 4,
    batch_size: int = 50_000,
    sharding: ShardingSpec | None = None,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> tuple[WriteConfig, str]:
    root_dir = str(tmp_path / "file_backed")
    config = WriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=file_backed_adapter_factory(root_dir),
        sharding=sharding or ShardingSpec(),
        output=OutputOptions(
            run_id="test-run",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(publisher=InMemoryPublisher()),
    )
    return config, root_dir


def _make_dask_df(
    records: list[dict[str, object]], npartitions: int = 2
) -> dd.DataFrame:
    pdf = pd.DataFrame(records)
    return dd.from_pandas(pdf, npartitions=npartitions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hash_routing_round_trip() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=4, factory=factory)

    records = [{"id": i} for i in range(40)]
    ddf = _make_dask_df(records, npartitions=2)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: f"v{row['id']}".encode()),
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

    records = [{"id": i} for i in range(30)]
    ddf = _make_dask_df(records, npartitions=2)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert len(result.winners) == 3
    # Shard 0: keys 0-9, shard 1: keys 10-19, shard 2: keys 20-29
    assert result.winners[0].row_count == 10
    assert result.winners[1].row_count == 10
    assert result.winners[2].row_count == 10


def test_range_auto_computed_boundaries() -> None:
    factory = _TrackingFactory()
    config = WriteConfig(
        num_dbs=3,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=factory,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=None,
        ),
        output=OutputOptions(run_id="test-run"),
        manifest=ManifestOptions(publisher=InMemoryPublisher()),
    )

    # 90 evenly distributed records — quantiles should produce reasonable boundaries
    records = [{"id": i} for i in range(90)]
    ddf = _make_dask_df(records, npartitions=3)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert len(result.winners) == 3
    assert sum(w.row_count for w in result.winners) == 90
    # All three shards should have some records (boundaries computed from quantiles)
    assert all(w.row_count > 0 for w in result.winners)


def test_custom_expr_raises() -> None:
    config = _make_config(
        num_dbs=2,
        sharding=ShardingSpec(strategy=ShardingStrategy.CUSTOM_EXPR),
    )

    records = [{"id": i} for i in range(10)]
    ddf = _make_dask_df(records, npartitions=1)

    with pytest.raises(ConfigValidationError, match="Custom expression"):
        write_sharded_dask(
            ddf,
            config,
            key_col="id",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )


def test_empty_input() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=4, factory=factory)

    ddf = _make_dask_df([], npartitions=1)

    # Empty DataFrames need a schema for Dask
    pdf = pd.DataFrame({"id": pd.Series(dtype="int64")})
    ddf = dd.from_pandas(pdf, npartitions=1)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert len(result.winners) == 4
    assert all(w.row_count == 0 for w in result.winners)


def test_batch_flushing() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=1, factory=factory, batch_size=3)

    # 7 records all go to shard 0 (only 1 shard)
    records = [{"id": i} for i in range(7)]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    # Find the adapter that got records
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

    records = [{"id": k} for k in [10, 20, 30, 60, 70, 80]]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    # Shard 0: keys 10, 20, 30; Shard 1: keys 60, 70, 80
    shard0 = next(w for w in result.winners if w.db_id == 0)
    shard1 = next(w for w in result.winners if w.db_id == 1)
    assert shard0.min_key == 10
    assert shard0.max_key == 30
    assert shard1.min_key == 60
    assert shard1.max_key == 80


def test_rate_limited_write() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=1, factory=factory, batch_size=1)

    records = [{"id": i} for i in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=1000.0,
    )

    assert result.stats.rows_written == 5


# ---------------------------------------------------------------------------
# Rate-limiter integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _patch_token_bucket(monkeypatch: pytest.MonkeyPatch) -> list[RecordingTokenBucket]:
    RecordingTokenBucket.instances = []
    monkeypatch.setattr(
        "slatedb_spark_sharded.writer.dask.writer.TokenBucket",
        RecordingTokenBucket,
    )
    return RecordingTokenBucket.instances


def test_rate_limiter_bucket_created_with_correct_rate(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(num_dbs=1, batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=42.5,
    )

    assert len(_patch_token_bucket) == 1
    assert _patch_token_bucket[0].rate == 42.5


def test_rate_limiter_no_bucket_when_rate_is_none(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(num_dbs=1, batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert len(_patch_token_bucket) == 0


def test_rate_limiter_acquire_count_matches_batch_writes(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(num_dbs=1, batch_size=3)
    records = [{"id": i} for i in range(7)]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # 7 rows / batch_size 3 → batches of [3, 3, 1] → 3 acquire calls
    assert len(bucket.acquire_calls) == 3
    assert bucket.acquire_calls == [3, 3, 1]
    assert sum(bucket.acquire_calls) == 7


def test_rate_limiter_single_batch_single_acquire(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(num_dbs=1, batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # All 5 rows fit in one batch → single acquire for all 5
    assert bucket.acquire_calls == [5]


def test_rate_limiter_exact_batch_boundary(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(num_dbs=1, batch_size=3)
    records = [{"id": i} for i in range(6)]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # 6 rows / batch_size 3 → exactly 2 full batches, no trailing partial
    assert bucket.acquire_calls == [3, 3]


def test_rate_limiter_multiple_shards_independent_buckets(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(
        num_dbs=2,
        batch_size=1,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=[50],
        ),
    )
    records = [{"id": k} for k in [10, 20, 30, 60, 70, 80]]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    # Each shard gets its own bucket
    assert len(_patch_token_bucket) == 2
    # batch_size=1 → each row is its own batch → acquire(1) per row
    total_acquires = sum(len(b.acquire_calls) for b in _patch_token_bucket)
    assert total_acquires == 6
    # Each shard has 3 rows
    acquire_counts = sorted(len(b.acquire_calls) for b in _patch_token_bucket)
    assert acquire_counts == [3, 3]


def test_rate_limiter_empty_shard_no_bucket(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(
        num_dbs=4,
        batch_size=50_000,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=[100, 200, 300],
        ),
    )
    # Both keys < 100, so all go to shard 0
    records = [{"id": 0}, {"id": 1}]
    ddf = _make_dask_df(records, npartitions=1)

    write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    # Only 1 bucket created (only shard 0 has rows)
    assert len(_patch_token_bucket) == 1
    assert _patch_token_bucket[0].acquire_calls == [2]


def test_value_spec_binary_col(tmp_path: pathlib.Path) -> None:
    config, root_dir = _make_file_backed_config(tmp_path, num_dbs=2)

    records = [{"id": i, "payload": f"data-{i}".encode()} for i in range(10)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )

    assert result.stats.rows_written == 10

    # Verify written values are correct
    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)
    for i in range(10):
        key_bytes = make_key_encoder(config.key_encoding)(i)
        assert all_kv[key_bytes] == f"data-{i}".encode()


def test_value_spec_json_cols(tmp_path: pathlib.Path) -> None:
    config, root_dir = _make_file_backed_config(tmp_path, num_dbs=2)

    records = [{"id": i, "name": f"user{i}"} for i in range(10)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.json_cols(["id", "name"]),
    )

    assert result.stats.rows_written == 10

    # Verify at least one value is valid JSON with expected fields
    import json

    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)
    assert len(all_kv) == 10

    key_bytes = make_key_encoder(config.key_encoding)(0)
    parsed = json.loads(all_kv[key_bytes].decode("utf-8"))
    assert parsed["name"] == "user0"


def test_sort_within_partitions(tmp_path: pathlib.Path) -> None:
    config, root_dir = _make_file_backed_config(
        tmp_path,
        num_dbs=2,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=[50],
        ),
    )

    # Deliberately unsorted records
    records = [{"id": k} for k in [30, 10, 20, 80, 60, 70]]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        sort_within_partitions=True,
    )

    # Verify keys are sorted within each shard
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        keys = list(shard_data.keys())
        assert keys == sorted(keys), f"Shard {winner.db_id} keys not sorted"


def test_u32be_key_encoding(tmp_path: pathlib.Path) -> None:
    config, root_dir = _make_file_backed_config(
        tmp_path,
        num_dbs=2,
        key_encoding=KeyEncoding.U32BE,
    )

    records = [{"id": i} for i in range(20)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: f"v{row['id']}".encode()),
    )

    assert result.stats.rows_written == 20

    # Verify u32be encoding: each key should be 4 bytes
    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)
    assert len(all_kv) == 20
    assert all(len(k) == 4 for k in all_kv.keys())


def test_verify_routing_enabled() -> None:
    config = _make_config(num_dbs=4)

    records = [{"id": i} for i in range(20)]
    ddf = _make_dask_df(records, npartitions=2)

    # Should not raise — verification passes when routing is correct
    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        verify_routing=True,
    )

    assert result.stats.rows_written == 20


def test_data_integrity(tmp_path: pathlib.Path) -> None:
    config, root_dir = _make_file_backed_config(tmp_path, num_dbs=4)

    records = [{"id": i} for i in range(100)]
    ddf = _make_dask_df(records, npartitions=4)

    result = write_sharded_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(
            lambda row: f"value-{row['id']}".encode()
        ),
    )

    assert result.stats.rows_written == 100

    # Collect all written data across all shards
    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)

    assert len(all_kv) == 100
    for r in range(100):
        key_bytes = make_key_encoder(config.key_encoding)(r)
        assert key_bytes in all_kv
        assert all_kv[key_bytes] == f"value-{r}".encode()

    # Verify each key is in the correct shard
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        for key_bytes in shard_data:
            key_int = int.from_bytes(key_bytes, byteorder="big", signed=False)
            expected_db_id = route_key(
                key_int,
                num_dbs=config.num_dbs,
                sharding=config.sharding,
                key_encoding=config.key_encoding,
            )
            assert expected_db_id == winner.db_id
