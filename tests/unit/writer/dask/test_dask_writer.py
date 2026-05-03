"""Tests for the Dask writer."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self

import dask.dataframe as dd
import pandas as pd
import pytest

from shardyfusion._shard_writer import results_pdf_to_attempts
from shardyfusion._writer_core import ShardAttemptResult, route_hash
from shardyfusion.config import (
    HashShardedWriteConfig,
    WriterManifestConfig,
    WriterOutputConfig,
)
from shardyfusion.manifest import BuildResult, WriterInfo
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.metrics import MetricEvent
from shardyfusion.run_registry import InMemoryRunRegistry
from shardyfusion.serde import ValueSpec, make_key_encoder
from shardyfusion.sharding_types import (
    DB_ID_COL,
    KeyEncoding,
)
from shardyfusion.testing import (
    ListMetricsCollector,
    file_backed_adapter_factory,
    file_backed_load_db,
)
from shardyfusion.writer.dask import sharding as dask_sharding
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_in_memory_run_record,
)
from tests.helpers.tracking import (
    RecordingTokenBucket,
    patch_token_bucket_fixture,
)
from tests.helpers.writer_api import write_dask_hash_sharded as write_hash_sharded

_patch_token_bucket = patch_token_bucket_fixture("shardyfusion._shard_writer")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class _TrackingFactory:
    """Factory that creates tracking adapters and keeps references for assertions."""

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
) -> HashShardedWriteConfig:
    return HashShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=factory or _TrackingFactory(),
        output=WriterOutputConfig(run_id="test-run"),
        manifest=WriterManifestConfig(store=InMemoryManifestStore()),
        run_registry=run_registry,
    )


def _make_file_backed_config(
    tmp_path: Path,
    *,
    num_dbs: int = 4,
    batch_size: int = 50_000,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> tuple[HashShardedWriteConfig, str]:
    root_dir = str(tmp_path / "file_backed")
    config = HashShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=file_backed_adapter_factory(root_dir),
        output=WriterOutputConfig(
            run_id="test-run",
            local_root=str(tmp_path / "local"),
        ),
        manifest=WriterManifestConfig(store=InMemoryManifestStore()),
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

    result = write_hash_sharded(
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


def test_hash_routing_round_trip_records_succeeded_run_record() -> None:
    registry = InMemoryRunRegistry()
    config = _make_config(num_dbs=4, run_registry=registry)

    records = [{"id": i} for i in range(12)]
    ddf = _make_dask_df(records, npartitions=2)

    result = write_hash_sharded(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: f"v{row['id']}".encode()),
    )

    run_record = load_in_memory_run_record(registry, result)
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="dask",
        s3_prefix=config.s3_prefix,
    )


def test_empty_input() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=4, factory=factory)

    ddf = _make_dask_df([], npartitions=1)

    # Empty DataFrames need a schema for Dask
    pdf = pd.DataFrame({"id": pd.Series(dtype="int64")})
    ddf = dd.from_pandas(pdf, npartitions=1)

    result = write_hash_sharded(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    # No records → no non-empty shards in winners
    assert len(result.winners) == 0
    assert result.stats.rows_written == 0


def test_inferred_categorical_discovers_distinct_tokens_before_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pd.DataFrame({"region": ["ap", "ap", "eu", "ap", "eu"]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    captured: list[list[object]] = []

    def _capture(values):  # type: ignore[no-untyped-def]
        values_list = list(values)
        captured.append(values_list)
        return sorted(values_list)

    monkeypatch.setattr(dask_sharding, "build_categorical_routing_values", _capture)

    routing_values = dask_sharding._discover_categorical_routing_values(
        ddf,
        cel_expr="region",
        cel_columns={"region": "string"},
    )

    assert len(captured) == 1
    assert len(captured[0]) == 2
    assert set(captured[0]) == {"ap", "eu"}
    assert routing_values == ["ap", "eu"]


def test_batch_flushing() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=1, factory=factory, batch_size=3)

    # 7 records all go to shard 0 (only 1 shard)
    records = [{"id": i} for i in range(7)]
    ddf = _make_dask_df(records, npartitions=1)

    write_hash_sharded(
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


def test_write_partition_vector_fn_uses_distributed_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shardyfusion.writer.dask import writer as dask_writer

    captured_rows: list[
        tuple[object, bytes, tuple[int | str, object, dict[str, object] | None]]
    ] = []

    def _fake_distributed(*, db_id: int, rows_fn, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        rows = list(rows_fn())
        captured_rows.extend(rows)
        return ShardAttemptResult(
            db_id=db_id,
            db_url=f"s3://bucket/prefix/db={db_id:05d}/attempt=00",
            attempt=0,
            row_count=len(rows),
            min_key=rows[0][0] if rows else None,
            max_key=rows[-1][0] if rows else None,
            checkpoint_id="ckpt",
            writer_info=WriterInfo(),
            all_attempt_urls=(),
        )

    monkeypatch.setattr(dask_writer, "write_shard_with_retry", _fake_distributed)

    runtime = dask_writer.RetryingPartitionWriteRuntime(
        run_id="run-test",
        s3_prefix="s3://bucket/prefix",
        shard_prefix="shards",
        db_path_template="db={db_id:05d}",
        local_root="/tmp/shardyfusion",
        key_col="id",
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        value_spec=ValueSpec.callable_encoder(lambda row: f"v{row['id']}".encode()),
        batch_size=2,
        adapter_factory=_TrackingFactory(),  # type: ignore[arg-type]
        max_writes_per_second=None,
        vector_fn=lambda row: (int(row["id"]), [0.1, 0.2], {"src": "dask"}),
    )
    pdf = pd.DataFrame({DB_ID_COL: [0, 0], "id": [1, 2]})

    out = dask_writer._write_partition(pdf, runtime)

    assert len(out) == 1
    assert captured_rows[0][0] == 1
    assert captured_rows[0][2][0] == 1
    assert captured_rows[0][2][2] == {"src": "dask"}


def test_min_max_key_tracking() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=2, factory=factory)

    records = [{"id": k} for k in [10, 20, 30, 60, 70, 80]]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_hash_sharded(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    # Each shard should track its own min/max keys
    for winner in result.winners:
        assert winner.min_key is not None
        assert winner.max_key is not None
        assert winner.min_key <= winner.max_key


def test_rate_limited_write() -> None:
    factory = _TrackingFactory()
    config = _make_config(num_dbs=1, factory=factory, batch_size=1)

    records = [{"id": i} for i in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_hash_sharded(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=1000.0,
    )

    assert result.stats.rows_written == 5


def testresults_pdf_to_attempts_preserves_all_attempt_urls() -> None:
    results_pdf = pd.DataFrame(
        [
            {
                "db_id": 0,
                "db_url": "s3://bucket/prefix/db=00000/attempt=01",
                "attempt": 1,
                "row_count": 2,
                "min_key": 1,
                "max_key": 2,
                "checkpoint_id": "ckpt",
                "writer_info": WriterInfo(),
                "all_attempt_urls": (
                    "s3://bucket/prefix/db=00000/attempt=00",
                    "s3://bucket/prefix/db=00000/attempt=01",
                ),
            }
        ]
    )

    attempts = results_pdf_to_attempts(results_pdf)

    assert len(attempts) == 1
    assert attempts[0].all_attempt_urls == (
        "s3://bucket/prefix/db=00000/attempt=00",
        "s3://bucket/prefix/db=00000/attempt=01",
    )


# ---------------------------------------------------------------------------
# Rate-limiter integration tests
# ---------------------------------------------------------------------------


def test_rate_limiter_bucket_created_with_correct_rate(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(num_dbs=1, batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    write_hash_sharded(
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

    write_hash_sharded(
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

    write_hash_sharded(
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

    write_hash_sharded(
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

    write_hash_sharded(
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
    config = _make_config(num_dbs=2, batch_size=1)
    records = [{"id": i} for i in range(6)]
    ddf = _make_dask_df(records, npartitions=1)

    write_hash_sharded(
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


def test_rate_limiter_empty_shard_no_bucket(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(num_dbs=4, batch_size=50_000)
    # Use only 2 keys — with 4 shards, some shards will be empty
    records = [{"id": 0}, {"id": 1}]
    ddf = _make_dask_df(records, npartitions=1)

    write_hash_sharded(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    # Only buckets created for shards that actually received rows
    total_acquires = sum(len(b.acquire_calls) for b in _patch_token_bucket)
    assert total_acquires > 0
    assert len(_patch_token_bucket) <= 2  # at most 2 shards for 2 keys


def test_value_spec_binary_col(tmp_path: Path) -> None:
    config, root_dir = _make_file_backed_config(tmp_path, num_dbs=2)

    records = [{"id": i, "payload": f"data-{i}".encode()} for i in range(10)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_hash_sharded(
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


def test_value_spec_json_cols(tmp_path: Path) -> None:
    config, root_dir = _make_file_backed_config(tmp_path, num_dbs=2)

    records = [{"id": i, "name": f"user{i}"} for i in range(10)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_hash_sharded(
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


def test_sort_within_partitions(tmp_path: Path) -> None:
    config, root_dir = _make_file_backed_config(tmp_path, num_dbs=2)

    # Deliberately unsorted records
    records = [{"id": k} for k in [30, 10, 20, 80, 60, 70]]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_hash_sharded(
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


def test_u32be_key_encoding(tmp_path: Path) -> None:
    config, root_dir = _make_file_backed_config(
        tmp_path,
        num_dbs=2,
        key_encoding=KeyEncoding.U32BE,
    )

    records = [{"id": i} for i in range(20)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_hash_sharded(
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
    result = write_hash_sharded(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        verify_routing=True,
    )

    assert result.stats.rows_written == 20


def test_data_integrity(tmp_path: Path) -> None:
    config, root_dir = _make_file_backed_config(tmp_path, num_dbs=4)

    records = [{"id": i} for i in range(100)]
    ddf = _make_dask_df(records, npartitions=4)

    result = write_hash_sharded(
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
            expected_db_id = route_hash(
                key_int,
                num_dbs=config.num_dbs,
            )
            assert expected_db_id == winner.db_id


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_metrics_emitted_on_write() -> None:
    mc = ListMetricsCollector()
    config = _make_config(num_dbs=2)
    config.metrics_collector = mc

    ddf = dd.from_pandas(
        pd.DataFrame({"key": range(10), "val": [b"v"] * 10}),
        npartitions=2,
    )
    write_hash_sharded(
        ddf, config, key_col="key", value_spec=ValueSpec.binary_col("val")
    )

    event_names = [e[0] for e in mc.events]
    assert MetricEvent.WRITE_STARTED in event_names
    assert MetricEvent.SHARDING_COMPLETED in event_names
    assert MetricEvent.SHARD_WRITES_COMPLETED in event_names
    assert MetricEvent.MANIFEST_PUBLISHED in event_names
    assert MetricEvent.WRITE_COMPLETED in event_names

    # WRITE_STARTED should be first, WRITE_COMPLETED last
    assert event_names[0] is MetricEvent.WRITE_STARTED
    assert event_names[-1] is MetricEvent.WRITE_COMPLETED

    # Check payload structure
    write_completed = next(p for e, p in mc.events if e is MetricEvent.WRITE_COMPLETED)
    assert "elapsed_ms" in write_completed
    assert "rows_written" in write_completed
    assert write_completed["rows_written"] == 10
