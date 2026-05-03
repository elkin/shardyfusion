from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyspark.sql import Row

from shardyfusion._writer_core import ShardAttemptResult
from shardyfusion.metrics import MetricEvent
from shardyfusion.serde import ValueSpec, make_key_encoder
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.slatedb_adapter import DbAdapterFactory
from shardyfusion.testing import ListMetricsCollector
from shardyfusion.writer.spark.writer import (
    PartitionWriteRuntime,
    write_one_shard_partition,
)
from tests.helpers.tracking import (
    RecordingTokenBucket,
    patch_token_bucket_fixture,
)

_patch_token_bucket = patch_token_bucket_fixture("shardyfusion.writer.spark.writer")

# ---------------------------------------------------------------------------
# Fake adapter infrastructure
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """In-memory SlateDB adapter that records writes without touching disk or S3."""

    def __init__(self, checkpoint_id: str | None = None) -> None:
        self.write_calls: list[list[tuple[bytes, bytes]]] = []
        self._checkpoint_id = checkpoint_id
        self.flushed = False

    def __enter__(self) -> _FakeAdapter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    @property
    def db(self) -> object:
        return self

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self.write_calls.append(list(pairs))

    def flush(self) -> None:
        self.flushed = True

    def checkpoint(self) -> str | None:
        return self._checkpoint_id

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        pass


class _FakeVectorAdapter(_FakeAdapter):
    def __init__(self, checkpoint_id: str | None = None) -> None:
        super().__init__(checkpoint_id=checkpoint_id)
        self.vector_calls: list[
            tuple[object, object, list[dict[str, object] | None]]
        ] = []

    def write_vector_batch(
        self,
        ids: object,
        vectors: object,
        payloads: list[dict[str, object] | None] | None = None,
    ) -> None:
        self.vector_calls.append((ids, vectors, list(payloads or [])))


def _make_factory(adapter: _FakeAdapter) -> DbAdapterFactory:
    def factory(
        *,
        db_url: str,
        local_dir: Path,
    ) -> _FakeAdapter:
        return adapter

    return factory  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_runtime(
    tmp_path,
    *,
    adapter: _FakeAdapter | None = None,
    batch_size: int = 100,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> PartitionWriteRuntime:
    if adapter is None:
        adapter = _FakeAdapter()
    return PartitionWriteRuntime(
        run_id="run-test",
        s3_prefix="s3://bucket/prefix",
        shard_prefix="shards",
        db_path_template="db={db_id:05d}",
        local_root=str(tmp_path),
        key_col="key",
        key_encoding=key_encoding,
        key_encoder=make_key_encoder(key_encoding),
        value_spec=ValueSpec.binary_col("val"),
        batch_size=batch_size,
        adapter_factory=_make_factory(adapter),
        credential_provider=None,
        max_writes_per_second=None,
    )


def _rows(*keys: int) -> list[tuple[int, Row]]:
    """Build a list of (db_id_ignored, Row) pairs for the given integer keys."""
    return [(0, Row(key=k, val=b"v")) for k in keys]


def _run(
    db_id: int, rows: list[tuple[int, Row]], runtime: PartitionWriteRuntime
) -> ShardAttemptResult:
    """Consume the generator and return the single yielded result."""
    with patch("shardyfusion.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = None
        results = list(write_one_shard_partition(db_id, rows, runtime))
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_result_is_dataclass_not_string(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    with patch("shardyfusion.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = None
        results = list(write_one_shard_partition(0, _rows(1), runtime))
    assert len(results) == 1
    assert isinstance(results[0], ShardAttemptResult)


def test_yields_one_result_per_partition(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    result = _run(0, _rows(1, 2, 3), runtime)
    assert isinstance(result, ShardAttemptResult)


def test_correct_db_id_and_url(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    result = _run(7, _rows(10), runtime)
    assert result.db_id == 7
    # URL must embed the shard prefix, run_id, db template, and attempt
    assert result.db_url.startswith(
        "s3://bucket/prefix/shards/run_id=run-test/db=00007/attempt="
    )


def test_row_count_and_min_max_keys(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    result = _run(0, _rows(5, 1, 9, 3), runtime)
    assert result.row_count == 4
    assert result.min_key == 1
    assert result.max_key == 9


def test_empty_partition(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    result = _run(0, [], runtime)
    assert result.row_count == 0
    assert result.min_key is None
    assert result.max_key is None


def test_checkpoint_id_propagated(tmp_path) -> None:
    adapter = _FakeAdapter(checkpoint_id="ckpt-abc")
    runtime = _make_runtime(tmp_path, adapter=adapter)
    result = _run(0, _rows(1), runtime)
    assert result.checkpoint_id == "ckpt-abc"


def test_no_checkpoint_when_not_supported(tmp_path) -> None:
    adapter = _FakeAdapter(checkpoint_id=None)
    runtime = _make_runtime(tmp_path, adapter=adapter)
    result = _run(0, _rows(1), runtime)
    assert result.checkpoint_id is None


def test_writer_info_contains_attempt(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    result = _run(0, _rows(1), runtime)
    # No TaskContext → attempt defaults to 0
    assert result.writer_info.attempt == 0
    assert result.attempt == 0


def test_no_task_context_uses_attempt_zero(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    with patch("shardyfusion.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = None
        (result,) = write_one_shard_partition(0, _rows(1), runtime)
    assert result.attempt == 0
    assert result.writer_info.stage_id is None
    assert result.writer_info.task_attempt_id is None


def test_task_context_fields_propagated(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    mock_ctx = MagicMock()
    mock_ctx.attemptNumber.return_value = 2
    mock_ctx.stageId.return_value = 5
    mock_ctx.taskAttemptId.return_value = 99

    with patch("shardyfusion.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = mock_ctx
        (result,) = write_one_shard_partition(0, _rows(1), runtime)

    assert result.attempt == 2
    assert result.writer_info.stage_id == 5
    assert result.writer_info.task_attempt_id == 99
    assert "/attempt=02" in result.db_url


def test_batch_flushing(tmp_path) -> None:
    adapter = _FakeAdapter()
    runtime = _make_runtime(tmp_path, adapter=adapter, batch_size=2)
    # 5 rows with batch_size=2 → 2 full batches flushed mid-loop + 1 final batch
    _run(0, _rows(1, 2, 3, 4, 5), runtime)
    total_pairs = sum(len(call) for call in adapter.write_calls)
    assert total_pairs == 5
    # First two calls are the mid-loop flushes (2 pairs each)
    assert adapter.write_calls[0] == [
        (b"\x00\x00\x00\x00\x00\x00\x00\x01", b"v"),
        (b"\x00\x00\x00\x00\x00\x00\x00\x02", b"v"),
    ]


def test_rate_limited_partition_write(tmp_path) -> None:
    """Exercises the `bucket is not None` code path in write_one_shard_partition."""
    adapter = _FakeAdapter()
    runtime = _make_runtime(tmp_path, adapter=adapter, batch_size=2)
    # Enable rate limiting (high rate so no real delay)
    runtime = replace(runtime, max_writes_per_second=1000.0)
    result = _run(0, _rows(1, 2, 3, 4, 5), runtime)

    assert result.row_count == 5
    total_pairs = sum(len(call) for call in adapter.write_calls)
    assert total_pairs == 5


def test_vector_partition_uses_distributed_shard_writer(tmp_path) -> None:
    adapter = _FakeVectorAdapter()
    runtime = _make_runtime(tmp_path, adapter=adapter, batch_size=2)
    runtime = replace(
        runtime,
        vector_fn=lambda row: (int(row["key"]), [0.1, 0.2], {"kind": "spark"}),
    )

    result = _run(0, _rows(1, 2, 3), runtime)

    assert result.row_count == 3
    assert len(adapter.write_calls) == 2
    assert len(adapter.vector_calls) == 2
    first_ids, first_vectors, first_payloads = adapter.vector_calls[0]
    assert first_ids.shape == (2,)
    assert first_vectors.shape == (2, 2)
    assert first_payloads[0] == {"kind": "spark"}


# ---------------------------------------------------------------------------
# Rate-limiter integration tests
# ---------------------------------------------------------------------------


def _make_rate_limited_runtime(
    tmp_path,
    *,
    adapter: _FakeAdapter | None = None,
    batch_size: int = 100,
    max_writes_per_second: float | None = 100.0,
) -> PartitionWriteRuntime:
    if adapter is None:
        adapter = _FakeAdapter()
    return PartitionWriteRuntime(
        run_id="run-test",
        s3_prefix="s3://bucket/prefix",
        shard_prefix="shards",
        db_path_template="db={db_id:05d}",
        local_root=str(tmp_path),
        key_col="key",
        key_encoding=KeyEncoding.U64BE,
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        value_spec=ValueSpec.binary_col("val"),
        batch_size=batch_size,
        adapter_factory=_make_factory(adapter),
        credential_provider=None,
        max_writes_per_second=max_writes_per_second,
    )


def test_rate_limiter_bucket_created_with_correct_rate(
    tmp_path,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    runtime = _make_rate_limited_runtime(
        tmp_path, batch_size=50_000, max_writes_per_second=42.5
    )
    _run(0, _rows(1, 2, 3, 4, 5), runtime)

    assert len(_patch_token_bucket) == 1
    assert _patch_token_bucket[0].rate == 42.5


def test_rate_limiter_no_bucket_when_rate_is_none(
    tmp_path,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    runtime = _make_runtime(tmp_path)  # max_writes_per_second=None
    _run(0, _rows(1, 2, 3, 4, 5), runtime)

    assert len(_patch_token_bucket) == 0


def test_rate_limiter_acquire_count_matches_batch_writes(
    tmp_path,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    runtime = _make_rate_limited_runtime(tmp_path, batch_size=3)
    _run(0, _rows(1, 2, 3, 4, 5, 6, 7), runtime)

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # 7 rows / batch_size 3 → batches of [3, 3, 1] → 3 acquire calls
    assert len(bucket.acquire_calls) == 3
    assert bucket.acquire_calls == [3, 3, 1]
    assert sum(bucket.acquire_calls) == 7


def test_rate_limiter_single_batch_single_acquire(
    tmp_path,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    runtime = _make_rate_limited_runtime(tmp_path, batch_size=50_000)
    _run(0, _rows(1, 2, 3, 4, 5), runtime)

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # All 5 rows fit in one batch → single acquire for all 5
    assert bucket.acquire_calls == [5]


def test_rate_limiter_exact_batch_boundary(
    tmp_path,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    runtime = _make_rate_limited_runtime(tmp_path, batch_size=3)
    _run(0, _rows(1, 2, 3, 4, 5, 6), runtime)

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # 6 rows / batch_size 3 → exactly 2 full batches, no trailing partial
    assert bucket.acquire_calls == [3, 3]


def test_u32be_produces_4_byte_keys(tmp_path) -> None:
    adapter = _FakeAdapter()
    runtime = _make_runtime(tmp_path, adapter=adapter, key_encoding=KeyEncoding.U32BE)
    _run(0, _rows(1, 256), runtime)
    written = adapter.write_calls[0]
    # u32be keys should be 4 bytes
    assert written[0][0] == b"\x00\x00\x00\x01"
    assert written[1][0] == b"\x00\x00\x01\x00"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_metrics_emitted_on_shard_write(tmp_path) -> None:
    mc = ListMetricsCollector()
    runtime = _make_runtime(tmp_path, batch_size=5)
    runtime.metrics_collector = mc
    runtime.started = 0.0

    _run(0, _rows(1, 2, 3), runtime)

    event_names = [e[0] for e in mc.events]
    assert MetricEvent.SHARD_WRITE_STARTED in event_names
    assert MetricEvent.BATCH_WRITTEN in event_names
    assert MetricEvent.SHARD_WRITE_COMPLETED in event_names

    # SHARD_WRITE_STARTED should be first, SHARD_WRITE_COMPLETED last
    assert event_names[0] is MetricEvent.SHARD_WRITE_STARTED
    assert event_names[-1] is MetricEvent.SHARD_WRITE_COMPLETED

    # Check payload structure
    completed = next(p for e, p in mc.events if e is MetricEvent.SHARD_WRITE_COMPLETED)
    assert "duration_ms" in completed
    assert "row_count" in completed
    assert completed["row_count"] == 3
