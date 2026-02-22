from __future__ import annotations

from typing import Iterable
from unittest.mock import MagicMock, patch

from pyspark.sql import Row

from slatedb_spark_sharded._writer_core import _ShardAttemptResult
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.sharding_types import KeyEncoding
from slatedb_spark_sharded.slatedb_adapter import DbAdapterFactory
from slatedb_spark_sharded.writer.spark.writer import (
    _PartitionWriteConfig,
    _write_one_shard_partition,
)

# ---------------------------------------------------------------------------
# Fake adapter infrastructure
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """In-memory SlateDB adapter that records writes without touching disk or S3."""

    def __init__(self, checkpoint_id: str | None = None) -> None:
        self.write_calls: list[list[tuple[bytes, bytes]]] = []
        self._checkpoint_id = checkpoint_id
        self.flushed = False

    def __enter__(self) -> "_FakeAdapter":
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

    def close(self) -> None:
        pass


def _make_factory(adapter: _FakeAdapter) -> DbAdapterFactory:
    def factory(
        *,
        db_url: str,
        local_dir: str,
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
) -> _PartitionWriteConfig:
    if adapter is None:
        adapter = _FakeAdapter()
    return _PartitionWriteConfig(
        run_id="run-test",
        s3_prefix="s3://bucket/prefix",
        tmp_prefix="_tmp",
        db_path_template="db={db_id:05d}",
        local_root=str(tmp_path),
        key_col="key",
        key_encoding=key_encoding,
        value_spec=ValueSpec.binary_col("val"),
        batch_size=batch_size,
        adapter_factory=_make_factory(adapter),
        max_writes_per_second=None,
    )


def _rows(*keys: int) -> list[tuple[int, Row]]:
    """Build a list of (db_id_ignored, Row) pairs for the given integer keys."""
    return [(0, Row(key=k, val=b"v")) for k in keys]


def _run(
    db_id: int, rows: list[tuple[int, Row]], runtime: _PartitionWriteConfig
) -> _ShardAttemptResult:
    """Consume the generator and return the single yielded result."""
    with patch("slatedb_spark_sharded.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = None
        results = list(_write_one_shard_partition(db_id, rows, runtime))
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_result_is_dataclass_not_string(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    with patch("slatedb_spark_sharded.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = None
        results = list(_write_one_shard_partition(0, _rows(1), runtime))
    assert len(results) == 1
    assert isinstance(results[0], _ShardAttemptResult)


def test_yields_one_result_per_partition(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    result = _run(0, _rows(1, 2, 3), runtime)
    assert isinstance(result, _ShardAttemptResult)


def test_correct_db_id_and_url(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    result = _run(7, _rows(10), runtime)
    assert result.db_id == 7
    # URL must embed the tmp prefix, run_id, db template, and attempt
    assert result.db_url.startswith(
        "s3://bucket/prefix/_tmp/run_id=run-test/db=00007/attempt="
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
    assert result.writer_info["attempt"] == 0
    assert result.attempt == 0


def test_no_task_context_uses_attempt_zero(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    with patch("slatedb_spark_sharded.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = None
        (result,) = _write_one_shard_partition(0, _rows(1), runtime)
    assert result.attempt == 0
    assert result.writer_info["stage_id"] is None
    assert result.writer_info["task_attempt_id"] is None


def test_task_context_fields_propagated(tmp_path) -> None:
    runtime = _make_runtime(tmp_path)
    mock_ctx = MagicMock()
    mock_ctx.attemptNumber.return_value = 2
    mock_ctx.stageId.return_value = 5
    mock_ctx.taskAttemptId.return_value = 99

    with patch("slatedb_spark_sharded.writer.spark.writer.TaskContext") as mock_tc:
        mock_tc.get.return_value = mock_ctx
        (result,) = _write_one_shard_partition(0, _rows(1), runtime)

    assert result.attempt == 2
    assert result.writer_info["stage_id"] == 5
    assert result.writer_info["task_attempt_id"] == 99
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


def test_u32be_produces_4_byte_keys(tmp_path) -> None:
    adapter = _FakeAdapter()
    runtime = _make_runtime(tmp_path, adapter=adapter, key_encoding=KeyEncoding.U32BE)
    _run(0, _rows(1, 256), runtime)
    written = adapter.write_calls[0]
    # u32be keys should be 4 bytes
    assert written[0][0] == b"\x00\x00\x00\x01"
    assert written[1][0] == b"\x00\x00\x01\x00"
