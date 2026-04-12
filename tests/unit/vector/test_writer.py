"""Tests for vector writer using mock adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shardyfusion.config import OutputOptions
from shardyfusion.errors import ConfigValidationError
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)
from shardyfusion.vector.types import VectorRecord, VectorShardingStrategy
from shardyfusion.vector.writer import (
    _assign_shard,
    _flush_shard_batch,
    _ShardState,
    _validate_config,
)

# ---------------------------------------------------------------------------
# Mock adapter for testing without usearch
# ---------------------------------------------------------------------------


@dataclass
class MockVectorWriter:
    batches: list[tuple[np.ndarray, np.ndarray, list[dict[str, Any]] | None]] = field(
        default_factory=list
    )
    flushed: bool = False
    closed: bool = False
    _checkpoint_id: str = "mock-checkpoint"

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        self.batches.append((ids, vectors, payloads))

    def flush(self) -> None:
        self.flushed = True

    def checkpoint(self) -> str | None:
        return self._checkpoint_id

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> MockVectorWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MockWriterFactory:
    def __init__(self) -> None:
        self.writers: dict[str, MockVectorWriter] = {}

    def __call__(
        self, *, db_url: str, local_dir: Path, index_config: Any
    ) -> MockVectorWriter:
        w = MockVectorWriter()
        self.writers[db_url] = w
        return w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config(self):
        cfg = VectorWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                train_centroids=True,
            ),
        )
        _validate_config(cfg)  # should not raise

    def test_zero_dim_raises(self):
        cfg = VectorWriteConfig(
            s3_prefix="s3://bucket",
            index_config=VectorIndexConfig(dim=0),
        )
        with pytest.raises(ConfigValidationError, match="dim"):
            _validate_config(cfg)

    def test_empty_prefix_raises(self):
        cfg = VectorWriteConfig(
            s3_prefix="",
            index_config=VectorIndexConfig(dim=128),
        )
        with pytest.raises(ConfigValidationError, match="s3_prefix"):
            _validate_config(cfg)

    def test_cluster_without_centroids_or_training_raises(self):
        cfg = VectorWriteConfig(
            s3_prefix="s3://bucket",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                train_centroids=False,
                centroids=None,
            ),
        )
        with pytest.raises(ConfigValidationError, match="centroids"):
            _validate_config(cfg)


class TestAssignShard:
    def test_explicit_valid(self):
        record = VectorRecord(id=1, vector=np.zeros(8, dtype=np.float32), shard_id=2)
        db_id = _assign_shard(
            record=record,
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=None,
            centroids=None,
            hyperplanes=None,
        )
        assert db_id == 2

    def test_explicit_missing_shard_id_raises(self):
        record = VectorRecord(id=1, vector=np.zeros(8, dtype=np.float32))
        with pytest.raises(ConfigValidationError, match="shard_id"):
            _assign_shard(
                record=record,
                strategy=VectorShardingStrategy.EXPLICIT,
                num_dbs=4,
                metric=None,
                centroids=None,
                hyperplanes=None,
            )

    def test_explicit_out_of_range_raises(self):
        record = VectorRecord(id=1, vector=np.zeros(8, dtype=np.float32), shard_id=10)
        with pytest.raises(ConfigValidationError, match="out of range"):
            _assign_shard(
                record=record,
                strategy=VectorShardingStrategy.EXPLICIT,
                num_dbs=4,
                metric=None,
                centroids=None,
                hyperplanes=None,
            )

    def test_cluster_assign(self):
        centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
        record = VectorRecord(id=1, vector=np.array([0.9, 0.1], dtype=np.float32))
        db_id = _assign_shard(
            record=record,
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=2,
            metric=None,
            centroids=centroids,
            hyperplanes=None,
        )
        assert db_id == 0


class TestFlushShardBatch:
    def test_flush_clears_buffers(self):
        writer = MockVectorWriter()
        state = _ShardState(adapter=writer)
        state.ids = [1, 2, 3]
        state.vectors = [
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ]
        state.payloads = [None, None, None]

        _flush_shard_batch(state)

        assert len(writer.batches) == 1
        assert len(state.ids) == 0
        assert len(state.vectors) == 0

    def test_flush_empty_is_noop(self):
        writer = MockVectorWriter()
        state = _ShardState(adapter=writer)
        _flush_shard_batch(state)
        assert len(writer.batches) == 0

    def test_flush_with_payloads(self):
        writer = MockVectorWriter()
        state = _ShardState(adapter=writer)
        state.ids = [1]
        state.vectors = [np.zeros(4, dtype=np.float32)]
        state.payloads = [{"key": "value"}]

        _flush_shard_batch(state)

        assert writer.batches[0][2] is not None
        assert writer.batches[0][2][0] == {"key": "value"}

    def test_flush_with_mixed_none_payloads(self):
        writer = MockVectorWriter()
        state = _ShardState(adapter=writer)
        state.ids = [1, 2]
        state.vectors = [
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ]
        state.payloads = [{"key": "val"}, None]

        _flush_shard_batch(state)

        payloads = writer.batches[0][2]
        assert payloads is not None
        assert payloads[0] == {"key": "val"}
        assert payloads[1] == {}  # None replaced with empty dict

    def test_flush_string_ids_uses_object_dtype(self):
        """String IDs should be preserved as dtype=object, not int64."""
        writer = MockVectorWriter()
        state = _ShardState(adapter=writer)
        state.ids = ["doc_a", "doc_b"]
        state.vectors = [np.zeros(4, dtype=np.float32)] * 2
        state.payloads = [None, None]

        _flush_shard_batch(state)

        ids_arr = writer.batches[0][0]
        assert ids_arr.dtype == object
        assert list(ids_arr) == ["doc_a", "doc_b"]

    def test_flush_int_ids_uses_int64_dtype(self):
        writer = MockVectorWriter()
        state = _ShardState(adapter=writer)
        state.ids = [10, 20]
        state.vectors = [np.zeros(4, dtype=np.float32)] * 2
        state.payloads = [None, None]

        _flush_shard_batch(state)

        ids_arr = writer.batches[0][0]
        assert ids_arr.dtype == np.int64

    def test_flush_all_none_payloads_passes_none(self):
        """When all payloads are None, payloads_list should be None."""
        writer = MockVectorWriter()
        state = _ShardState(adapter=writer)
        state.ids = [1]
        state.vectors = [np.zeros(4, dtype=np.float32)]
        state.payloads = [None]

        _flush_shard_batch(state)

        assert writer.batches[0][2] is None


# ---------------------------------------------------------------------------
# _assign_shard — additional strategies
# ---------------------------------------------------------------------------


class TestAssignShardAdditional:
    def test_cluster_missing_centroids_raises(self):
        record = VectorRecord(id=1, vector=np.zeros(4, dtype=np.float32))
        with pytest.raises(ConfigValidationError, match="centroids"):
            _assign_shard(
                record=record,
                strategy=VectorShardingStrategy.CLUSTER,
                num_dbs=4,
                metric=None,
                centroids=None,
                hyperplanes=None,
            )

    def test_lsh_missing_hyperplanes_raises(self):
        record = VectorRecord(id=1, vector=np.zeros(4, dtype=np.float32))
        with pytest.raises(ConfigValidationError, match="hyperplanes"):
            _assign_shard(
                record=record,
                strategy=VectorShardingStrategy.LSH,
                num_dbs=4,
                metric=None,
                centroids=None,
                hyperplanes=None,
            )

    def test_lsh_assignment(self):
        from shardyfusion.vector.sharding import lsh_generate_hyperplanes

        hps = lsh_generate_hyperplanes(num_hash_bits=3, dim=4, seed=42)
        record = VectorRecord(id=1, vector=np.array([1, 0, 0, 0], dtype=np.float32))
        db_id = _assign_shard(
            record=record,
            strategy=VectorShardingStrategy.LSH,
            num_dbs=8,
            metric=None,
            centroids=None,
            hyperplanes=hps,
        )
        assert 0 <= db_id < 8


# ---------------------------------------------------------------------------
# _write_single_process (core write loop)
# ---------------------------------------------------------------------------


class TestWriteSingleProcess:
    def _make_config(self, tmp_path: Path, num_dbs: int = 2) -> VectorWriteConfig:
        return VectorWriteConfig(
            num_dbs=num_dbs,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=4),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
            output=OutputOptions(local_root=str(tmp_path)),
            batch_size=100,
        )

    def test_records_distributed_to_correct_shards(self, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import _write_single_process

        factory = MockWriterFactory()
        config = self._make_config(tmp_path)

        records = [
            VectorRecord(id=1, vector=np.zeros(4, dtype=np.float32), shard_id=0),
            VectorRecord(id=2, vector=np.ones(4, dtype=np.float32), shard_id=1),
            VectorRecord(id=3, vector=np.zeros(4, dtype=np.float32), shard_id=0),
        ]

        states = _write_single_process(
            records=records,
            config=config,
            num_dbs=2,
            run_id="test-run",
            adapter_factory=factory,
            resolved_centroids=None,
            resolved_hyperplanes=None,
            ops_limiter=None,
            s3_client=None,
        )

        assert 0 in states
        assert 1 in states
        assert states[0].row_count == 2
        assert states[1].row_count == 1

    def test_batching_flushes_at_threshold(self, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import _write_single_process

        factory = MockWriterFactory()
        config = self._make_config(tmp_path)
        config.batch_size = 2  # Flush every 2 records

        records = [
            VectorRecord(id=i, vector=np.zeros(4, dtype=np.float32), shard_id=0)
            for i in range(5)
        ]

        states = _write_single_process(
            records=records,
            config=config,
            num_dbs=1,
            run_id="test-run",
            adapter_factory=factory,
            resolved_centroids=None,
            resolved_hyperplanes=None,
            ops_limiter=None,
            s3_client=None,
        )

        # 5 records / batch_size 2 = 2 full batches + 1 final flush
        writer = list(factory.writers.values())[0]
        assert len(writer.batches) == 3
        assert states[0].row_count == 5

    def test_checkpoint_called_on_all_shards(self, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import _write_single_process

        factory = MockWriterFactory()
        config = self._make_config(tmp_path)

        records = [
            VectorRecord(id=1, vector=np.zeros(4, dtype=np.float32), shard_id=0),
            VectorRecord(id=2, vector=np.zeros(4, dtype=np.float32), shard_id=1),
        ]

        states = _write_single_process(
            records=records,
            config=config,
            num_dbs=2,
            run_id="test-run",
            adapter_factory=factory,
            resolved_centroids=None,
            resolved_hyperplanes=None,
            ops_limiter=None,
            s3_client=None,
        )

        assert states[0].checkpoint_id == "mock-checkpoint"
        assert states[1].checkpoint_id == "mock-checkpoint"

    def test_empty_records_produces_empty_states(self, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import _write_single_process

        factory = MockWriterFactory()
        config = self._make_config(tmp_path)

        states = _write_single_process(
            records=[],
            config=config,
            num_dbs=2,
            run_id="test-run",
            adapter_factory=factory,
            resolved_centroids=None,
            resolved_hyperplanes=None,
            ops_limiter=None,
            s3_client=None,
        )

        assert len(states) == 0

    def test_db_url_includes_run_id(self, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import _write_single_process

        factory = MockWriterFactory()
        config = self._make_config(tmp_path)

        records = [
            VectorRecord(id=1, vector=np.zeros(4, dtype=np.float32), shard_id=0),
        ]

        states = _write_single_process(
            records=records,
            config=config,
            num_dbs=1,
            run_id="my-run-123",
            adapter_factory=factory,
            resolved_centroids=None,
            resolved_hyperplanes=None,
            ops_limiter=None,
            s3_client=None,
        )

        assert "run_id=my-run-123" in states[0].db_url


# ---------------------------------------------------------------------------
# _validate_config — additional cases
# ---------------------------------------------------------------------------


class TestValidateConfigAdditional:
    def test_negative_batch_size_raises(self):
        cfg = VectorWriteConfig(
            s3_prefix="s3://bucket",
            index_config=VectorIndexConfig(dim=128),
            batch_size=0,
        )
        with pytest.raises(ConfigValidationError, match="batch_size"):
            _validate_config(cfg)

    def test_cel_missing_expr_raises(self):
        cfg = VectorWriteConfig(
            s3_prefix="s3://bucket",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_columns={"key": "string"},
            ),
        )
        with pytest.raises(ConfigValidationError, match="cel_expr"):
            _validate_config(cfg)

    def test_cel_missing_columns_raises(self):
        cfg = VectorWriteConfig(
            s3_prefix="s3://bucket",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_expr="shard_hash(key) % 4u",
            ),
        )
        with pytest.raises(ConfigValidationError, match="cel_columns"):
            _validate_config(cfg)

    def test_num_probes_zero_raises(self):
        cfg = VectorWriteConfig(
            s3_prefix="s3://bucket",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
                num_probes=0,
            ),
        )
        with pytest.raises(ConfigValidationError, match="num_probes"):
            _validate_config(cfg)


# ---------------------------------------------------------------------------
# write_vector_sharded — parallel not supported
# ---------------------------------------------------------------------------


class TestWriteVectorShardedEntryPoint:
    def test_parallel_raises(self):
        from shardyfusion.vector.writer import write_vector_sharded

        cfg = VectorWriteConfig(
            num_dbs=2,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=4),
        )
        with pytest.raises(ConfigValidationError, match="Parallel"):
            write_vector_sharded([], cfg, parallel=True)
