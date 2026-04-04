"""Tests for vector writer using mock adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
