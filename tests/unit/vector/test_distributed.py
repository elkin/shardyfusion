"""Tests for vector _distributed.py core functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.vector._distributed import (
    ResolvedVectorRouting,
    VectorShardState,
    assign_vector_shard,
    flush_vector_shard_batch,
    resolve_vector_routing,
)
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardedWriteConfig,
    VectorShardingSpec,
)
from shardyfusion.vector.types import (
    DistanceMetric,
    VectorShardingStrategy,
)


class MockVectorWriter:
    """Mock vector writer for testing flush_vector_shard_batch."""

    def __init__(self) -> None:
        self.flushed: bool = False
        self.last_batch: tuple | None = None

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        self.last_batch = (ids, vectors, payloads)

    def flush(self) -> None:
        self.flushed = True


class TestVectorShardedWriteConfigValidation:
    def test_valid_cluster_config(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=np.random.rand(4, 128).astype(np.float32),
            ),
        )
        cfg.validate()

    def test_valid_cluster_with_training(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                train_centroids=True,
            ),
        )
        cfg.validate()

    def test_valid_lsh_config(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=8,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.LSH,
                num_hash_bits=8,
            ),
        )
        cfg.validate()

    def test_valid_explicit_config(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
        )
        cfg.validate()

    def test_invalid_zero_dim(self) -> None:
        with pytest.raises(ConfigValidationError, match=r"index_config\.dim"):
            VectorShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=0),
            )

    def test_invalid_negative_dim(self) -> None:
        with pytest.raises(ConfigValidationError, match=r"index_config\.dim"):
            VectorShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=-1),
            )

    def test_invalid_empty_prefix(self) -> None:
        with pytest.raises(ConfigValidationError, match="s3_prefix"):
            VectorShardedWriteConfig(
                s3_prefix="",
                index_config=VectorIndexConfig(dim=128),
            )

    def test_invalid_cluster_without_centroids(self) -> None:
        with pytest.raises(ConfigValidationError, match="centroids"):
            VectorShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=128),
                sharding=VectorShardingSpec(
                    strategy=VectorShardingStrategy.CLUSTER,
                    centroids=None,
                    train_centroids=False,
                ),
            )

    def test_invalid_num_probes(self) -> None:
        with pytest.raises(ConfigValidationError, match="num_probes"):
            VectorShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=128),
                sharding=VectorShardingSpec(
                    strategy=VectorShardingStrategy.CLUSTER,
                    train_centroids=True,
                    num_probes=0,
                ),
            )

    def test_explicit_num_probes_must_be_one(self) -> None:
        with pytest.raises(ConfigValidationError, match="only supported"):
            VectorShardedWriteConfig(
                num_dbs=4,
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=128),
                sharding=VectorShardingSpec(
                    strategy=VectorShardingStrategy.EXPLICIT,
                    num_probes=2,
                ),
            )

    def test_cel_num_probes_must_be_one(self) -> None:
        with pytest.raises(ConfigValidationError, match="only supported"):
            VectorShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=128),
                sharding=VectorShardingSpec(
                    strategy=VectorShardingStrategy.CEL,
                    num_probes=2,
                    cel_expr="shard_hash(region) % 2u",
                    cel_columns={"region": "string"},
                    routing_values=[0, 1],
                ),
            )

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ConfigValidationError, match="batch_size"):
            VectorShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=128),
                batch_size=0,
            )


class TestResolveVectorRouting:
    def test_cluster_resolve_with_provided_centroids(self) -> None:
        centroids = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        cfg = VectorShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert routing.strategy == VectorShardingStrategy.CLUSTER
        assert routing.num_dbs == 4
        assert np.array_equal(routing.centroids, centroids)

    def test_cluster_resolve_with_training(self) -> None:
        sample_vectors = np.random.rand(100, 8).astype(np.float32)
        cfg = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=8),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                train_centroids=True,
            ),
        )

        routing = resolve_vector_routing(cfg, sample_vectors=sample_vectors)

        assert routing.strategy == VectorShardingStrategy.CLUSTER
        assert routing.num_dbs == 4
        assert routing.centroids is not None
        assert routing.centroids.shape == (4, 8)

    def test_cluster_num_dbs_inferred_from_centroids(self) -> None:
        centroids = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        cfg = VectorShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert routing.num_dbs == 3

    def test_cluster_centroids_count_mismatch(self) -> None:
        centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
        cfg = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
            ),
        )
        with pytest.raises(ConfigValidationError, match="centroids count"):
            resolve_vector_routing(cfg)

    def test_cluster_missing_sample_vectors_raises(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=8),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                train_centroids=True,
            ),
        )
        with pytest.raises(ConfigValidationError, match="sample_vectors"):
            resolve_vector_routing(cfg)

    def test_lsh_resolve_generates_hyperplanes(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=8,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.LSH,
                num_hash_bits=8,
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert routing.strategy == VectorShardingStrategy.LSH
        assert routing.num_dbs == 8
        assert routing.hyperplanes is not None
        assert routing.hyperplanes.shape == (8, 128)

    def test_lsh_with_provided_hyperplanes(self) -> None:
        hyperplanes = np.random.rand(8, 128).astype(np.float32)
        cfg = VectorShardedWriteConfig(
            num_dbs=8,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.LSH,
                hyperplanes=hyperplanes,
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert np.array_equal(routing.hyperplanes, hyperplanes)

    def test_explicit_resolve(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert routing.strategy == VectorShardingStrategy.EXPLICIT
        assert routing.num_dbs == 4

    def test_cel_resolve_compiles_expression(self) -> None:
        cfg = VectorShardedWriteConfig(
            num_dbs=3,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_expr="region",
                cel_columns={"region": "string"},
                routing_values=["us", "eu", "asia"],
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert routing.strategy == VectorShardingStrategy.CEL
        assert routing.num_dbs == 3
        assert routing.compiled_cel is not None

    def test_cel_includes_cel_expr_in_routing(self) -> None:
        """Bug fix verification: cel_expr is stored in resolved routing."""
        cfg = VectorShardedWriteConfig(
            num_dbs=3,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_expr="region",
                cel_columns={"region": "string"},
                routing_values=["us", "eu", "asia"],
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert routing.cel_expr == "region"

    def test_cel_num_dbs_inferred_from_routing_values(self) -> None:
        cfg = VectorShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_expr="region",
                cel_columns={"region": "string"},
                routing_values=["us", "eu", "asia"],
            ),
        )

        routing = resolve_vector_routing(cfg)

        assert routing.num_dbs == 3

    def test_invalid_num_dbs_raises(self) -> None:
        cfg = VectorShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
        )
        with pytest.raises(ConfigValidationError, match="num_dbs"):
            resolve_vector_routing(cfg)

    def test_invalid_negative_num_dbs_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="num_dbs"):
            VectorShardedWriteConfig(
                num_dbs=-1,
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=128),
                sharding=VectorShardingSpec(
                    strategy=VectorShardingStrategy.EXPLICIT,
                ),
            )


class TestAssignVectorShard:
    def test_explicit_assign(self) -> None:
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        db_id = assign_vector_shard(
            vector=np.zeros(128, dtype=np.float32),
            routing=routing,
            shard_id=2,
        )

        assert db_id == 2

    def test_explicit_missing_shard_id_raises(self) -> None:
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        with pytest.raises(ConfigValidationError, match="shard_id"):
            assign_vector_shard(
                vector=np.zeros(128, dtype=np.float32),
                routing=routing,
            )

    def test_explicit_out_of_range_raises(self) -> None:
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        with pytest.raises(ConfigValidationError, match="out of range"):
            assign_vector_shard(
                vector=np.zeros(128, dtype=np.float32),
                routing=routing,
                shard_id=10,
            )

    def test_cluster_assign(self) -> None:
        centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=2,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        db_id = assign_vector_shard(
            vector=np.array([0.9, 0.1], dtype=np.float32),
            routing=routing,
        )

        assert db_id == 0

    def test_cluster_missing_centroids_raises(self) -> None:
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=2,
            metric=DistanceMetric.COSINE,
            centroids=None,
        )

        with pytest.raises(ConfigValidationError, match="centroids"):
            assign_vector_shard(
                vector=np.array([0.9, 0.1], dtype=np.float32),
                routing=routing,
            )

    def test_lsh_assign(self) -> None:
        hyperplanes = np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        db_id = assign_vector_shard(
            vector=np.array([1.0, 0.0], dtype=np.float32),
            routing=routing,
        )

        assert 0 <= db_id < 4

    def test_lsh_missing_hyperplanes_raises(self) -> None:
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            hyperplanes=None,
        )

        with pytest.raises(ConfigValidationError, match="hyperplanes"):
            assign_vector_shard(
                vector=np.array([1.0, 0.0], dtype=np.float32),
                routing=routing,
            )


class TestFlushVectorShardBatch:
    def test_flush_calls_add_batch(self) -> None:
        mock_writer = MockVectorWriter()
        state = VectorShardState(
            adapter=mock_writer,
            ids=[1, 2, 3],
            vectors=[np.zeros(8, dtype=np.float32) for _ in range(3)],
            payloads=[None, None, None],
            row_count=3,
        )

        flush_vector_shard_batch(state)

        assert mock_writer.last_batch is not None
        ids, vectors, payloads = mock_writer.last_batch
        assert len(ids) == 3
        assert len(vectors) == 3
        assert state.ids == []
        assert state.vectors == []
        assert state.payloads == []

    def test_flush_empty_batch(self) -> None:
        mock_writer = MockVectorWriter()
        state = VectorShardState(adapter=mock_writer)

        flush_vector_shard_batch(state)

        assert mock_writer.last_batch is None
