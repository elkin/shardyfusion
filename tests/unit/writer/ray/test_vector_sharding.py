"""Tests for Ray vector sharding."""

from __future__ import annotations

import numpy as np
import pytest

from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy

pytest.importorskip("ray", reason="requires writer-ray extra")

import ray.data

from shardyfusion.writer.ray.sharding import (  # noqa: E402
    VECTOR_DB_ID_COL,
    add_vector_db_id_column,
)


class TestAddVectorDbIdColumnRay:
    def test_explicit_adds_column(self) -> None:
        """EXPLICIT: uses existing shard_id column."""
        ds = ray.data.from_items(
            [
                {"id": 1, "shard_id": 0},
                {"id": 2, "shard_id": 1},
                {"id": 3, "shard_id": 2},
                {"id": 4, "shard_id": 3},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        ds_with_id, num_dbs = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )

        assert VECTOR_DB_ID_COL in ds_with_id.columns()
        assert num_dbs == 4

    def test_explicit_missing_shard_id_col_raises(self) -> None:
        """EXPLICIT: requires shard_id_col."""
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [0.1] * 128},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        with pytest.raises(AssertionError):
            add_vector_db_id_column(
                ds,
                vector_col="embedding",
                routing=routing,
            )

    def test_cluster_returns_correct_columns(self) -> None:
        """CLUSTER: returns correct column."""
        centroids = np.random.rand(4, 128).astype(np.float32)
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [0.5] * 128},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        ds_with_id, num_dbs = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
        )

        assert VECTOR_DB_ID_COL in ds_with_id.columns()
        assert num_dbs == 4

    def test_lsh_returns_correct_columns(self) -> None:
        """LSH: returns correct column."""
        hyperplanes = np.random.rand(8, 128).astype(np.float32)
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [0.5] * 128},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=8,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        ds_with_id, num_dbs = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
        )

        assert VECTOR_DB_ID_COL in ds_with_id.columns()
        assert num_dbs == 8
