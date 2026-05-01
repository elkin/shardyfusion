"""Tests for Ray vector sharding."""

from __future__ import annotations

import numpy as np
import pytest

from shardyfusion.errors import ShardAssignmentError
from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy

pytest.importorskip("ray", reason="requires writer-ray-slatedb extra")

import ray.data

from shardyfusion.writer.ray.sharding import (  # noqa: E402
    VECTOR_DB_ID_COL,
    add_vector_db_id_column,
)
from shardyfusion.writer.ray.vector_writer import (
    _verify_vector_routing_agreement,  # noqa: E402
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


class TestVerifyVectorRoutingAgreementRay:
    def test_explicit_matches_python_routing(self) -> None:
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [1.0, 0.0], "shard_id": 0},
                {"id": 2, "embedding": [0.0, 1.0], "shard_id": 1},
                {"id": 3, "embedding": [0.5, 0.5], "shard_id": 2},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        ds_with_id, _ = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )

        _verify_vector_routing_agreement(
            ds_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )

    def test_cluster_matches_python_routing(self) -> None:
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [0.9, 0.1]},
                {"id": 2, "embedding": [0.1, 0.9]},
                {"id": 3, "embedding": [0.8, 0.2]},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=2,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        ds_with_id, _ = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
        )

        _verify_vector_routing_agreement(
            ds_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
        )

    def test_lsh_matches_python_routing(self) -> None:
        hyperplanes = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float32,
        )
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [1.0, 0.0]},
                {"id": 2, "embedding": [0.0, 1.0]},
                {"id": 3, "embedding": [-1.0, 1.0]},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        ds_with_id, _ = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
        )

        _verify_vector_routing_agreement(
            ds_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
        )

    def test_cel_matches_python_routing(self) -> None:
        from shardyfusion.cel import compile_cel

        cel_expr = "shard_hash(region) % 4u"
        cel_cols = {"region": "string"}
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [0.1, 0.2], "region": "us-east"},
                {"id": 2, "embedding": [0.2, 0.3], "region": "eu-west"},
                {"id": 3, "embedding": [0.3, 0.4], "region": "ap-south"},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CEL,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            compiled_cel=compile_cel(cel_expr, cel_cols),
            cel_expr=cel_expr,
        )

        ds_with_id, _ = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
            routing_context_cols=cel_cols,
        )

        _verify_vector_routing_agreement(
            ds_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
            routing_context_cols=cel_cols,
        )

    def test_catches_wrong_vector_db_ids(self) -> None:
        ds = ray.data.from_items(
            [
                {"id": 1, "embedding": [1.0, 0.0], "shard_id": 0},
                {"id": 2, "embedding": [0.0, 1.0], "shard_id": 1},
                {"id": 3, "embedding": [0.5, 0.5], "shard_id": 2},
            ]
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        ds_with_id, _ = add_vector_db_id_column(
            ds,
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )
        wrong_ds = ray.data.from_items(
            [
                {**row, VECTOR_DB_ID_COL: (int(row[VECTOR_DB_ID_COL]) + 1) % 4}
                for row in ds_with_id.take_all()
            ]
        )

        with pytest.raises(
            ShardAssignmentError,
            match="Ray/Python vector routing mismatch",
        ):
            _verify_vector_routing_agreement(
                wrong_ds,
                id_col="id",
                vector_col="embedding",
                routing=routing,
                shard_id_col="shard_id",
            )
