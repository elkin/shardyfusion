"""Tests for Spark vector sharding."""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.sql import functions as F

from shardyfusion.errors import ShardAssignmentError
from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy

pytest.importorskip("pyspark", reason="requires writer-spark-slatedb extra")

from shardyfusion.writer.spark.sharding import (  # noqa: E402
    VECTOR_DB_ID_COL,
    add_vector_db_id_column,
)
from shardyfusion.writer.spark.vector_writer import (
    _verify_vector_routing_agreement,  # noqa: E402
)
from shardyfusion.writer.spark.vector_writer import (  # noqa: E402
    write_sharded as module_write_sharded,
)


def test_package_exports_vector_writer_entry_points() -> None:
    from shardyfusion.writer.spark import write_sharded, write_vector_sharded

    assert write_sharded is module_write_sharded
    assert write_vector_sharded is module_write_sharded


class TestAddVectorDbIdColumn:
    def test_explicit_adds_column(self, spark) -> None:
        """EXPLICIT: uses existing shard_id column."""
        df = spark.createDataFrame([(1, 0), (2, 1), (3, 2), (4, 3)], ["id", "shard_id"])
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        df_with_id, num_dbs = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )

        assert VECTOR_DB_ID_COL in df_with_id.columns
        ids = [row[VECTOR_DB_ID_COL] for row in df_with_id.collect()]
        assert ids == [0, 1, 2, 3]
        assert num_dbs == 4

    def test_explicit_missing_shard_id_col_raises(self, spark) -> None:
        """EXPLICIT: requires shard_id_col."""
        df = spark.createDataFrame([(1, [0.1] * 128)], ["id", "embedding"])
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        with pytest.raises(AssertionError):
            add_vector_db_id_column(
                df,
                vector_col="embedding",
                routing=routing,
            )

    def test_cluster_adds_column(self, spark) -> None:
        """CLUSTER: applies cluster_assign to vectors."""
        centroids = np.array([[1, 0, 0] * 42 + [0] * 2] * 4, dtype=np.float32).reshape(
            4, 128
        )
        df = spark.createDataFrame(
            [
                (1, [1.0, 0.0, 0.0] * 42),
                (2, [0.0, 1.0, 0.0] * 42),
            ],
            ["id", "embedding"],
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        df_with_id, num_dbs = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
        )

        assert VECTOR_DB_ID_COL in df_with_id.columns
        assert num_dbs == 4

    def test_cluster_bounded_db_id(self, spark) -> None:
        """All cluster-assigned db_ids are in valid range."""
        vectors = np.random.rand(100, 128).astype(np.float32)
        rows = [(i, vectors[i].tolist()) for i in range(100)]
        df = spark.createDataFrame(rows, ["id", "embedding"])

        centroids = np.random.rand(4, 128).astype(np.float32)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
        )

        bad = df_with_id.where(
            (F.col(VECTOR_DB_ID_COL) < 0) | (F.col(VECTOR_DB_ID_COL) >= 4)
        ).count()
        assert bad == 0

    def test_lsh_adds_column(self, spark) -> None:
        """LSH: applies lsh_assign to vectors."""
        hyperplanes = np.random.rand(8, 128).astype(np.float32)
        df = spark.createDataFrame(
            [(1, [1.0] * 128), (2, [-1.0] * 128)],
            ["id", "embedding"],
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=8,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        df_with_id, num_dbs = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
        )

        assert VECTOR_DB_ID_COL in df_with_id.columns
        ids = [row[VECTOR_DB_ID_COL] for row in df_with_id.collect()]
        assert all(0 <= i < 8 for i in ids)
        assert num_dbs == 8

    def test_lsh_bounded_db_id(self, spark) -> None:
        """All lsh-assigned db_ids are in valid range."""
        hyperplanes = np.random.rand(8, 128).astype(np.float32)
        vectors = np.random.rand(100, 128).astype(np.float32)
        rows = [(i, vectors[i].tolist()) for i in range(100)]
        df = spark.createDataFrame(rows, ["id", "embedding"])

        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=8,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
        )

        bad = df_with_id.where(
            (F.col(VECTOR_DB_ID_COL) < 0) | (F.col(VECTOR_DB_ID_COL) >= 8)
        ).count()
        assert bad == 0


class TestCelVectorSharding:
    """CEL-based vector sharding via add_vector_db_id_column."""

    def test_cel_assigns_shard_ids(self, spark) -> None:
        """CEL: evaluates expression per row to produce shard ids."""
        from shardyfusion.cel import compile_cel

        df = spark.createDataFrame(
            [
                (1, [0.1] * 4, "us-east"),
                (2, [0.2] * 4, "eu-west"),
                (3, [0.3] * 4, "us-east"),
                (4, [0.4] * 4, "eu-west"),
            ],
            ["id", "embedding", "region"],
        )
        cel_expr = "shard_hash(region) % 4u"
        cel_cols = {"region": "string"}

        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CEL,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            compiled_cel=compile_cel(cel_expr, cel_cols),
            cel_expr=cel_expr,
        )

        df_with_id, num_dbs = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
            routing_context_cols=cel_cols,
        )

        assert VECTOR_DB_ID_COL in df_with_id.columns
        assert num_dbs == 4
        ids = [row[VECTOR_DB_ID_COL] for row in df_with_id.collect()]
        assert all(0 <= i < 4 for i in ids)
        # Same region → same shard
        assert ids[0] == ids[2]
        assert ids[1] == ids[3]

    def test_cel_closure_is_picklable_regression(self, spark) -> None:
        """Regression: compiled CEL object must not leak into the mapInArrow closure.

        The mapInArrow closure is pickled by Spark to ship to executors.
        A previous bug captured the compiled CEL object (which uses C
        extensions and is not picklable) directly in the closure, causing
        PicklingError on any non-local Spark cluster.  The fix captures
        only the expression string and column dict, then re-compiles
        per-executor inside the closure.

        This test exercises the CEL path through Spark with local[2] so
        the closure actually gets serialized to a separate thread/executor.
        """
        from shardyfusion.cel import compile_cel

        df = spark.createDataFrame(
            [(i, [float(i)] * 4, f"region_{i % 3}") for i in range(20)],
            ["id", "embedding", "region"],
        )
        cel_expr = "shard_hash(region) % 4u"
        cel_cols = {"region": "string"}

        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CEL,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            compiled_cel=compile_cel(cel_expr, cel_cols),
            cel_expr=cel_expr,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
            routing_context_cols=cel_cols,
        )

        # Force materialization — triggers pickle of the mapInArrow closure.
        rows = df_with_id.collect()
        assert len(rows) == 20
        assert all(0 <= row[VECTOR_DB_ID_COL] < 4 for row in rows)


class TestVerifyVectorRoutingAgreement:
    def test_explicit_matches_python_routing(self, spark) -> None:
        df = spark.createDataFrame(
            [
                (1, [1.0, 0.0], 0),
                (2, [0.0, 1.0], 1),
                (3, [0.5, 0.5], 2),
            ],
            ["id", "embedding", "shard_id"],
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )

        _verify_vector_routing_agreement(
            df_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )

    def test_cluster_matches_python_routing(self, spark) -> None:
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        df = spark.createDataFrame(
            [
                (1, [0.9, 0.1]),
                (2, [0.1, 0.9]),
                (3, [0.8, 0.2]),
            ],
            ["id", "embedding"],
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=2,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
        )

        _verify_vector_routing_agreement(
            df_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
        )

    def test_lsh_matches_python_routing(self, spark) -> None:
        hyperplanes = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float32,
        )
        df = spark.createDataFrame(
            [
                (1, [1.0, 0.0]),
                (2, [0.0, 1.0]),
                (3, [-1.0, 1.0]),
            ],
            ["id", "embedding"],
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
        )

        _verify_vector_routing_agreement(
            df_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
        )

    def test_cel_matches_python_routing(self, spark) -> None:
        from shardyfusion.cel import compile_cel

        cel_expr = "shard_hash(region) % 4u"
        cel_cols = {"region": "string"}
        df = spark.createDataFrame(
            [
                (1, [0.1, 0.2], "us-east"),
                (2, [0.2, 0.3], "eu-west"),
                (3, [0.3, 0.4], "ap-south"),
            ],
            ["id", "embedding", "region"],
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CEL,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            compiled_cel=compile_cel(cel_expr, cel_cols),
            cel_expr=cel_expr,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
            routing_context_cols=cel_cols,
        )

        _verify_vector_routing_agreement(
            df_with_id,
            id_col="id",
            vector_col="embedding",
            routing=routing,
            routing_context_cols=cel_cols,
        )

    def test_catches_wrong_vector_db_ids(self, spark) -> None:
        df = spark.createDataFrame(
            [
                (1, [1.0, 0.0], 0),
                (2, [0.0, 1.0], 1),
                (3, [0.5, 0.5], 2),
            ],
            ["id", "embedding", "shard_id"],
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        df_with_id, _ = add_vector_db_id_column(
            df,
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )
        wrong_df = df_with_id.withColumn(
            VECTOR_DB_ID_COL,
            ((F.col(VECTOR_DB_ID_COL) + F.lit(1)) % F.lit(4)).cast("int"),
        )

        with pytest.raises(
            ShardAssignmentError,
            match="Spark/Python vector routing mismatch",
        ):
            _verify_vector_routing_agreement(
                wrong_df,
                id_col="id",
                vector_col="embedding",
                routing=routing,
                shard_id_col="shard_id",
            )
