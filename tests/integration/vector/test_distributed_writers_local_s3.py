"""Integration tests for distributed vector writers - setup verification only.

These tests verify the framework-specific add_vector_db_id_column functions
work correctly with the CLUSTER and LSH strategies. They test setup/flow
without requiring full end-to-end S3 writes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shardyfusion.vector._distributed import resolve_vector_routing
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)

# Conditionally import distributed frameworks
dask = pytest.importorskip("dask", reason="requires writer-dask extra")
dd = pytest.importorskip("dask.dataframe")
spark = pytest.importorskip("pyspark", reason="requires writer-spark extra")
ray_data = pytest.importorskip("ray.data", reason="requires writer-ray extra")


@pytest.fixture(autouse=True)
def _dask_sync():
    """Use synchronous Dask scheduler."""
    with dask.config.set(scheduler="synchronous"):
        yield


def test_spark_write_cluster_sharded(
    spark,
    local_s3_service: dict[str, object],
    tmp_path: Path,
) -> None:
    """Spark writer with CLUSTER strategy - setup verification."""
    from shardyfusion.vector.types import VectorShardingStrategy
    from shardyfusion.writer.spark.sharding import add_vector_db_id_column

    bucket = local_s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/spark-cluster-test"

    # Create sample data
    vectors = np.random.rand(100, 128).astype(np.float32)
    rows = [(i, vectors[i].tolist()) for i in range(100)]
    df = spark.createDataFrame(rows, ["id", "embedding"])

    # Resolve routing with CLUSTER strategy
    centroids = np.random.rand(4, 128).astype(np.float32)
    vec_config = VectorWriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        index_config=VectorIndexConfig(dim=128),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER,
            centroids=centroids,
        ),
    )
    routing = resolve_vector_routing(vec_config)

    # Add vector db_id column - just verify setup works
    df_with_id, num_dbs = add_vector_db_id_column(
        df,
        vector_col="embedding",
        routing=routing,
    )

    assert "_vector_db_id" in df_with_id.columns
    assert num_dbs == 4


def test_dask_write_cluster_sharded(
    local_s3_service: dict[str, object],
    tmp_path: Path,
) -> None:
    """Dask writer with CLUSTER strategy - setup verification."""
    from shardyfusion.vector.types import VectorShardingStrategy
    from shardyfusion.writer.dask.sharding import add_vector_db_id_column

    bucket = local_s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/dask-cluster-test"

    # Create sample data - single row to avoid compute issues
    ddf = dd.from_pandas(
        pytest.importorskip("pandas").DataFrame(
            {
                "id": [1],
                "embedding": [[0.5] * 128],
            }
        ),
        npartitions=1,
    )

    # Resolve routing with CLUSTER strategy
    centroids = np.random.rand(4, 128).astype(np.float32)
    vec_config = VectorWriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        index_config=VectorIndexConfig(dim=128),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER,
            centroids=centroids,
        ),
    )
    routing = resolve_vector_routing(vec_config)

    # Add vector db_id column - just verify setup works
    ddf_with_id, num_dbs = add_vector_db_id_column(
        ddf,
        vector_col="embedding",
        routing=routing,
    )

    assert "_vector_db_id" in ddf_with_id.columns
    assert num_dbs == 4


def test_spark_write_lsh_sharded(
    spark,
    local_s3_service: dict[str, object],
    tmp_path: Path,
) -> None:
    """Spark writer with LSH strategy."""
    from shardyfusion.vector.types import VectorShardingStrategy
    from shardyfusion.writer.spark.sharding import add_vector_db_id_column

    bucket = local_s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/spark-lsh-test"

    vectors = np.random.rand(100, 128).astype(np.float32)
    rows = [(i, vectors[i].tolist()) for i in range(100)]
    df = spark.createDataFrame(rows, ["id", "embedding"])

    hyperplanes = np.random.rand(8, 128).astype(np.float32)
    vec_config = VectorWriteConfig(
        num_dbs=8,
        s3_prefix=s3_prefix,
        index_config=VectorIndexConfig(dim=128),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.LSH,
            hyperplanes=hyperplanes,
            num_hash_bits=8,
        ),
    )
    routing = resolve_vector_routing(vec_config)

    df_with_id, num_dbs = add_vector_db_id_column(
        df,
        vector_col="embedding",
        routing=routing,
    )

    assert "_vector_db_id" in df_with_id.columns
    assert num_dbs == 8
