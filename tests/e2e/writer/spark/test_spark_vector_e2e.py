"""E2E tests for Spark vector writes with CLUSTER and LSH strategies."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyspark", reason="requires writer-spark-slatedb extra")

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)


@pytest.mark.e2e
@pytest.mark.vector
def test_spark_vector_cluster_write_to_sqlite(
    spark, garage_s3_service, tmp_path
) -> None:
    """Spark writes 1000 vectors with CLUSTER strategy to SQLite backend."""
    from shardyfusion.config import (
        VectorSpec,
        WriterManifestConfig,
        WriterOutputConfig,
    )
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.vector.config import (
        VectorIndexConfig,
        VectorShardedWriteConfig,
        VectorShardingSpec,
        VectorShardingStrategy,
        VectorSpecSharding,
    )
    from tests.helpers.s3_test_scenarios import _make_s3_manifest_store
    from tests.helpers.writer_api import write_spark_vector_sharded as write_sharded

    bucket = garage_s3_service["bucket"]
    prefix = f"{bucket}/spark-vector-cluster-e2e"
    run_id = "spark-cluster-1000"

    creds = credential_provider_from_service(garage_s3_service)
    opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 1000
    dim = 128
    num_dbs = 4

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    rows = [(i, vectors[i].tolist()) for i in range(num_records)]
    df = spark.createDataFrame(rows, ["id", "embedding"])

    config = VectorShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix=f"s3://{prefix}",
        index_config=VectorIndexConfig(dim=dim),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER,
            train_centroids=True,
        ),
        output=WriterOutputConfig(run_id=run_id, local_root=str(tmp_path)),
        adapter_factory=SqliteVecFactory(
            vector_spec=VectorSpec(
                dim=dim,
                vector_col="embedding",
                sharding=VectorSpecSharding(
                    strategy="cluster",
                    train_centroids=True,
                ),
            ),
            s3_connection_options=opts,
            credential_provider=creds,
        ),
        credential_provider=creds,
        s3_connection_options=opts,
        manifest=WriterManifestConfig(
            store=_make_s3_manifest_store(
                f"s3://{prefix}",
                credential_provider=creds,
                s3_connection_options=opts,
            ),
        ),
        batch_size=100,
    )

    result = write_sharded(
        df,
        config,
        vector_col="embedding",
        id_col="id",
    )

    assert result.run_id == run_id
    assert result.manifest_ref is not None
    assert len(result.winners) > 0
    total_vectors = sum(w.row_count for w in result.winners)
    assert total_vectors == num_records

    # Verification: Read back
    from shardyfusion.sqlite_vec_adapter import SqliteVecReaderFactory
    from shardyfusion.vector.reader import ShardedVectorReader

    reader = ShardedVectorReader(
        s3_prefix=f"s3://{prefix}",
        local_root=str(tmp_path / f"reader_{run_id}"),
        reader_factory=SqliteVecReaderFactory(
            credential_provider=creds,
            s3_connection_options=opts,
        ),
        manifest_store=_make_s3_manifest_store(
            f"s3://{prefix}",
            credential_provider=creds,
            s3_connection_options=opts,
        ),
    )

    try:
        # Vector search
        query = np.array(vectors[0], dtype=np.float32)
        response = reader.search(query, top_k=5)
        assert len(response.results) <= 5
        assert response.num_shards_queried > 0
        # The record's own vector should be the best match
        assert response.results[0].id == 0
        assert response.results[0].score < 1e-4
    finally:
        reader.close()


@pytest.mark.e2e
@pytest.mark.vector
def test_spark_vector_lsh_write_to_sqlite(spark, garage_s3_service, tmp_path) -> None:
    """Spark writes 1000 vectors with LSH strategy to SQLite backend."""
    from shardyfusion.config import (
        VectorSpec,
        WriterManifestConfig,
        WriterOutputConfig,
    )
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.vector.config import (
        VectorIndexConfig,
        VectorShardedWriteConfig,
        VectorShardingSpec,
        VectorShardingStrategy,
        VectorSpecSharding,
    )
    from tests.helpers.s3_test_scenarios import _make_s3_manifest_store
    from tests.helpers.writer_api import write_spark_vector_sharded as write_sharded

    bucket = garage_s3_service["bucket"]
    prefix = f"{bucket}/spark-vector-lsh-e2e"
    run_id = "spark-lsh-1000"

    creds = credential_provider_from_service(garage_s3_service)
    opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 1000
    dim = 128
    num_dbs = 8

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    rows = [(i, vectors[i].tolist()) for i in range(num_records)]
    df = spark.createDataFrame(rows, ["id", "embedding"])

    config = VectorShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix=f"s3://{prefix}",
        index_config=VectorIndexConfig(dim=dim),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.LSH,
            num_hash_bits=4,
        ),
        output=WriterOutputConfig(run_id=run_id, local_root=str(tmp_path)),
        adapter_factory=SqliteVecFactory(
            vector_spec=VectorSpec(
                dim=dim,
                vector_col="embedding",
                sharding=VectorSpecSharding(
                    strategy="lsh",
                    num_hash_bits=4,
                ),
            ),
            s3_connection_options=opts,
            credential_provider=creds,
        ),
        credential_provider=creds,
        s3_connection_options=opts,
        manifest=WriterManifestConfig(
            store=_make_s3_manifest_store(
                f"s3://{prefix}",
                credential_provider=creds,
                s3_connection_options=opts,
            ),
        ),
        batch_size=100,
    )

    result = write_sharded(
        df,
        config,
        vector_col="embedding",
        id_col="id",
    )

    assert result.run_id == run_id
    assert result.manifest_ref is not None
    assert len(result.winners) > 0
    total_vectors = sum(w.row_count for w in result.winners)
    assert total_vectors == num_records

    # Verification: Read back
    from shardyfusion.sqlite_vec_adapter import SqliteVecReaderFactory
    from shardyfusion.vector.reader import ShardedVectorReader

    reader = ShardedVectorReader(
        s3_prefix=f"s3://{prefix}",
        local_root=str(tmp_path / f"reader_{run_id}"),
        reader_factory=SqliteVecReaderFactory(
            credential_provider=creds,
            s3_connection_options=opts,
        ),
        manifest_store=_make_s3_manifest_store(
            f"s3://{prefix}",
            credential_provider=creds,
            s3_connection_options=opts,
        ),
    )

    try:
        # Vector search
        query = np.array(vectors[0], dtype=np.float32)
        response = reader.search(query, top_k=5)
        assert len(response.results) <= 5
        assert response.num_shards_queried > 0
        # The record's own vector should be the best match
        assert response.results[0].id == 0
        assert response.results[0].score < 1e-4
    finally:
        reader.close()


@pytest.mark.e2e
@pytest.mark.vector
def test_spark_vector_lancedb_write_and_read(
    spark, garage_s3_service, tmp_path
) -> None:
    """Spark writes 1000 vectors to LanceDB and reads back."""
    from shardyfusion.config import (
        WriterManifestConfig,
        WriterOutputConfig,
    )
    from shardyfusion.vector.adapters.lancedb_adapter import (
        LanceDbReaderFactory,
        LanceDbWriterFactory,
    )
    from shardyfusion.vector.config import (
        VectorIndexConfig,
        VectorShardedWriteConfig,
        VectorShardingSpec,
        VectorShardingStrategy,
    )
    from shardyfusion.vector.reader import ShardedVectorReader
    from tests.helpers.s3_test_scenarios import _make_s3_manifest_store
    from tests.helpers.writer_api import write_spark_vector_sharded as write_sharded

    bucket = garage_s3_service["bucket"]
    prefix = f"s3://{bucket}/spark-vector-lancedb-e2e"
    run_id = "spark-lancedb-1000"

    creds = credential_provider_from_service(garage_s3_service)
    opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 1000
    dim = 128
    num_dbs = 4

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    rows = [(i, vectors[i].tolist()) for i in range(num_records)]
    df = spark.createDataFrame(rows, ["id", "embedding"])

    config = VectorShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix=prefix,
        index_config=VectorIndexConfig(dim=dim),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.LSH,
            num_hash_bits=4,
            num_probes=2,
        ),
        output=WriterOutputConfig(run_id=run_id, local_root=str(tmp_path)),
        adapter_factory=LanceDbWriterFactory(
            credential_provider=creds,
            s3_connection_options=opts,
        ),
        credential_provider=creds,
        s3_connection_options=opts,
        manifest=WriterManifestConfig(
            store=_make_s3_manifest_store(
                prefix,
                credential_provider=creds,
                s3_connection_options=opts,
            ),
        ),
        batch_size=100,
    )

    result = write_sharded(
        df,
        config,
        vector_col="embedding",
        id_col="id",
    )

    assert result.run_id == run_id
    assert result.manifest_ref is not None
    assert len(result.winners) > 0

    reader = ShardedVectorReader(
        s3_prefix=prefix,
        local_root=str(tmp_path / "reader_lancedb"),
        reader_factory=LanceDbReaderFactory(
            credential_provider=creds,
            s3_connection_options=opts,
        ),
        manifest_store=_make_s3_manifest_store(
            prefix,
            credential_provider=creds,
            s3_connection_options=opts,
        ),
    )

    try:
        query = np.array(vectors[0], dtype=np.float32)
        response = reader.search(query, top_k=5)
        assert len(response.results) <= 5
        assert response.num_shards_queried > 0
        assert int(response.results[0].id) == 0
        assert response.results[0].score < 1e-4
    finally:
        reader.close()
