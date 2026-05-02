"""E2E tests for Ray vector writes with CLUSTER and LSH strategies."""

from __future__ import annotations

import numpy as np
import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)

ray = pytest.importorskip("ray")
ray.data = pytest.importorskip("ray.data")


@pytest.mark.e2e
@pytest.mark.vector
def test_ray_vector_cluster_write_to_sqlite(garage_s3_service, tmp_path) -> None:
    """Ray writes 1000 vectors with CLUSTER strategy to SQLite backend."""
    from shardyfusion.config import (
        ManifestOptions,
        OutputOptions,
        VectorSpec,
    )
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.vector.config import (
        VectorIndexConfig,
        VectorShardingSpec,
        VectorShardingStrategy,
        VectorSpecSharding,
        VectorWriteConfig,
    )
    from shardyfusion.writer.ray import write_vector_sharded
    from tests.helpers.s3_test_scenarios import _make_s3_manifest_store

    bucket = garage_s3_service["bucket"]
    prefix = f"{bucket}/ray-vector-cluster-e2e"
    run_id = "ray-cluster-1000"

    creds = credential_provider_from_service(garage_s3_service)
    opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 1000
    dim = 128
    num_dbs = 4

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    data = [{"id": i, "embedding": vectors[i].tolist()} for i in range(num_records)]
    ds = ray.data.from_items(data)

    config = VectorWriteConfig(
        num_dbs=num_dbs,
        s3_prefix=f"s3://{prefix}",
        index_config=VectorIndexConfig(dim=dim),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER,
            train_centroids=True,
        ),
        output=OutputOptions(run_id=run_id, local_root=str(tmp_path)),
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
        manifest=ManifestOptions(
            store=_make_s3_manifest_store(
                f"s3://{prefix}",
                credential_provider=creds,
                s3_connection_options=opts,
            ),
        ),
        batch_size=100,
    )

    result = write_vector_sharded(
        ds,
        config,
        vector_col="embedding",
        id_col="id",
    )

    assert result.run_id == run_id
    assert result.manifest_ref is not None
    assert len(result.winners) > 0
    total_vectors = sum(w.row_count for w in result.winners)
    assert total_vectors == num_records


@pytest.mark.e2e
@pytest.mark.vector
def test_ray_vector_lsh_write_to_sqlite(garage_s3_service, tmp_path) -> None:
    """Ray writes 1000 vectors with LSH strategy to SQLite backend."""
    from shardyfusion.config import (
        ManifestOptions,
        OutputOptions,
        VectorSpec,
    )
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.vector.config import (
        VectorIndexConfig,
        VectorShardingSpec,
        VectorShardingStrategy,
        VectorSpecSharding,
        VectorWriteConfig,
    )
    from shardyfusion.writer.ray import write_vector_sharded
    from tests.helpers.s3_test_scenarios import _make_s3_manifest_store

    bucket = garage_s3_service["bucket"]
    prefix = f"{bucket}/ray-vector-lsh-e2e"
    run_id = "ray-lsh-1000"

    creds = credential_provider_from_service(garage_s3_service)
    opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 1000
    dim = 128
    num_dbs = 8

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    data = [{"id": i, "embedding": vectors[i].tolist()} for i in range(num_records)]
    ds = ray.data.from_items(data)

    config = VectorWriteConfig(
        num_dbs=num_dbs,
        s3_prefix=f"s3://{prefix}",
        index_config=VectorIndexConfig(dim=dim),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.LSH,
            num_hash_bits=4,
        ),
        output=OutputOptions(run_id=run_id, local_root=str(tmp_path)),
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
        manifest=ManifestOptions(
            store=_make_s3_manifest_store(
                f"s3://{prefix}",
                credential_provider=creds,
                s3_connection_options=opts,
            ),
        ),
        batch_size=100,
    )

    result = write_vector_sharded(
        ds,
        config,
        vector_col="embedding",
        id_col="id",
    )

    assert result.run_id == run_id
    assert result.manifest_ref is not None
    assert len(result.winners) > 0
    total_vectors = sum(w.row_count for w in result.winners)
    assert total_vectors == num_records
