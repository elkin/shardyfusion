"""E2E tests for Dask vector writes with CLUSTER and LSH strategies."""

from __future__ import annotations

import numpy as np
import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)

dask = pytest.importorskip("dask")
dd = pytest.importorskip("dask.dataframe")


@pytest.mark.e2e
@pytest.mark.vector
def test_dask_vector_cluster_write_to_sqlite(garage_s3_service, tmp_path) -> None:
    """Dask writes 1000 vectors with CLUSTER strategy to SQLite backend."""
    import pandas as pd

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
    from shardyfusion.writer.dask import write_vector_sharded
    from tests.helpers.s3_test_scenarios import _make_s3_manifest_store

    bucket = garage_s3_service["bucket"]
    prefix = f"{bucket}/dask-vector-cluster-e2e"
    run_id = "dask-cluster-1000"

    creds = credential_provider_from_service(garage_s3_service)
    opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 1000
    dim = 128
    num_dbs = 4

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    pdf = pd.DataFrame(
        {
            "id": list(range(num_records)),
            "embedding": [vectors[i].tolist() for i in range(num_records)],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=4)

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
        ddf,
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
def test_dask_vector_lsh_write_to_sqlite(garage_s3_service, tmp_path) -> None:
    """Dask writes 1000 vectors with LSH strategy to SQLite backend."""
    import pandas as pd

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
    from shardyfusion.writer.dask import write_vector_sharded
    from tests.helpers.s3_test_scenarios import _make_s3_manifest_store

    bucket = garage_s3_service["bucket"]
    prefix = f"{bucket}/dask-vector-lsh-e2e"
    run_id = "dask-lsh-1000"

    creds = credential_provider_from_service(garage_s3_service)
    opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 1000
    dim = 128
    num_dbs = 8

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    pdf = pd.DataFrame(
        {
            "id": list(range(num_records)),
            "embedding": [vectors[i].tolist() for i in range(num_records)],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=4)

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
        ddf,
        config,
        vector_col="embedding",
        id_col="id",
    )

    assert result.run_id == run_id
    assert result.manifest_ref is not None
    assert len(result.winners) > 0
    total_vectors = sum(w.row_count for w in result.winners)
    assert total_vectors == num_records
