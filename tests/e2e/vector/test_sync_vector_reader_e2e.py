"""End-to-end tests for the sync vector reader against the Garage S3 backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shardyfusion.vector.adapters.lancedb_adapter import (
    LanceDbReaderFactory,
    LanceDbWriterFactory,
)
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)
from shardyfusion.vector.reader import ShardedVectorReader
from shardyfusion.vector.types import (
    DistanceMetric,
    VectorRecord,
    VectorShardingStrategy,
)
from shardyfusion.vector.writer import write_vector_sharded
from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)

pytest.importorskip("lancedb")


@pytest.mark.e2e
@pytest.mark.vector
def test_sync_vector_reader_e2e(
    garage_s3_service: dict[str, str], tmp_path: Path
) -> None:
    s3_prefix = f"s3://{garage_s3_service['bucket']}/e2e-vector-sync"
    cred_provider = credential_provider_from_service(garage_s3_service)
    s3_conn_opts = s3_connection_options_from_service(garage_s3_service)

    records = [
        VectorRecord(
            id=f"id_{i}",
            vector=np.array([1.0, float(i)], dtype=np.float32),
            payload={"val": i},
            shard_id=i % 2,
        )
        for i in range(10)
    ]

    writer_factory = LanceDbWriterFactory(
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    config = VectorWriteConfig(
        s3_prefix=s3_prefix,
        run_id="e2e-run-sync",
        index_config=VectorIndexConfig(dim=2, metric=DistanceMetric.L2),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.EXPLICIT, num_dbs=2
        ),
    )

    write_vector_sharded(
        records,
        config=config,
        local_dir=str(tmp_path / "writer"),
        writer_factory=writer_factory,
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    reader_factory = LanceDbReaderFactory(
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    reader = ShardedVectorReader(
        s3_prefix=s3_prefix,
        local_root=str(tmp_path / "reader"),
        reader_factory=reader_factory,
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    try:
        query = np.array([1.0, 5.0], dtype=np.float32)
        response = reader.search(query, top_k=3, shard_ids=[0, 1])

        assert response.num_shards_queried == 2
        assert len(response.results) == 3
        assert response.results[0].id == "id_5"
    finally:
        reader.close()
