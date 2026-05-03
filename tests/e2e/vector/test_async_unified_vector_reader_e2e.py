"""End-to-end tests for the async unified KV+vector reader against Garage S3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shardyfusion import HashShardedWriteConfig, VectorSpec
from shardyfusion.reader.async_unified_reader import AsyncUnifiedShardedReader
from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.writer_api import write_python_hash_sharded as write_hash_sharded

pytest.importorskip("aiobotocore")
pytest.importorskip("sqlite_vec")


@pytest.mark.e2e
@pytest.mark.vector_sqlite
@pytest.mark.asyncio
async def test_async_unified_vector_reader_e2e(
    garage_s3_service: dict[str, str], tmp_path: Path
) -> None:
    s3_prefix = f"s3://{garage_s3_service['bucket']}/e2e-unified-async"
    cred_provider = credential_provider_from_service(garage_s3_service)
    s3_conn_opts = s3_connection_options_from_service(garage_s3_service)

    num_records = 20
    dim = 8
    num_dbs = 2

    vectors = np.random.rand(num_records, dim).astype(np.float32)
    records = [
        {
            "id": f"key_{i}",
            "payload": f"value_{i}".encode(),
            "embedding": vectors[i],
        }
        for i in range(num_records)
    ]

    vector_spec = VectorSpec(dim=dim, metric="cosine")

    config = HashShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix=s3_prefix,
        adapter_factory=SqliteVecFactory(
            vector_spec=vector_spec,
            s3_connection_options=s3_conn_opts,
            credential_provider=cred_provider,
        ),
        vector_spec=vector_spec,
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    result = write_hash_sharded(
        records,
        config,
        key_fn=lambda r: r["id"].encode(),
        value_fn=lambda r: r["payload"],
        vector_fn=lambda r: (r["id"], r["embedding"], None),
    )

    assert result.run_id is not None
    assert result.manifest_ref is not None

    reader = await AsyncUnifiedShardedReader.open(
        s3_prefix=s3_prefix,
        local_root=str(tmp_path / "reader"),
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    try:
        # Async KV lookup
        val = await reader.get(b"key_5")
        assert val == b"value_5"

        # Async vector search
        query = np.array(vectors[5], dtype=np.float32)
        response = await reader.search(query, top_k=5)
        assert len(response.results) <= 5
        assert response.num_shards_queried > 0
    finally:
        await reader.close()
