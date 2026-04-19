"""Integration test for async unified KV+vector write→read round-trip on moto S3.

Uses sqlite-vec adapters that buffer KV + vector data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("cel_expr_python", reason="requires cel extra")
pytest.importorskip("aiobotocore", reason="requires aiobotocore for async S3")

from shardyfusion.async_manifest_store import AsyncS3ManifestStore
from shardyfusion.config import ManifestOptions, OutputOptions, VectorSpec, WriteConfig
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def s3_info(local_s3_service: dict[str, Any]) -> dict[str, Any]:
    return local_s3_service


@pytest.fixture
def s3_prefix(s3_info: dict[str, Any]) -> str:
    return f"s3://{s3_info['bucket']}/async-unified-test"


@pytest.fixture
def cred_provider(s3_info: dict[str, Any]) -> StaticCredentialProvider:
    return StaticCredentialProvider(
        access_key_id=s3_info["access_key_id"],
        secret_access_key=s3_info["secret_access_key"],
    )


@pytest.fixture
def s3_conn_opts(s3_info: dict[str, Any]) -> S3ConnectionOptions:
    return S3ConnectionOptions(
        endpoint_url=s3_info["endpoint_url"],
        region_name=s3_info["region_name"],
    )


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


class MockRecord:
    def __init__(self, key: int, value: str, vector: np.ndarray) -> None:
        self.key = key
        self.value = value
        self.vector = vector


def _make_records(
    rng: np.random.Generator, n: int = 60, dim: int = 8
) -> list[MockRecord]:
    return [
        MockRecord(
            key=i,
            value=f"val_{i}",
            vector=rng.standard_normal(dim).astype(np.float32),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncUnifiedWriteReadRoundTrip:
    """Full unified KV+vector write → manifest → async read round-trip."""

    @pytest.mark.asyncio
    async def test_async_write_and_read_with_vector_fn(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        from shardyfusion.reader.async_unified_reader import AsyncUnifiedShardedReader
        from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
        from shardyfusion.sqlite_vec_adapter import (
            AsyncSqliteVecReaderFactory,
            SqliteVecFactory,
        )
        from shardyfusion.writer.python.writer import write_sharded

        rng = np.random.default_rng(42)
        records = _make_records(rng)

        sharding = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="shard_hash(key) % 3u",
            cel_columns={"key": "int"},
            routing_values=[0, 1, 2],
        )

        config = WriteConfig(
            s3_prefix=s3_prefix,
            sharding=sharding,
            vector_spec=VectorSpec(dim=8),
            adapter_factory=SqliteVecFactory(
                vector_spec=VectorSpec(dim=8),
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            manifest=ManifestOptions(
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=OutputOptions(local_root=str(tmp_path / "async_unified_writer")),
        )

        # Write synchronously
        result = write_sharded(
            records,
            config,
            key_fn=lambda r: r.key,
            value_fn=lambda r: r.value.encode(),
            vector_fn=lambda r: (r.key, r.vector, None),
        )
        assert result.stats.rows_written == 60
        assert result.manifest_ref is not None

        # Read asynchronously
        reader = await AsyncUnifiedShardedReader.open(
            s3_prefix=s3_prefix,
            local_root=str(tmp_path / "async_unified_reader"),
            reader_factory=AsyncSqliteVecReaderFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            manifest_store=AsyncS3ManifestStore(
                s3_prefix,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
        )

        # 1. Point lookup (KV)
        val = await reader.get(10)
        assert val == b"val_10"

        # 2. Vector search (Vector)
        query = rng.standard_normal(8).astype(np.float32)
        res = await reader.search(query, top_k=5)

        assert len(res.results) == 5
        assert res.num_shards_queried == 3

        # 3. Vector search with routing constraint
        res_routed = await reader.search(query, top_k=3, routing_context={"key": 10})
        assert res_routed.num_shards_queried == 1
        assert len(res_routed.results) <= 3

        # Verify auto-parsing of vector custom metadata
        assert reader.vector_meta.dim == 8
        assert reader.vector_meta.backend == "sqlite-vec"

        await reader.close()
