"""Integration tests for async vector reader against moto S3.

These tests verify the full write→manifest→read round-trip using LanceDB adapters
against a real (moto) S3 service, specifically targeting the async reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shardyfusion.async_manifest_store import AsyncS3ManifestStore
from shardyfusion.config import OutputOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.storage import AsyncObstoreBackend, create_s3_store, parse_s3_url
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.vector.adapters.lancedb_adapter import (
    AsyncLanceDbReaderFactory,
    LanceDbWriterFactory,
)
from shardyfusion.vector.async_reader import AsyncShardedVectorReader
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)
from shardyfusion.vector.types import (
    DistanceMetric,
    VectorRecord,
    VectorShardingStrategy,
)
from shardyfusion.vector.writer import write_vector_sharded

pytest.importorskip("lancedb")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def s3_prefix(s3_info: dict[str, Any]) -> str:
    return f"s3://{s3_info['bucket']}/vector-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_async_manifest_store(
    s3_prefix: str,
    cred_provider: StaticCredentialProvider,
    s3_conn_opts: S3ConnectionOptions,
) -> AsyncS3ManifestStore:
    credentials = cred_provider.resolve() if cred_provider else None
    bucket, _ = parse_s3_url(s3_prefix)
    store = create_s3_store(
        bucket=bucket,
        credentials=credentials,
        connection_options=s3_conn_opts,
    )
    backend = AsyncObstoreBackend(store)
    return AsyncS3ManifestStore(backend, s3_prefix)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_records(
    rng: np.random.Generator, n: int = 100, dim: int = 32
) -> list[VectorRecord]:
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return [
        VectorRecord(
            id=i,
            vector=vectors[i],
            payload={"label": f"item-{i}"},
        )
        for i in range(n)
    ]


class TestVectorWriterReaderRoundTrip:
    """Full write → publish → read round-trip against moto S3."""

    @pytest.mark.asyncio
    async def test_cluster_sharding_round_trip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        records = _make_records(rng, n=100, dim=32)

        # Train centroids from data
        centroids = np.array(
            [rng.standard_normal(32).astype(np.float32) for _ in range(4)]
        )

        config = VectorWriteConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            index_config=VectorIndexConfig(dim=32, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
                num_probes=2,
            ),
            adapter_factory=LanceDbWriterFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            batch_size=50,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=OutputOptions(local_root=str(tmp_path / "writer")),
        )

        # Write
        result = write_vector_sharded(records, config)
        assert result.manifest_ref is not None
        assert result.stats.rows_written == 100

        # Read
        reader = await AsyncShardedVectorReader.open(
            s3_prefix=s3_prefix,
            local_root=str(tmp_path / "reader"),
            reader_factory=AsyncLanceDbReaderFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            manifest_store=_make_async_manifest_store(
                s3_prefix, cred_provider, s3_conn_opts
            ),
        )

        info = reader.snapshot_info()
        assert info.num_dbs == 4
        assert info.dim == 32
        assert info.total_vectors == 100

        # Search all shards explicitly
        query = rng.standard_normal(32).astype(np.float32)
        response = await reader.search(query, top_k=5, shard_ids=list(range(4)))
        assert response.num_shards_queried == 4
        assert len(response.results) == 5
        assert response.latency_ms > 0

        # Verify results are sorted by score (L2 = lower is better)
        scores = [r.score for r in response.results]
        assert scores == sorted(scores)

        await reader.close()

    @pytest.mark.asyncio
    async def test_explicit_sharding_round_trip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/explicit"

        # Create records with explicit shard assignments
        records = []
        for i in range(60):
            records.append(
                VectorRecord(
                    id=i,
                    vector=rng.standard_normal(16).astype(np.float32),
                    payload={"idx": i},
                    shard_id=i % 3,
                )
            )

        config = VectorWriteConfig(
            num_dbs=3,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=16, metric=DistanceMetric.COSINE),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
            adapter_factory=LanceDbWriterFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            batch_size=25,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=OutputOptions(local_root=str(tmp_path / "writer_explicit")),
        )

        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 60

        # Read back
        reader = await AsyncShardedVectorReader.open(
            s3_prefix=prefix,
            local_root=str(tmp_path / "reader_explicit"),
            reader_factory=AsyncLanceDbReaderFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            manifest_store=_make_async_manifest_store(
                prefix, cred_provider, s3_conn_opts
            ),
        )

        details = reader.shard_details()
        assert len(details) == 3
        assert all(d.vector_count == 20 for d in details)

        # Query single shard
        query = rng.standard_normal(16).astype(np.float32)
        response = await reader.search(query, top_k=3, shard_ids=[0])
        assert response.num_shards_queried == 1
        assert len(response.results) == 3

        await reader.close()

    @pytest.mark.asyncio
    async def test_lsh_sharding_round_trip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/lsh"
        records = _make_records(rng, n=50, dim=16)

        config = VectorWriteConfig(
            num_dbs=4,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=16, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.LSH,
                num_hash_bits=6,
                num_probes=2,
            ),
            adapter_factory=LanceDbWriterFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            batch_size=20,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=OutputOptions(local_root=str(tmp_path / "writer_lsh")),
        )

        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 50
        assert result.manifest_ref is not None

        # Read back — for LSH, reader loads hyperplanes from manifest
        reader = await AsyncShardedVectorReader.open(
            s3_prefix=prefix,
            local_root=str(tmp_path / "reader_lsh"),
            reader_factory=AsyncLanceDbReaderFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            manifest_store=_make_async_manifest_store(
                prefix, cred_provider, s3_conn_opts
            ),
        )

        info = reader.snapshot_info()
        assert info.num_dbs == 4
        assert info.sharding == VectorShardingStrategy.LSH

        # Search via routing
        query = rng.standard_normal(16).astype(np.float32)
        response = await reader.search(query, top_k=5)
        assert response.num_shards_queried == 2  # num_probes=2
        assert len(response.results) <= 5

        await reader.close()

    @pytest.mark.asyncio
    async def test_health_and_refresh(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/health"
        records = _make_records(rng, n=20, dim=8)

        config = VectorWriteConfig(
            num_dbs=2,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=8, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
            adapter_factory=LanceDbWriterFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=OutputOptions(local_root=str(tmp_path / "writer_health")),
        )

        # All records go to shard 0
        for r in records:
            r.shard_id = 0

        write_vector_sharded(records, config)

        reader = await AsyncShardedVectorReader.open(
            s3_prefix=prefix,
            local_root=str(tmp_path / "reader_health"),
            reader_factory=AsyncLanceDbReaderFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            manifest_store=_make_async_manifest_store(
                prefix, cred_provider, s3_conn_opts
            ),
        )

        health = reader.health()
        assert health.status == "healthy"
        assert health.is_closed is False
        assert health.num_shards == 2

        # Refresh with same manifest returns False
        assert await reader.refresh() is False

        await reader.close()
        health = reader.health()
        assert health.status == "unhealthy"
        assert health.is_closed is True
