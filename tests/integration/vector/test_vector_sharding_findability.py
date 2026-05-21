"""End-to-end findability: a vector written to shard S by the writer must be
routed to S by the reader's *automatic* routing (no explicit ``shard_ids``)
and be returned when queried by itself.

This closes the writer↔reader sharding loop that the unit contract test
proves at the function level: here the centroids/hyperplanes actually make a
round trip through the manifest as ``.npy`` artifacts, the per-shard HNSW
indexes are real (LanceDB), and the reader resolves routing solely from the
published manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shardyfusion.config import WriterOutputConfig
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.vector._distributed import (
    assign_vector_shard,
    resolve_vector_routing,
)
from shardyfusion.vector.adapters.lancedb_adapter import (
    LanceDbReaderFactory,
    LanceDbWriterFactory,
)
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardedWriteConfig,
    VectorShardingSpec,
)
from shardyfusion.vector.reader import ShardedVectorReader
from shardyfusion.vector.types import (
    DistanceMetric,
    VectorRecord,
    VectorShardingStrategy,
)
from shardyfusion.vector.writer import write_sharded
from tests.helpers.s3_test_scenarios import _make_s3_manifest_store


@pytest.fixture
def s3_prefix(s3_info: dict[str, Any]) -> str:
    return f"s3://{s3_info['bucket']}/vector-findability"


def _make_records(rng: np.random.Generator, n: int, dim: int) -> list[VectorRecord]:
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return [
        VectorRecord(id=i, vector=vectors[i], payload={"label": f"item-{i}"})
        for i in range(n)
    ]


class TestVectorShardingFindability:
    def test_cluster_findability_and_manifest_roundtrip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/cluster"
        records = _make_records(rng, n=80, dim=32)
        centroids = rng.standard_normal((4, 32)).astype(np.float32)

        config = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=32, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
                num_probes=1,
            ),
            adapter_factory=LanceDbWriterFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            batch_size=40,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=WriterOutputConfig(local_root=str(tmp_path / "w")),
        )

        result = write_sharded(records, config)
        assert result.stats.rows_written == 80

        # Context-managed so the executor and cached shard readers are released
        # even when an assertion below fails mid-test.
        with ShardedVectorReader(
            s3_prefix=prefix,
            local_root=str(tmp_path / "r"),
            reader_factory=LanceDbReaderFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            manifest_store=_make_s3_manifest_store(prefix, cred_provider, s3_conn_opts),
        ) as reader:
            # Manifest round-trip: reader's centroids equal the writer's. No
            # public accessor exposes the resolved routing arrays, so the
            # round-trip is asserted against the private attribute by design.
            assert reader._centroids is not None
            np.testing.assert_array_equal(reader._centroids, centroids)

            routing = resolve_vector_routing(config)
            for rec in records[::8]:
                writer_shard = assign_vector_shard(vector=rec.vector, routing=routing)

                # Reader auto-routes the same vector to the same shard.
                assert reader.route_vector(rec.vector, num_probes=1) == [writer_shard]

                # Querying the vector by itself (no shard_ids → automatic
                # routing) finds it, in exactly its own shard.
                resp = reader.search(rec.vector, top_k=5)
                assert resp.num_shards_queried == 1
                assert str(rec.id) in {str(r.id) for r in resp.results}

    def test_lsh_findability_and_manifest_roundtrip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(7)
        prefix = f"{s3_prefix}/lsh"
        records = _make_records(rng, n=80, dim=16)

        config = VectorShardedWriteConfig(
            num_dbs=4,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=16, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.LSH,
                num_hash_bits=6,
                num_probes=1,
            ),
            adapter_factory=LanceDbWriterFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            batch_size=40,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=WriterOutputConfig(local_root=str(tmp_path / "w")),
        )

        result = write_sharded(records, config)
        assert result.stats.rows_written == 80

        with ShardedVectorReader(
            s3_prefix=prefix,
            local_root=str(tmp_path / "r"),
            reader_factory=LanceDbReaderFactory(
                s3_connection_options=s3_conn_opts,
                credential_provider=cred_provider,
            ),
            manifest_store=_make_s3_manifest_store(prefix, cred_provider, s3_conn_opts),
        ) as reader:
            # Manifest round-trip: reader's hyperplanes equal the writer's
            # (deterministically generated, seed=42) exactly.
            routing = resolve_vector_routing(config)
            assert reader._hyperplanes is not None
            np.testing.assert_array_equal(reader._hyperplanes, routing.hyperplanes)

            for rec in records[::8]:
                writer_shard = assign_vector_shard(vector=rec.vector, routing=routing)
                assert reader.route_vector(rec.vector, num_probes=1) == [writer_shard]

                resp = reader.search(rec.vector, top_k=5)
                assert resp.num_shards_queried == 1
                assert str(rec.id) in {str(r.id) for r in resp.results}
