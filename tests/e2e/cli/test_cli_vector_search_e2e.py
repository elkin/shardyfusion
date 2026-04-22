"""End-to-end CLI vector search tests against Garage S3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shardyfusion import OutputOptions, VectorSpec, WriteConfig
from shardyfusion.vector.adapters.lancedb_adapter import LanceDbWriterFactory
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
from tests.e2e.cli.conftest import _invoke_cli, _write_cli_configs
from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)

if TYPE_CHECKING:
    from tests.conftest import LocalS3Service


# ---------------------------------------------------------------------------
# LanceDB (non-unified) vector snapshot
# ---------------------------------------------------------------------------


def _write_lancedb_vector_data(s3_service: LocalS3Service, tmp_path: Path) -> str:
    """Write a small LanceDB vector snapshot and return the S3 prefix."""
    pytest.importorskip("lancedb")

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/cli-e2e-vector-lancedb-{tmp_path.name}"
    cred_provider = credential_provider_from_service(s3_service)
    s3_conn_opts = s3_connection_options_from_service(s3_service)

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
        num_dbs=2,
        s3_prefix=s3_prefix,
        index_config=VectorIndexConfig(dim=2, metric=DistanceMetric.L2),
        sharding=VectorShardingSpec(strategy=VectorShardingStrategy.EXPLICIT),
        output=OutputOptions(
            run_id="cli-e2e-vector-run", local_root=str(tmp_path / "writer")
        ),
        adapter_factory=writer_factory,
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    write_vector_sharded(records, config)
    return s3_prefix


@pytest.mark.e2e
@pytest.mark.vector
class TestCliVectorSearchLanceDb:
    def test_search_comma_query(self, garage_s3_service, tmp_path) -> None:
        s3_prefix = _write_lancedb_vector_data(garage_s3_service, tmp_path)
        current_url = f"{s3_prefix}/_CURRENT"
        _write_cli_configs(tmp_path, garage_s3_service, current_url)
        result = _invoke_cli(
            tmp_path, ["search", "1.0,5.0", "--top-k", "3", "--shard-ids", "0,1"]
        )
        assert result.exit_code == 0, result.output + (result.stderr or "")
        parsed = json.loads(result.output)
        assert parsed["op"] == "search"
        assert parsed["top_k"] == 3
        assert parsed["num_shards_queried"] > 0

    def test_search_shard_ids(self, garage_s3_service, tmp_path) -> None:
        s3_prefix = _write_lancedb_vector_data(garage_s3_service, tmp_path)
        current_url = f"{s3_prefix}/_CURRENT"
        _write_cli_configs(tmp_path, garage_s3_service, current_url)
        result = _invoke_cli(tmp_path, ["search", "1.0,5.0", "--shard-ids", "0"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "search"
        assert parsed["num_shards_queried"] == 1


# ---------------------------------------------------------------------------
# sqlite-vec (unified) vector snapshot
# ---------------------------------------------------------------------------


def _write_sqlite_vec_data(s3_service: LocalS3Service, tmp_path: Path) -> str:
    """Write a small unified KV+vector sqlite-vec snapshot and return the S3 prefix."""
    pytest.importorskip("sqlite_vec")

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/cli-e2e-vector-sqlite-{tmp_path.name}"
    cred_provider = credential_provider_from_service(s3_service)
    s3_conn_opts = s3_connection_options_from_service(s3_service)

    from shardyfusion.sharding_types import KeyEncoding
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
    from shardyfusion.writer.python.writer import write_sharded

    num_records = 10
    dim = 4
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

    config = WriteConfig(
        num_dbs=2,
        s3_prefix=s3_prefix,
        adapter_factory=SqliteVecFactory(
            vector_spec=vector_spec,
            s3_connection_options=s3_conn_opts,
            credential_provider=cred_provider,
        ),
        vector_spec=vector_spec,
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
        key_encoding=KeyEncoding.RAW,
        output=OutputOptions(
            run_id="cli-e2e-vector-sqlite", local_root=str(tmp_path / "writer")
        ),
    )

    write_sharded(
        records,
        config,
        key_fn=lambda r: r["id"].encode(),
        value_fn=lambda r: r["payload"],
        vector_fn=lambda r: (r["id"], r["embedding"], None),
    )
    return s3_prefix


@pytest.mark.e2e
@pytest.mark.vector_sqlite
class TestCliVectorSearchSqliteVec:
    def test_search_unified_snapshot(self, garage_s3_service, tmp_path) -> None:
        s3_prefix = _write_sqlite_vec_data(garage_s3_service, tmp_path)
        current_url = f"{s3_prefix}/_CURRENT"
        _write_cli_configs(tmp_path, garage_s3_service, current_url)
        result = _invoke_cli(tmp_path, ["search", "1.0,0.0,0.0,0.0", "--top-k", "5"])
        assert result.exit_code == 0, result.output + (result.stderr or "")
        parsed = json.loads(result.output)
        assert parsed["op"] == "search"
        assert parsed["num_shards_queried"] > 0
        assert len(parsed["results"]) <= 5
