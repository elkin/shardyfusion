"""Integration tests for distributed unified KV+vector write mode."""

from __future__ import annotations

import importlib.util
import json

import pytest

from shardyfusion._writer_core import VectorColumnMapping
from shardyfusion.config import (
    ManifestOptions,
    OutputOptions,
    VectorSpec,
    WriteConfig,
)
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import parse_manifest_payload
from shardyfusion.serde import ValueSpec
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.type_defs import S3ConnectionOptions
from tests.integration.vector.test_unified_write_read_local_s3 import (
    FakeUnifiedFactory,
)

pytestmark = [
    pytest.mark.cel,
    pytest.mark.skipif(
        importlib.util.find_spec("cel_expr_python") is None,
        reason="requires cel extra",
    ),
]


def _base_config(local_s3_service: dict[str, object], *, prefix: str) -> WriteConfig:
    bucket = local_s3_service["bucket"]
    cred_provider = StaticCredentialProvider(
        access_key_id=local_s3_service["access_key_id"],  # type: ignore[index]
        secret_access_key=local_s3_service["secret_access_key"],  # type: ignore[index]
    )
    s3_conn_opts = S3ConnectionOptions(
        endpoint_url=local_s3_service["endpoint_url"],  # type: ignore[index]
        region_name=local_s3_service["region_name"],  # type: ignore[index]
    )
    return WriteConfig(
        num_dbs=None,
        s3_prefix=f"s3://{bucket}/{prefix}",
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
        adapter_factory=FakeUnifiedFactory(
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        ),
        vector_spec=VectorSpec(dim=4, vector_col="embedding"),
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key % 2",
            cel_columns={"key": "int"},
        ),
        output=OutputOptions(run_id=prefix, local_root="/tmp/shardyfusion-int"),
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        ),
    )


def _assert_vector_manifest(
    local_s3_service: dict[str, object], manifest_ref: str
) -> None:
    bucket = local_s3_service["bucket"]
    manifest_key = manifest_ref.split(f"s3://{bucket}/", 1)[1]  # type: ignore[index]
    raw = (
        local_s3_service["client"]
        .get_object(Bucket=bucket, Key=manifest_key)["Body"]
        .read()
    )  # type: ignore[index]
    manifest = parse_manifest_payload(raw)
    vec = manifest.custom.get("vector")
    assert vec is not None
    assert vec["dim"] == 4
    assert vec["unified"] is True


def test_dask_unified_vector_write(local_s3_service):
    dask = pytest.importorskip("dask")
    dd = pytest.importorskip("dask.dataframe")
    pandas = pytest.importorskip("pandas")
    from shardyfusion.writer.dask import write_sharded as write_dask_sharded

    config = _base_config(local_s3_service, prefix="dask-unified-vectors")
    pdf = pandas.DataFrame(
        {
            "key": [0, 1, 2, 3],
            "value": ["v0", "v1", "v2", "v3"],
            "embedding": [json.dumps([0.0, 0.1, 0.2, 0.3])] * 4,
        }
    )
    with dask.config.set(scheduler="synchronous"):
        ddf = dd.from_pandas(pdf, npartitions=2)
        result = write_dask_sharded(
            ddf,
            config,
            key_col="key",
            value_spec=ValueSpec.callable_encoder(lambda row: row["value"].encode()),
            vector_fn=lambda row: (int(row["key"]), json.loads(row["embedding"]), None),
        )
    assert result.stats.rows_written == 4
    _assert_vector_manifest(local_s3_service, result.manifest_ref)


def test_ray_unified_vector_write(local_s3_service):
    ray_data = pytest.importorskip("ray.data")
    from shardyfusion.writer.ray import write_sharded as write_ray_sharded

    config = _base_config(local_s3_service, prefix="ray-unified-vectors")
    ds = ray_data.from_items(
        [
            {"key": i, "value": f"v{i}", "embedding": [0.1, 0.2, 0.3, 0.4]}
            for i in range(4)
        ]
    )
    result = write_ray_sharded(
        ds,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: row["value"].encode()),
        vector_fn=lambda row: (int(row["key"]), row["embedding"], None),
    )
    assert result.stats.rows_written == 4
    _assert_vector_manifest(local_s3_service, result.manifest_ref)


def test_spark_unified_vector_write(spark, local_s3_service):
    pytest.importorskip("pyspark", reason="requires writer-spark-slatedb extra")
    from shardyfusion.writer.spark import write_sharded as write_spark_sharded

    config = _base_config(local_s3_service, prefix="spark-unified-vectors")
    df = spark.createDataFrame(
        [(0, "v0", [0.0, 0.1, 0.2, 0.3]), (1, "v1", [0.4, 0.5, 0.6, 0.7])],
        schema=["key", "value", "embedding"],
    )
    result = write_spark_sharded(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: row["value"].encode()),
        vector_columns=VectorColumnMapping(vector_col="embedding"),
    )
    assert result.stats.rows_written == 2
    _assert_vector_manifest(local_s3_service, result.manifest_ref)
