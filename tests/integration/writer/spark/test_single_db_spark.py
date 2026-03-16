"""Integration test: Spark single-db writer → moto S3 → verify manifest and CURRENT."""

from __future__ import annotations

import json

import pytest
import yaml

from shardyfusion.config import (
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.serde import ValueSpec
from shardyfusion.testing import file_backed_adapter_factory
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.spark.single_db_writer import write_single_db


@pytest.mark.spark
def test_single_db_spark_publishes_manifest_and_current(
    spark, local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/single-db-spark"
    file_backed_root = str(tmp_path / "file-backed")

    config = WriteConfig(
        num_dbs=1,
        s3_prefix=s3_prefix,
        adapter_factory=file_backed_adapter_factory(file_backed_root),
        output=OutputOptions(
            run_id="single-db-spark-test",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(
            credential_provider=StaticCredentialProvider(
                access_key_id=local_s3_service["access_key_id"],
                secret_access_key=local_s3_service["secret_access_key"],
            ),
            s3_connection_options=S3ConnectionOptions(
                endpoint_url=endpoint_url,
                region_name=local_s3_service["region_name"],
            ),
        ),
    )

    df = spark.createDataFrame([(i, f"val-{i}") for i in range(20)], ["key", "val"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.binary_col("val"),
    )

    assert len(result.winners) == 1
    assert result.winners[0].row_count == 20
    assert result.manifest_ref.startswith(f"s3://{bucket}/single-db-spark/manifests/")

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "single-db-spark/_CURRENT"

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "single-db-spark-test"
    assert manifest_payload["required"]["num_dbs"] == 1
    assert len(manifest_payload["shards"]) == 1
    assert current_payload["manifest_ref"] == result.manifest_ref
