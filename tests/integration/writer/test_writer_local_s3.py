from __future__ import annotations

import json

import pytest
import slatedb

from slatedb_spark_sharded.config import (
    EngineOptions,
    ManifestOptions,
    OutputOptions,
    ShardingOptions,
    SlateDbConfig,
)
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.sharding import ShardingSpec, ShardingStrategy
from slatedb_spark_sharded.testing import (
    map_s3_db_url_to_file_url,
    real_file_adapter_factory,
    writer_local_dir_for_db_url,
)
from slatedb_spark_sharded.writer import write_sharded_slatedb


@pytest.mark.spark
def test_writer_publishes_manifest_and_current_to_local_s3(
    spark, local_s3_service, tmp_path
) -> None:
    rows = [(i, f"v{i}".encode("utf-8")) for i in range(24)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/writer-only"
    local_root = str(tmp_path / "writer-local")
    object_store_root = str(tmp_path / "object-store")

    config = SlateDbConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        sharding=ShardingOptions(
            spec=ShardingSpec(
                strategy=ShardingStrategy.RANGE,
                boundaries=[6, 12, 18],
            )
        ),
        engine=EngineOptions(
            slatedb_adapter_factory=real_file_adapter_factory(object_store_root),
        ),
        manifest=ManifestOptions(
            s3_client_config={
                "endpoint_url": endpoint_url,
                "region_name": local_s3_service["region_name"],
                "access_key_id": local_s3_service["access_key_id"],
                "secret_access_key": local_s3_service["secret_access_key"],
            }
        ),
        output=OutputOptions(
            run_id="writer-local-s3",
            local_root=local_root,
        ),
    )

    result = write_sharded_slatedb(df, config)

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/writer-only/manifests/")
    assert result.current_ref == f"s3://{bucket}/writer-only/_CURRENT"

    manifest_bucket, manifest_key = (
        bucket,
        result.manifest_ref.split(f"s3://{bucket}/", 1)[1],
    )
    current_key = result.current_ref.split(f"s3://{bucket}/", 1)[1]

    client = local_s3_service["client"]
    manifest_obj = client.get_object(Bucket=manifest_bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = json.loads(manifest_obj["Body"].read().decode("utf-8"))
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "writer-local-s3"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert current_payload["manifest_content_type"] == "application/json"

    # Verify each shard was physically written and can be read via real SlateDB.
    for winner in result.winners:
        reader = slatedb.SlateDBReader(
            writer_local_dir_for_db_url(winner.db_url, local_root),
            url=map_s3_db_url_to_file_url(winner.db_url, object_store_root),
            checkpoint_id=winner.checkpoint_id,
        )
        try:
            probe_key = winner.db_id * 6
            got = reader.get(probe_key.to_bytes(8, "big", signed=False))
            assert got == f"v{probe_key}".encode("utf-8")
        finally:
            reader.close()
