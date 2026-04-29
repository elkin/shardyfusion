"""Integration test: Ray writer → moto S3 → verify manifest and CURRENT."""

from __future__ import annotations

import json
from datetime import timedelta

import ray
import ray.data

from shardyfusion.config import (
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import parse_manifest_payload
from shardyfusion.serde import ValueSpec
from shardyfusion.sharding_types import (
    ShardingSpec,
    ShardingStrategy,
)
from shardyfusion.testing import FailOnceAdapterFactory, file_backed_adapter_factory
from shardyfusion.type_defs import RetryConfig, S3ConnectionOptions
from shardyfusion.writer.ray import write_sharded
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_s3_run_record,
)


def test_ray_writer_publishes_manifest_and_current_to_local_s3(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/ray-writer"
    file_backed_root = str(tmp_path / "file-backed")
    credential_provider = StaticCredentialProvider(
        access_key_id=local_s3_service["access_key_id"],
        secret_access_key=local_s3_service["secret_access_key"],
    )
    connection_options = S3ConnectionOptions(
        endpoint_url=endpoint_url,
        region_name=local_s3_service["region_name"],
    )

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        credential_provider=credential_provider,
        s3_connection_options=connection_options,
        adapter_factory=file_backed_adapter_factory(file_backed_root),
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        output=OutputOptions(
            run_id="ray-writer-local-s3",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(
            credential_provider=credential_provider,
            s3_connection_options=connection_options,
        ),
    )

    ds = ray.data.from_items(
        [{"id": i, "val": f"v{i}"} for i in range(24)],
        override_num_blocks=2,
    )

    result = write_sharded(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: str(row["val"]).encode()),
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/ray-writer/manifests/")

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "ray-writer/_CURRENT"

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest = parse_manifest_payload(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))
    run_record = load_s3_run_record(local_s3_service, result.run_record_ref)

    assert manifest.required_build.run_id == "ray-writer-local-s3"
    assert manifest.required_build.num_dbs == 4
    assert len(manifest.shards) == 4
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="ray",
        s3_prefix=s3_prefix,
    )
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert sum(w.row_count for w in result.winners) == 24


def test_ray_writer_retry_publishes_succeeded_run_record_to_local_s3(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/ray-writer-retry"
    file_backed_root = str(tmp_path / "file-backed-retry")
    credential_provider = StaticCredentialProvider(
        access_key_id=local_s3_service["access_key_id"],
        secret_access_key=local_s3_service["secret_access_key"],
    )
    connection_options = S3ConnectionOptions(
        endpoint_url=endpoint_url,
        region_name=local_s3_service["region_name"],
    )

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        credential_provider=credential_provider,
        s3_connection_options=connection_options,
        adapter_factory=FailOnceAdapterFactory(
            file_backed_adapter_factory(file_backed_root),
            marker_root=str(tmp_path / "retry-markers"),
            fail_db_ids=(0,),
        ),
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        shard_retry=RetryConfig(
            max_retries=1,
            initial_backoff=timedelta(seconds=0),
        ),
        output=OutputOptions(
            run_id="ray-writer-retry-local-s3",
            local_root=str(tmp_path / "local-retry"),
        ),
        manifest=ManifestOptions(
            credential_provider=credential_provider,
            s3_connection_options=connection_options,
        ),
    )

    ds = ray.data.from_items(
        [{"id": i, "val": f"v{i}"} for i in range(24)],
        override_num_blocks=2,
    )

    result = write_sharded(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: str(row["val"]).encode()),
    )

    assert len(result.winners) == 4
    assert any(winner.attempt == 1 for winner in result.winners)
    assert result.run_record_ref is not None

    run_record = load_s3_run_record(local_s3_service, result.run_record_ref)
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="ray",
        s3_prefix=s3_prefix,
    )
