"""Integration test: pure-Python writer → moto S3 → verify manifest and CURRENT."""

from __future__ import annotations

import json
from datetime import timedelta

import yaml

from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.testing import FailOnceAdapterFactory, file_backed_adapter_factory
from shardyfusion.type_defs import RetryConfig, S3ConnectionOptions
from shardyfusion.writer.python import write_sharded
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_s3_run_record,
)


def test_python_writer_publishes_manifest_and_current_to_local_s3(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/python-writer"
    file_backed_root = str(tmp_path / "file-backed")

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        adapter_factory=file_backed_adapter_factory(file_backed_root),
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        output=OutputOptions(
            run_id="python-writer-local-s3",
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

    records = list(range(24))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/python-writer/manifests/")
    assert result.run_record_ref is not None
    assert result.run_record_ref.startswith(f"s3://{bucket}/python-writer/runs/")

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "python-writer/_CURRENT"

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))
    run_record = load_s3_run_record(local_s3_service, result.run_record_ref)

    assert manifest_payload["required"]["run_id"] == "python-writer-local-s3"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="python",
        s3_prefix=s3_prefix,
    )
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert sum(w.row_count for w in result.winners) == 24


def test_python_writer_parallel_publishes_manifest_and_current_to_local_s3(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/python-writer-parallel"
    file_backed_root = str(tmp_path / "file-backed-parallel")

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        adapter_factory=file_backed_adapter_factory(file_backed_root),
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        output=OutputOptions(
            run_id="python-writer-parallel-local-s3",
            local_root=str(tmp_path / "local-parallel"),
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

    records = list(range(24))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
        parallel=True,
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(
        f"s3://{bucket}/python-writer-parallel/manifests/"
    )
    assert result.run_record_ref is not None

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "python-writer-parallel/_CURRENT"

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))
    run_record = load_s3_run_record(local_s3_service, result.run_record_ref)

    assert manifest_payload["required"]["run_id"] == "python-writer-parallel-local-s3"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="python",
        s3_prefix=s3_prefix,
    )
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert sum(w.row_count for w in result.winners) == 24


def test_python_writer_parallel_retry_publishes_succeeded_run_record_to_local_s3(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/python-writer-retry"
    file_backed_root = str(tmp_path / "file-backed-retry")

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
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
            run_id="python-writer-retry-local-s3",
            local_root=str(tmp_path / "local-retry"),
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

    result = write_sharded(
        list(range(24)),
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
        parallel=True,
    )

    assert len(result.winners) == 4
    assert any(winner.attempt == 1 for winner in result.winners)
    assert result.run_record_ref is not None

    run_record = load_s3_run_record(local_s3_service, result.run_record_ref)
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="python",
        s3_prefix=s3_prefix,
    )
