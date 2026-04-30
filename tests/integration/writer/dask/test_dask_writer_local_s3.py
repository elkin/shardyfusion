"""Integration test: Dask writer → moto S3 → verify manifest and CURRENT."""

from __future__ import annotations

import json
from datetime import timedelta

import dask.dataframe as dd
import pandas as pd

from shardyfusion.config import (
    HashWriteConfig,
    ManifestOptions,
    OutputOptions,
)
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import parse_manifest_payload
from shardyfusion.serde import ValueSpec
from shardyfusion.testing import FailOnceAdapterFactory, file_backed_adapter_factory
from shardyfusion.type_defs import RetryConfig, S3ConnectionOptions
from shardyfusion.writer.dask import write_sharded_by_hash
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_s3_run_record,
)


def test_dask_writer_publishes_manifest_and_current_to_local_s3(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/dask-writer"
    file_backed_root = str(tmp_path / "file-backed")
    credential_provider = StaticCredentialProvider(
        access_key_id=local_s3_service["access_key_id"],
        secret_access_key=local_s3_service["secret_access_key"],
    )
    connection_options = S3ConnectionOptions(
        endpoint_url=endpoint_url,
        region_name=local_s3_service["region_name"],
    )

    config = HashWriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        credential_provider=credential_provider,
        s3_connection_options=connection_options,
        adapter_factory=file_backed_adapter_factory(file_backed_root),
        output=OutputOptions(
            run_id="dask-writer-local-s3",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(
            credential_provider=credential_provider,
            s3_connection_options=connection_options,
        ),
    )

    pdf = pd.DataFrame({"id": list(range(24)), "val": [f"v{i}" for i in range(24)]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    result = write_sharded_by_hash(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: str(row["val"]).encode()),
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/dask-writer/manifests/")

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "dask-writer/_CURRENT"

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest = parse_manifest_payload(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))
    run_record = load_s3_run_record(local_s3_service, result.run_record_ref)

    assert manifest.required_build.run_id == "dask-writer-local-s3"
    assert manifest.required_build.num_dbs == 4
    assert len(manifest.shards) == 4
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="dask",
        s3_prefix=s3_prefix,
    )
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert sum(w.row_count for w in result.winners) == 24


def test_dask_writer_retry_publishes_succeeded_run_record_to_local_s3(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/dask-writer-retry"
    file_backed_root = str(tmp_path / "file-backed-retry")
    credential_provider = StaticCredentialProvider(
        access_key_id=local_s3_service["access_key_id"],
        secret_access_key=local_s3_service["secret_access_key"],
    )
    connection_options = S3ConnectionOptions(
        endpoint_url=endpoint_url,
        region_name=local_s3_service["region_name"],
    )

    config = HashWriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        credential_provider=credential_provider,
        s3_connection_options=connection_options,
        adapter_factory=FailOnceAdapterFactory(
            file_backed_adapter_factory(file_backed_root),
            marker_root=str(tmp_path / "retry-markers"),
            fail_db_ids=(0,),
        ),
        shard_retry=RetryConfig(
            max_retries=1,
            initial_backoff=timedelta(seconds=0),
        ),
        output=OutputOptions(
            run_id="dask-writer-retry-local-s3",
            local_root=str(tmp_path / "local-retry"),
        ),
        manifest=ManifestOptions(
            credential_provider=credential_provider,
            s3_connection_options=connection_options,
        ),
    )

    pdf = pd.DataFrame({"id": list(range(24)), "val": [f"v{i}" for i in range(24)]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    result = write_sharded_by_hash(
        ddf,
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
        writer_type="dask",
        s3_prefix=s3_prefix,
    )
