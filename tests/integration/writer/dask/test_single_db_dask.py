"""Integration test: Dask single-db writer → moto S3 → verify manifest and CURRENT."""

from __future__ import annotations

import json

import dask.dataframe as dd
import pandas as pd

from shardyfusion.config import (
    HashShardedWriteConfig,
    WriterManifestConfig,
    WriterOutputConfig,
)
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import parse_manifest_payload
from shardyfusion.serde import ValueSpec
from shardyfusion.testing import file_backed_adapter_factory
from shardyfusion.type_defs import S3ConnectionOptions
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_s3_run_record,
)
from tests.helpers.writer_api import write_dask_single_db as write_single_db


def test_single_db_dask_publishes_manifest_and_current(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/single-db-dask"
    file_backed_root = str(tmp_path / "file-backed")
    credential_provider = StaticCredentialProvider(
        access_key_id=local_s3_service["access_key_id"],
        secret_access_key=local_s3_service["secret_access_key"],
    )
    connection_options = S3ConnectionOptions(
        endpoint_url=endpoint_url,
        region_name=local_s3_service["region_name"],
    )

    config = HashShardedWriteConfig(
        num_dbs=1,
        s3_prefix=s3_prefix,
        credential_provider=credential_provider,
        s3_connection_options=connection_options,
        adapter_factory=file_backed_adapter_factory(file_backed_root),
        output=WriterOutputConfig(
            run_id="single-db-dask-test",
            local_root=str(tmp_path / "local"),
        ),
        manifest=WriterManifestConfig(
            credential_provider=credential_provider,
            s3_connection_options=connection_options,
        ),
    )

    pdf = pd.DataFrame({"id": list(range(20)), "val": [f"val-{i}" for i in range(20)]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    result = write_single_db(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: str(row["val"]).encode()),
    )

    assert len(result.winners) == 1
    assert result.winners[0].row_count == 20
    assert result.manifest_ref.startswith(f"s3://{bucket}/single-db-dask/manifests/")

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "single-db-dask/_CURRENT"

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest = parse_manifest_payload(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))
    run_record = load_s3_run_record(local_s3_service, result.run_record_ref)

    assert manifest.required_build.run_id == "single-db-dask-test"
    assert manifest.required_build.num_dbs == 1
    assert len(manifest.shards) == 1
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="dask",
        s3_prefix=s3_prefix,
    )
    assert current_payload["manifest_ref"] == result.manifest_ref
