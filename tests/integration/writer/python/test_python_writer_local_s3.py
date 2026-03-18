"""Integration test: pure-Python writer → moto S3 → verify manifest and CURRENT."""

from __future__ import annotations

import json

import yaml

from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.testing import file_backed_adapter_factory
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.python import write_sharded


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

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "python-writer/_CURRENT"

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "python-writer-local-s3"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert sum(w.row_count for w in result.winners) == 24
