"""Integration test: Dask single-db writer → moto S3 → verify manifest and CURRENT."""

from __future__ import annotations

import json

import pandas as pd
import pytest

dd = pytest.importorskip("dask.dataframe")
import dask  # noqa: E402

from slatedb_spark_sharded.config import (  # noqa: E402
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from slatedb_spark_sharded.serde import ValueSpec  # noqa: E402
from slatedb_spark_sharded.testing import file_backed_adapter_factory  # noqa: E402
from slatedb_spark_sharded.writer.dask.single_db_writer import (  # noqa: E402
    write_single_db,
)


@pytest.fixture(autouse=True)
def _synchronous_scheduler():
    with dask.config.set(scheduler="synchronous"):
        yield


def test_single_db_dask_publishes_manifest_and_current(
    local_s3_service, tmp_path
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/single-db-dask"
    file_backed_root = str(tmp_path / "file-backed")

    config = WriteConfig(
        num_dbs=1,
        s3_prefix=s3_prefix,
        adapter_factory=file_backed_adapter_factory(file_backed_root),
        output=OutputOptions(
            run_id="single-db-dask-test",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(
            s3_client_config={
                "endpoint_url": endpoint_url,
                "region_name": local_s3_service["region_name"],
                "access_key_id": local_s3_service["access_key_id"],
                "secret_access_key": local_s3_service["secret_access_key"],
            }
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
    assert result.current_ref == f"s3://{bucket}/single-db-dask/_CURRENT"

    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = result.current_ref.split(f"s3://{bucket}/", 1)[1]

    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = json.loads(manifest_obj["Body"].read().decode("utf-8"))
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "single-db-dask-test"
    assert manifest_payload["required"]["num_dbs"] == 1
    assert len(manifest_payload["shards"]) == 1
    assert current_payload["manifest_ref"] == result.manifest_ref
