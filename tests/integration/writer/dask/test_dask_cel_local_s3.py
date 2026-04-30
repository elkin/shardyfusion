"""Integration: Dask writer with CEL sharding → moto S3 → verify manifest."""

from __future__ import annotations

import pytest

cel_expr_python = pytest.importorskip("cel_expr_python")  # noqa: F841

import dask.dataframe as dd
import pandas as pd

from shardyfusion.cel import compile_cel, route_cel
from shardyfusion.config import CelWriteConfig, ManifestOptions, OutputOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import parse_manifest_payload
from shardyfusion.routing import SnapshotRouter
from shardyfusion.serde import ValueSpec
from shardyfusion.testing import file_backed_adapter_factory
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.dask import write_sharded_by_cel

pytestmark = pytest.mark.cel


def test_dask_cel_unified_publishes_manifest(local_s3_service, tmp_path):
    bucket = local_s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/dask-cel"
    root = str(tmp_path / "file-backed")

    config = CelWriteConfig(
        s3_prefix=s3_prefix,
        adapter_factory=file_backed_adapter_factory(root),
        cel_expr="key % 4",
        cel_columns={"key": "int"},
        output=OutputOptions(
            run_id="dask-cel",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(
            credential_provider=StaticCredentialProvider(
                access_key_id=local_s3_service["access_key_id"],
                secret_access_key=local_s3_service["secret_access_key"],
            ),
            s3_connection_options=S3ConnectionOptions(
                endpoint_url=local_s3_service["endpoint_url"],
                region_name=local_s3_service["region_name"],
            ),
        ),
    )

    pdf = pd.DataFrame({"key": list(range(40)), "val": [f"v{i}" for i in range(40)]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    result = write_sharded_by_cel(
        ddf,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: str(row["val"]).encode()),
    )

    assert len(result.winners) == 4
    assert sum(w.row_count for w in result.winners) == 40

    # Verify manifest on S3
    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    raw = client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    manifest = parse_manifest_payload(raw)

    assert manifest.required_build.sharding.strategy == "cel"
    assert manifest.required_build.sharding.cel_expr == "key % 4"

    # Router round-trip
    router = SnapshotRouter(manifest.required_build, manifest.shards)

    compiled = compile_cel("key % 4", {"key": "int"})
    for key in range(40):
        assert route_cel(compiled, {"key": key}) == router.route_one(key)
