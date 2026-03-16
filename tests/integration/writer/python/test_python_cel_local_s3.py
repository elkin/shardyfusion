"""Integration: Python writer with CEL sharding → moto S3 → verify manifest."""

from __future__ import annotations

import pytest
import yaml

cel_expr_python = pytest.importorskip("cel_expr_python")  # noqa: F841
fastdigest = pytest.importorskip("fastdigest")  # noqa: F841

from shardyfusion.cel import compile_cel, route_cel
from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import parse_manifest
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from shardyfusion.testing import file_backed_adapter_factory, file_backed_load_db
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.python import write_sharded

pytestmark = pytest.mark.cel


def _cel_config(local_s3_service, tmp_path, *, run_id, s3_subpath, **overrides):
    bucket = local_s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/{s3_subpath}"
    root = str(tmp_path / "file-backed")
    defaults = dict(
        num_dbs=4,
        s3_prefix=s3_prefix,
        key_encoding=KeyEncoding.U64BE,
        adapter_factory=file_backed_adapter_factory(root),
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key % 4",
            cel_columns={"key": "int"},
            boundaries=[1, 2, 3],
        ),
        output=OutputOptions(run_id=run_id, local_root=str(tmp_path / "local")),
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
    defaults.update(overrides)
    return WriteConfig(**defaults), root


def test_cel_unified_publishes_manifest_to_s3(local_s3_service, tmp_path):
    """CEL unified mode: write → S3 manifest → verify CEL metadata."""
    config, root = _cel_config(
        local_s3_service, tmp_path, run_id="cel-unified", s3_subpath="cel-uni"
    )

    result = write_sharded(
        list(range(100)),
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
    )

    assert len(result.winners) == 4
    assert sum(w.row_count for w in result.winners) == 100

    # Verify manifest on S3 contains CEL metadata
    bucket = local_s3_service["bucket"]
    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())

    assert manifest_payload["required"]["sharding"]["strategy"] == "cel"
    assert manifest_payload["required"]["sharding"]["cel_expr"] == "key % 4"
    assert manifest_payload["required"]["sharding"]["cel_columns"] == {"key": "int"}
    assert manifest_payload["required"]["sharding"]["boundaries"] == [1, 2, 3]


def test_cel_unified_router_round_trip(local_s3_service, tmp_path):
    """Write with CEL → load manifest → build router → verify routing matches."""
    config, root = _cel_config(
        local_s3_service, tmp_path, run_id="cel-rt", s3_subpath="cel-rt"
    )

    result = write_sharded(
        list(range(40)),
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: b"v",
    )

    # Load manifest from S3 and reconstruct router
    bucket = local_s3_service["bucket"]
    client = local_s3_service["client"]
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    raw = client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    manifest = parse_manifest(raw)
    router = SnapshotRouter(manifest.required_build, manifest.shards)

    # Verify every key routes to the correct shard
    compiled = compile_cel("key % 4", {"key": "int"})
    for key in range(40):
        write_db_id = route_cel(compiled, {"key": key}, [1, 2, 3])
        read_db_id = router.route_one(key)
        assert write_db_id == read_db_id, f"key={key}"


def test_cel_split_mode_routes_by_context(local_s3_service, tmp_path):
    """CEL split mode: route by 'region' context, store by user ID."""
    bucket = local_s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/cel-split"
    root = str(tmp_path / "file-backed")

    config = WriteConfig(
        num_dbs=3,
        s3_prefix=s3_prefix,
        key_encoding=KeyEncoding.UTF8,
        adapter_factory=file_backed_adapter_factory(root),
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="region",
            cel_columns={"region": "string"},
            boundaries=["eu", "us"],
        ),
        output=OutputOptions(run_id="cel-split", local_root=str(tmp_path / "local")),
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

    records = [
        ("alice", "ap"),
        ("bob", "eu"),
        ("carol", "us"),
        ("dave", "ap"),
        ("eve", "eu"),
        ("frank", "us"),
    ]

    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r[0],
        value_fn=lambda r: f"{r[0]}@{r[1]}".encode(),
        columns_fn=lambda r: {"region": r[1]},
    )

    assert result.stats.rows_written == 6
    assert len(result.winners) == 3

    # Verify router from manifest
    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    raw = (
        local_s3_service["client"]
        .get_object(Bucket=bucket, Key=manifest_key)["Body"]
        .read()
    )
    manifest = parse_manifest(raw)
    router = SnapshotRouter(manifest.required_build, manifest.shards)

    assert router.route_with_context({"region": "ap"}) == 0
    assert router.route_with_context({"region": "eu"}) == 1
    assert router.route_with_context({"region": "us"}) == 2


def test_cel_data_integrity(local_s3_service, tmp_path):
    """Every record written with CEL can be read back from the correct shard."""
    config, root = _cel_config(
        local_s3_service, tmp_path, run_id="cel-integ", s3_subpath="cel-integ"
    )

    records = list(range(200))
    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"val-{r}".encode(),
    )

    from shardyfusion.serde import make_key_encoder

    encoder = make_key_encoder(config.key_encoding)

    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root, winner.db_url)
        all_kv.update(shard_data)

    assert len(all_kv) == 200
    for r in records:
        assert all_kv[encoder(r)] == f"val-{r}".encode()
