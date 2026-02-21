from __future__ import annotations

import json

import slatedb

from slatedb_spark_sharded.manifest import (
    CurrentPointer,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from slatedb_spark_sharded.reader import SlateShardedReader
from slatedb_spark_sharded.sharding import ShardingStrategy


def test_reader_loads_current_and_manifest_from_local_s3(
    tmp_path, local_s3_service
) -> None:
    bucket = local_s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/reader-only"
    manifest_ref = f"{s3_prefix}/manifests/run_id=reader-local/manifest"
    current_ref = f"{s3_prefix}/_CURRENT"
    local_root = tmp_path / "reader-cache"
    object_store_root = tmp_path / "object-store"
    object_store_root.mkdir(parents=True, exist_ok=True)

    db0_local = local_root / "shard=00000"
    db1_local = local_root / "shard=00001"
    db0_local.mkdir(parents=True, exist_ok=True)
    db1_local.mkdir(parents=True, exist_ok=True)
    db0_url = f"file://{(object_store_root / 'db0').as_posix()}"
    db1_url = f"file://{(object_store_root / 'db1').as_posix()}"

    db0 = slatedb.SlateDB(str(db0_local), url=db0_url)
    db0.put((1).to_bytes(8, "big", signed=False), b"v1")
    db0.put((8).to_bytes(8, "big", signed=False), b"v8")
    db0.flush_with_options("wal")
    db0_ckpt = db0.create_checkpoint(scope="durable")["id"]
    db0.close()

    db1 = slatedb.SlateDB(str(db1_local), url=db1_url)
    db1.put((10).to_bytes(8, "big", signed=False), b"v10")
    db1.put((15).to_bytes(8, "big", signed=False), b"v15")
    db1.flush_with_options("wal")
    db1_ckpt = db1.create_checkpoint(scope="durable")["id"]
    db1.close()

    required = RequiredBuildMeta(
        run_id="reader-local",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
        s3_prefix=s3_prefix,
        key_col="id",
        key_encoding="u64be",
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[10]),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url=db0_url,
            attempt=0,
            row_count=2,
            min_key=0,
            max_key=9,
            checkpoint_id=db0_ckpt,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=1,
            db_url=db1_url,
            attempt=0,
            row_count=2,
            min_key=10,
            max_key=None,
            checkpoint_id=db1_ckpt,
            writer_info={},
        ),
    ]
    manifest_payload = json.dumps(
        {
            "required": required.model_dump(mode="json"),
            "shards": [shard.model_dump(mode="json") for shard in shards],
            "custom": {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    current_payload = json.dumps(
        CurrentPointer(
            manifest_ref=manifest_ref,
            manifest_content_type="application/json",
            run_id="reader-local",
            updated_at="2026-01-01T00:00:00+00:00",
        ).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    client = local_s3_service["client"]
    client.put_object(
        Bucket=bucket,
        Key=manifest_ref.split(f"s3://{bucket}/", 1)[1],
        Body=manifest_payload,
        ContentType="application/json",
    )
    client.put_object(
        Bucket=bucket,
        Key=current_ref.split(f"s3://{bucket}/", 1)[1],
        Body=current_payload,
        ContentType="application/json",
    )

    reader = SlateShardedReader(
        s3_prefix=s3_prefix,
        local_root=str(local_root),
    )
    try:
        assert reader.get(1) == b"v1"
        assert reader.get(10) == b"v10"
        got = reader.multi_get([8, 15, 1, 10])
        assert got[8] == b"v8"
        assert got[15] == b"v15"
        assert got[1] == b"v1"
        assert got[10] == b"v10"
    finally:
        reader.close()
