from __future__ import annotations

from dataclasses import asdict
import json

import pytest
import slatedb

from slatedb_spark_sharded.manifest import (
    CurrentPointer,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from slatedb_spark_sharded.manifest_readers import ManifestReader, parse_json_manifest
from slatedb_spark_sharded.reader import SlateShardedReader
from slatedb_spark_sharded.sharding import ShardingSpec, ShardingStrategy


class CustomPublisher:
    pass


class InMemoryManifestReader(ManifestReader):
    def __init__(
        self,
        *,
        current_ref: str,
        pointers: dict[str, bytes],
        manifests: dict[str, bytes],
    ) -> None:
        self.current_ref = current_ref
        self.pointers = pointers
        self.manifests = manifests

    def load_current(self) -> CurrentPointer | None:
        payload = self.pointers.get(self.current_ref)
        if payload is None:
            return None
        obj = json.loads(payload.decode("utf-8"))
        return CurrentPointer(
            manifest_ref=obj["manifest_ref"],
            manifest_content_type=obj["manifest_content_type"],
            run_id=obj["run_id"],
            updated_at=obj["updated_at"],
            format_version=int(obj.get("format_version", 1)),
        )

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        _ = content_type
        return parse_json_manifest(self.manifests[ref])


def test_sharded_reader_get_and_multi_get_with_custom_manifest_reader(tmp_path) -> None:
    local_root = tmp_path / "reader-cache"
    object_store_root = tmp_path / "object-store"
    object_store_root.mkdir(parents=True, exist_ok=True)

    db0_local = local_root / "shard=00000"
    db1_local = local_root / "shard=00001"
    db2_local = local_root / "shard=00002"
    db0_local.mkdir(parents=True, exist_ok=True)
    db1_local.mkdir(parents=True, exist_ok=True)
    db2_local.mkdir(parents=True, exist_ok=True)

    db0_url = f"file://{(object_store_root / 'db0').as_posix()}"
    db1_url = f"file://{(object_store_root / 'db1').as_posix()}"
    db2_url = f"file://{(object_store_root / 'db2').as_posix()}"

    db0 = slatedb.SlateDB(str(db0_local), url=db0_url)
    db0.put((1).to_bytes(8, "big", signed=False), b"v1")
    db0.put((9).to_bytes(8, "big", signed=False), b"v9")
    db0.flush_with_options("wal")
    db0_ckpt = db0.create_checkpoint(scope="durable")["id"]
    db0.close()

    db1 = slatedb.SlateDB(str(db1_local), url=db1_url)
    db1.put((10).to_bytes(8, "big", signed=False), b"v10")
    db1.put((15).to_bytes(8, "big", signed=False), b"v15")
    db1.flush_with_options("wal")
    db1_ckpt = db1.create_checkpoint(scope="durable")["id"]
    db1.close()

    db2 = slatedb.SlateDB(str(db2_local), url=db2_url)
    db2.put((20).to_bytes(8, "big", signed=False), b"v20")
    db2.put((27).to_bytes(8, "big", signed=False), b"v27")
    db2.flush_with_options("wal")
    db2_ckpt = db2.create_checkpoint(scope="durable")["id"]
    db2.close()

    required = RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=3,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding="u64be",
        sharding=ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[10, 20]),
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
            max_key=19,
            checkpoint_id=db1_ckpt,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=2,
            db_url=db2_url,
            attempt=0,
            row_count=2,
            min_key=20,
            max_key=None,
            checkpoint_id=db2_ckpt,
            writer_info={},
        ),
    ]

    manifest_payload = json.dumps(
        {
            "required": asdict(required),
            "shards": [asdict(shard) for shard in shards],
            "custom": {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    pointers = {
        "mem://current": json.dumps(
            {
                "format_version": 1,
                "manifest_ref": "mem://manifest/1",
                "manifest_content_type": "application/json",
                "run_id": "run-1",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    }
    manifests = {"mem://manifest/1": manifest_payload}

    reader = SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(local_root),
        publisher=CustomPublisher(),
        manifest_reader=InMemoryManifestReader(
            current_ref="mem://current",
            pointers=pointers,
            manifests=manifests,
        ),
        max_workers=4,
    )
    try:
        assert reader.get(15) == b"v15"
        got = reader.multi_get([1, 20, 10, 27, 9])
        assert got[1] == b"v1"
        assert got[20] == b"v20"
        assert got[10] == b"v10"
        assert got[27] == b"v27"
        assert got[9] == b"v9"
    finally:
        reader.close()
