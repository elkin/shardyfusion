from __future__ import annotations

from dataclasses import asdict, dataclass
import json

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

    def load_manifest(self, ref: str, content_type: str | None = None) -> ParsedManifest:
        _ = content_type
        return parse_json_manifest(self.manifests[ref])


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        return None


def test_sharded_reader_get_and_multi_get_with_custom_manifest_reader(monkeypatch, tmp_path) -> None:
    required = RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=3,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding="u64be",
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE, boundaries=[10, 20]
        ),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url="mem://db/0",
            attempt=0,
            row_count=2,
            min_key=0,
            max_key=9,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=1,
            db_url="mem://db/1",
            attempt=0,
            row_count=2,
            min_key=10,
            max_key=19,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=2,
            db_url="mem://db/2",
            attempt=0,
            row_count=2,
            min_key=20,
            max_key=None,
            checkpoint_id=None,
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

    kv_by_url = {
        "mem://db/0": {
            (1).to_bytes(8, "big", signed=False): b"v1",
            (9).to_bytes(8, "big", signed=False): b"v9",
        },
        "mem://db/1": {
            (10).to_bytes(8, "big", signed=False): b"v10",
            (15).to_bytes(8, "big", signed=False): b"v15",
        },
        "mem://db/2": {
            (20).to_bytes(8, "big", signed=False): b"v20",
            (27).to_bytes(8, "big", signed=False): b"v27",
        },
    }

    def fake_open_reader(*, local_path, db_url, checkpoint_id, env_file, settings):
        _ = (local_path, checkpoint_id, env_file, settings)
        return _FakeReader(kv_by_url[db_url])

    monkeypatch.setattr("slatedb_spark_sharded.reader._open_slatedb_reader", fake_open_reader)

    reader = SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
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
