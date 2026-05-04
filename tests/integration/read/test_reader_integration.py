from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from shardyfusion._checkpoint_id import generate_checkpoint_id
from shardyfusion._slatedb_runtime import run_coro
from shardyfusion._slatedb_symbols import get_flush_options_classes, get_writer_symbols
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader import ConcurrentShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.testing import open_slatedb_db


class InMemoryManifestStore:
    """In-memory manifest store for reader integration tests."""

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

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        raise NotImplementedError("publish not used in reader tests")

    def load_current(self) -> ManifestRef | None:
        payload = self.pointers.get(self.current_ref)
        if payload is None:
            return None
        obj = json.loads(payload.decode("utf-8"))
        return ManifestRef(
            ref=obj["manifest_ref"],
            run_id=obj["run_id"],
            published_at=datetime.fromisoformat(obj["updated_at"]),
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        from shardyfusion.manifest_store import parse_manifest_payload

        return parse_manifest_payload(self.manifests[ref])

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass


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

    # With xxh3 hash routing and num_dbs=3:
    #   shard 0: keys 4, 9
    #   shard 1: keys 2, 15
    #   shard 2: keys 1, 10
    _, _, write_batch_cls, _ = get_writer_symbols()
    flush_options_cls, flush_type_cls = get_flush_options_classes()

    def _seed(db_url: str, items: list[tuple[int, bytes]]) -> str:
        db = open_slatedb_db(db_url)
        try:
            wb = write_batch_cls()
            for k, v in items:
                wb.put(k.to_bytes(8, "big", signed=False), v)
            run_coro(db.write(wb))
            run_coro(
                db.flush_with_options(flush_options_cls(flush_type=flush_type_cls.WAL))
            )
        finally:
            run_coro(db.shutdown())
        return generate_checkpoint_id()

    db0_ckpt = _seed(db0_url, [(4, b"v4"), (9, b"v9")])
    db1_ckpt = _seed(db1_url, [(2, b"v2"), (15, b"v15")])
    db2_ckpt = _seed(db2_url, [(1, b"v1"), (10, b"v10")])

    required = RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=3,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url=db0_url,
            attempt=0,
            row_count=2,
            min_key=None,
            max_key=None,
            checkpoint_id=db0_ckpt,
            writer_info={},
            db_bytes=0,
        ),
        RequiredShardMeta(
            db_id=1,
            db_url=db1_url,
            attempt=0,
            row_count=2,
            min_key=None,
            max_key=None,
            checkpoint_id=db1_ckpt,
            writer_info={},
            db_bytes=0,
        ),
        RequiredShardMeta(
            db_id=2,
            db_url=db2_url,
            attempt=0,
            row_count=2,
            min_key=None,
            max_key=None,
            checkpoint_id=db2_ckpt,
            writer_info={},
            db_bytes=0,
        ),
    ]

    from shardyfusion.manifest import SqliteManifestBuilder

    manifest_payload = (
        SqliteManifestBuilder()
        .build(required_build=required, shards=shards, custom_fields={})
        .payload
    )

    pointers = {
        "mem://current": json.dumps(
            {
                "format_version": 2,
                "manifest_ref": "mem://manifest/1",
                "manifest_content_type": "application/x-sqlite3",
                "run_id": "run-1",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    }
    manifests = {"mem://manifest/1": manifest_payload}

    with ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(local_root),
        manifest_store=InMemoryManifestStore(
            current_ref="mem://current",
            pointers=pointers,
            manifests=manifests,
        ),
        max_workers=4,
    ) as reader:
        assert reader.get(15) == b"v15"
        got = reader.multi_get([1, 2, 10, 4, 9])
        assert got[1] == b"v1"
        assert got[2] == b"v2"
        assert got[10] == b"v10"
        assert got[4] == b"v4"
        assert got[9] == b"v9"
