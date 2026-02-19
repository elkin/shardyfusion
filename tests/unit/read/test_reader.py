from __future__ import annotations

import sys
import types
from dataclasses import dataclass

from slatedb_spark_sharded.manifest import (
    CurrentPointer,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from slatedb_spark_sharded.reader import SlateShardedReader, _open_slatedb_reader
from slatedb_spark_sharded.sharding import ShardingSpec, ShardingStrategy


class _MutableManifestReader:
    def __init__(self, manifests: dict[str, ParsedManifest], initial_ref: str) -> None:
        self.manifests = manifests
        self.current_ref = initial_ref

    def load_current(self) -> CurrentPointer | None:
        return CurrentPointer(
            manifest_ref=self.current_ref,
            manifest_content_type="application/json",
            run_id="run",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        _ = content_type
        return self.manifests[ref]


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        return None


def _required_build() -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding="u64be",
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
    )


def _manifest(db_url: str) -> ParsedManifest:
    return ParsedManifest(
        required_build=_required_build(),
        shards=[
            RequiredShardMeta(
                db_id=0,
                db_url=db_url,
                attempt=0,
                row_count=1,
                min_key=None,
                max_key=None,
                checkpoint_id=None,
                writer_info={},
            )
        ],
        custom={},
    )


def test_refresh_swaps_manifest_ref_and_readers(monkeypatch, tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
        "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
    }

    def fake_open_reader(*, local_path, db_url, checkpoint_id, env_file):
        _ = (local_path, checkpoint_id, env_file)
        return _FakeReader(stores[db_url])

    monkeypatch.setattr(
        "slatedb_spark_sharded.reader._open_slatedb_reader", fake_open_reader
    )

    reader = SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
    )
    try:
        assert reader.get(1) == b"one"
        manifest_reader.current_ref = "mem://manifest/two"

        changed = reader.refresh()
        assert changed is True
        assert reader.get(1) == b"two"

        unchanged = reader.refresh()
        assert unchanged is False
    finally:
        reader.close()


def test_open_slatedb_reader_uses_official_slatedbreader_signature(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = _FakeReader({})

    def fake_reader_ctor(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    fake_module = types.ModuleType("slatedb")
    fake_module.SlateDBReader = fake_reader_ctor
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    reader = _open_slatedb_reader(
        local_path="/tmp/local",
        db_url="s3://bucket/db",
        checkpoint_id="ckpt-1",
        env_file="slatedb.env",
    )

    assert reader is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("/tmp/local",)
    assert kwargs["url"] == "s3://bucket/db"
    assert kwargs["checkpoint_id"] == "ckpt-1"
    assert kwargs["env_file"] == "slatedb.env"
