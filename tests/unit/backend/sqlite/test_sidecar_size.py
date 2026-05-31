"""Tests for recording each shard's decompressed page-cache sidecar size.

The writer captures the exact ``len(body)`` of the v5 sidecar at ``seal()`` and
threads it into ``RequiredShardMeta.sidecar_decompressed_bytes`` so a reader can
size a download before fetching + decompressing the sidecar.  These tests cover
the adapter accessor and the manifest round-trip (including the
backward-compatible "old manifest, no column" path).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    SqliteManifestBuilder,
)
from shardyfusion.manifest_store import parse_manifest_payload


def _build_meta() -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        key_col="_key",
        sharding=ManifestShardingSpec(hash_algorithm="xxh3_64"),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )


def _shard(sidecar: int | None) -> RequiredShardMeta:
    return RequiredShardMeta(
        db_id=0,
        db_url="s3://bucket/prefix/shards/db=00000",
        attempt=0,
        row_count=10,
        db_bytes=4096,
        sidecar_decompressed_bytes=sidecar,
    )


class TestManifestRoundTrip:
    @pytest.mark.parametrize("size", [12345, None])
    def test_sidecar_size_round_trips(self, size: int | None) -> None:
        artifact = SqliteManifestBuilder().build(
            required_build=_build_meta(), shards=[_shard(size)], custom_fields={}
        )
        parsed = parse_manifest_payload(artifact.payload)
        assert parsed.shards[0].sidecar_decompressed_bytes == size

    def test_parse_tolerates_old_manifest_without_column(self) -> None:
        """A manifest written before this field existed (its ``shards`` table has
        no ``sidecar_decompressed_bytes`` column) parses with the field
        defaulting to ``None`` rather than raising."""
        con = sqlite3.connect(":memory:")
        try:
            con.execute(
                "CREATE TABLE build_meta ("
                " run_id TEXT, created_at TEXT, num_dbs INTEGER, s3_prefix TEXT,"
                " key_col TEXT, sharding TEXT, db_path_template TEXT,"
                " shard_prefix TEXT, format_version INTEGER, key_encoding TEXT,"
                " custom TEXT)"
            )
            sharding = (
                '{"strategy": "hash", "hash_algorithm": "xxh3_64",'
                ' "routing_values": null, "cel_expr": null, "cel_columns": null}'
            )
            con.execute(
                "INSERT INTO build_meta VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "run-1",
                    "2026-01-01T00:00:00+00:00",
                    1,
                    "s3://bucket/prefix",
                    "_key",
                    sharding,
                    "db={db_id:05d}",
                    "shards",
                    4,
                    "u64be",
                    "{}",
                ),
            )
            # Old shards schema — deliberately no sidecar_decompressed_bytes column.
            con.execute(
                "CREATE TABLE shards ("
                " db_id INTEGER PRIMARY KEY, db_url TEXT, attempt INTEGER,"
                " row_count INTEGER, db_bytes INTEGER, min_key TEXT, max_key TEXT,"
                " checkpoint_id TEXT, writer_info TEXT NOT NULL DEFAULT '{}')"
            )
            con.execute(
                "INSERT INTO shards VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    0,
                    "s3://bucket/prefix/shards/db=00000",
                    0,
                    10,
                    4096,
                    None,
                    None,
                    None,
                    "{}",
                ),
            )
            con.commit()
            payload = con.serialize()
        finally:
            con.close()

        parsed = parse_manifest_payload(payload)
        assert parsed.shards[0].sidecar_decompressed_bytes is None


@pytest.fixture()
def _mock_backend() -> MagicMock:
    with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as m:
        inst = m.return_value
        inst.put = MagicMock()
        inst.head = MagicMock(return_value='"etag"')
        yield inst


class TestAdapterRecordsSize:
    def test_seal_records_exact_decompressed_size(
        self, tmp_path: Path, _mock_backend: MagicMock
    ) -> None:
        pytest.importorskip("apsw")
        zstandard = pytest.importorskip("zstandard")
        from shardyfusion.sqlite_adapter import SqliteAdapter

        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://bucket/shard", local_dir=local_dir) as adapter:
            # Enough rows that the kv btree has at least one interior page.
            adapter.write_batch(
                [(i.to_bytes(8, "big"), b"v" * 32) for i in range(5000)]
            )
            adapter.seal()
            size = adapter.sidecar_decompressed_bytes()

        assert isinstance(size, int)
        assert size > 0
        # The recorded size equals the decompressed body of the uploaded sidecar.
        keys = [c.args[0] for c in _mock_backend.put.call_args_list]
        idx = next(i for i, k in enumerate(keys) if k.endswith("/shard.sidecar"))
        blob = _mock_backend.put.call_args_list[idx].args[1]
        tag_len = blob[5]
        body = zstandard.ZstdDecompressor().decompress(blob[6 + tag_len :])
        assert size == len(body)

    def test_no_size_when_emit_sidecar_false(
        self, tmp_path: Path, _mock_backend: MagicMock
    ) -> None:
        from shardyfusion.sqlite_adapter import SqliteAdapter

        local_dir = tmp_path / "shard"
        with SqliteAdapter(
            db_url="s3://bucket/shard", local_dir=local_dir, emit_sidecar=False
        ) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.seal()
            assert adapter.sidecar_decompressed_bytes() is None
