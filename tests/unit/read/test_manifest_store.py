from __future__ import annotations

import json
import sqlite3

import pytest

from shardyfusion.errors import ManifestParseError
from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    SqliteManifestBuilder,
)
from shardyfusion.manifest_store import (
    S3ManifestStore,
    parse_sqlite_manifest,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


def _make_build(num_dbs: int = 2) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
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


def _make_shards(ids: list[int]) -> list[RequiredShardMeta]:
    return [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/db={i:05d}",
            attempt=0,
            row_count=1,
            min_key=i + 1,
            max_key=i + 1,
            checkpoint_id=None,
            db_bytes=0,
        )
        for i in ids
    ]


def _to_sqlite(
    build: RequiredBuildMeta,
    shards: list[RequiredShardMeta],
    custom: dict | None = None,
) -> bytes:
    builder = SqliteManifestBuilder()
    return builder.build(
        required_build=build, shards=shards, custom_fields=custom or {}
    ).payload


def _tamper(payload: bytes, sql: str) -> bytes:
    con = sqlite3.connect(":memory:")
    con.deserialize(payload)
    con.execute(sql)
    con.commit()
    result = con.serialize()
    con.close()
    return result


def test_parse_manifest_round_trip() -> None:
    build = _make_build()
    shards = _make_shards([0, 1])
    payload = _to_sqlite(build, shards, custom={"env": "test"})

    parsed = parse_sqlite_manifest(payload)

    assert parsed.required_build.num_dbs == 2
    assert [shard.db_id for shard in parsed.shards] == [0, 1]
    assert parsed.custom == {"env": "test"}


def test_parse_manifest_round_trip_categorical_cel() -> None:
    build = RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=3,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="region",
            cel_columns={"region": "string"},
            routing_values=["ap", "eu", "us"],
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
        format_version=4,
    )
    shards = _make_shards([0, 1, 2])
    payload = _to_sqlite(build, shards)

    parsed = parse_sqlite_manifest(payload)
    assert parsed.required_build.format_version == 4
    assert parsed.required_build.sharding.routing_values == ["ap", "eu", "us"]


def test_parse_manifest_rejects_bad_shard_coverage() -> None:
    build = _make_build(num_dbs=1)
    shards = _make_shards([0])
    payload = _to_sqlite(build, shards)
    # Add extra shard beyond num_dbs
    payload = _tamper(
        payload,
        "INSERT INTO shards (db_id, db_url, attempt, row_count, db_bytes, writer_info)"
        " VALUES (1, 's3://x', 0, 1, 0, '{}')",
    )

    with pytest.raises(ManifestParseError, match="exceeds num_dbs"):
        parse_sqlite_manifest(payload)


def test_parse_manifest_rejects_corrupt_sqlite() -> None:
    with pytest.raises(ManifestParseError):
        parse_sqlite_manifest(b"SQLite format 3\x00" + b"\x00" * 84)


def test_parse_manifest_rejects_missing_build_meta() -> None:
    build = _make_build()
    payload = _to_sqlite(build, [])
    payload = _tamper(payload, "DELETE FROM build_meta")
    with pytest.raises(ManifestParseError, match="no build_meta row"):
        parse_sqlite_manifest(payload)


def test_load_current_rejects_corrupt_json(monkeypatch) -> None:
    def fake_try_get_bytes(url, *, s3_client=None, metrics_collector=None, **kwargs):
        return b"not-json{{{"

    monkeypatch.setattr("shardyfusion.manifest_store.try_get_bytes", fake_try_get_bytes)
    store = S3ManifestStore("s3://bucket/prefix")
    with pytest.raises(ManifestParseError, match="validation failed"):
        store.load_current()


def test_load_current_rejects_missing_manifest_ref(monkeypatch) -> None:
    def fake_try_get_bytes(url, *, s3_client=None, metrics_collector=None, **kwargs):
        return json.dumps(
            {
                "manifest_content_type": "application/x-sqlite3",
                "run_id": "run-1",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ).encode("utf-8")

    monkeypatch.setattr("shardyfusion.manifest_store.try_get_bytes", fake_try_get_bytes)
    store = S3ManifestStore("s3://bucket/prefix")
    with pytest.raises(ManifestParseError, match="manifest_ref"):
        store.load_current()
