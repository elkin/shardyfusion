"""Tests for manifest parse error paths (SQLite format)."""

from __future__ import annotations

import sqlite3

import pytest

from shardyfusion.errors import ManifestParseError
from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    SqliteManifestBuilder,
)
from shardyfusion.manifest_store import parse_manifest_payload, parse_sqlite_manifest
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


def _make_required_build(num_dbs: int = 2) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="test-run",
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


def _make_shards(num: int) -> list[RequiredShardMeta]:
    return [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/shards/db={i:05d}",
            attempt=0,
            row_count=100,
            db_bytes=0,
        )
        for i in range(num)
    ]


def _to_sqlite(build: RequiredBuildMeta, shards: list[RequiredShardMeta]) -> bytes:
    builder = SqliteManifestBuilder()
    artifact = builder.build(required_build=build, shards=shards, custom_fields={})
    return artifact.payload


def _tamper_sqlite(payload: bytes, sql: str) -> bytes:
    """Deserialize a SQLite manifest, run arbitrary SQL, re-serialize."""
    con = sqlite3.connect(":memory:")
    con.deserialize(payload)
    con.execute(sql)
    con.commit()
    result = con.serialize()
    con.close()
    return result


def test_valid_manifest_parses() -> None:
    payload = _to_sqlite(_make_required_build(), _make_shards(2))
    result = parse_sqlite_manifest(payload)
    assert result.required_build.num_dbs == 2
    assert len(result.shards) == 2


def test_non_sqlite_payload_rejected() -> None:
    with pytest.raises(ManifestParseError, match="expected SQLite"):
        parse_manifest_payload(b"not-sqlite-data")


def test_binary_garbage_rejected() -> None:
    with pytest.raises(ManifestParseError, match="expected SQLite"):
        parse_manifest_payload(b"\xff\xfe\x00\x01")


def test_empty_payload_rejected() -> None:
    with pytest.raises(ManifestParseError, match="expected SQLite"):
        parse_manifest_payload(b"")


def test_shard_count_exceeds_num_dbs() -> None:
    build = _make_required_build(num_dbs=2)
    shards = _make_shards(2)
    payload = _to_sqlite(build, shards)
    # Add extra shard beyond num_dbs
    payload = _tamper_sqlite(
        payload,
        "INSERT INTO shards (db_id, db_url, attempt, row_count, db_bytes, writer_info)"
        " VALUES (2, 's3://x', 0, 1, 0, '{}')",
    )
    with pytest.raises(ManifestParseError, match="exceeds num_dbs"):
        parse_sqlite_manifest(payload)


def test_sparse_shards_accepted() -> None:
    """Manifests with fewer shards than num_dbs are valid (empty shards omitted)."""
    build = _make_required_build(num_dbs=8)
    shards = _make_shards(5)
    payload = _to_sqlite(build, shards)
    result = parse_sqlite_manifest(payload)
    assert len(result.shards) == 5


def test_out_of_range_shard_ids() -> None:
    build = _make_required_build(num_dbs=3)
    shards = _make_shards(3)
    payload = _to_sqlite(build, shards)
    # Change shard 2's db_id to 5 (out of range for num_dbs=3)
    payload = _tamper_sqlite(payload, "UPDATE shards SET db_id = 5 WHERE db_id = 2")
    with pytest.raises(ManifestParseError, match="out of range"):
        parse_sqlite_manifest(payload)


def test_duplicate_shard_ids() -> None:
    build = _make_required_build(num_dbs=2)
    shards = _make_shards(1)  # Only shard 0
    payload = _to_sqlite(build, shards)
    # Recreate shards table without PK constraint to allow duplicate db_id
    con = sqlite3.connect(":memory:")
    con.deserialize(payload)
    con.execute("CREATE TABLE shards_new AS SELECT * FROM shards")
    con.execute("DROP TABLE shards")
    con.execute("ALTER TABLE shards_new RENAME TO shards")
    con.execute(
        "INSERT INTO shards (db_id, db_url, attempt, row_count, db_bytes, writer_info)"
        " VALUES (0, 's3://dup', 0, 1, 0, '{}')"
    )
    con.commit()
    payload = con.serialize()
    con.close()
    with pytest.raises(ManifestParseError, match="duplicate"):
        parse_sqlite_manifest(payload)


def test_missing_build_meta() -> None:
    build = _make_required_build()
    shards = _make_shards(2)
    payload = _to_sqlite(build, shards)
    payload = _tamper_sqlite(payload, "DELETE FROM build_meta")
    with pytest.raises(ManifestParseError, match="no build_meta row"):
        parse_sqlite_manifest(payload)


def test_corrupt_sqlite_payload() -> None:
    # Valid SQLite magic header but truncated/corrupt
    payload = b"SQLite format 3\x00" + b"\x00" * 84
    with pytest.raises(ManifestParseError):
        parse_sqlite_manifest(payload)


def _make_cel_build(num_dbs: int = 2, format_version: int = 4) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="test-run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="region",
            cel_columns={"region": "string"},
            routing_values=["ap", "eu"] if num_dbs == 2 else ["ap", "eu", "us"],
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
        format_version=format_version,
    )


def test_manifest_rejects_removed_boundaries_field() -> None:
    import json

    build = _make_required_build(num_dbs=3)
    shards = _make_shards(3)
    payload = _to_sqlite(build, shards)
    # Inject boundaries field into sharding JSON (extra="forbid" rejects it)
    sharding_with_boundaries = json.dumps(
        {
            "strategy": "cel",
            "cel_expr": "region",
            "cel_columns": {"region": "string"},
            "hash_algorithm": "xxh3_64",
            "boundaries": ["eu", "us"],
        }
    )
    payload = _tamper_sqlite(
        payload,
        f"UPDATE build_meta SET sharding = '{sharding_with_boundaries}'",
    )
    with pytest.raises(ManifestParseError, match="boundaries"):
        parse_sqlite_manifest(payload)


def test_manifest_requires_hash_algorithm() -> None:
    import json

    build = _make_required_build(num_dbs=2)
    shards = _make_shards(2)
    payload = _to_sqlite(build, shards)
    sharding_without_hash = json.dumps({"strategy": "hash"})
    payload = _tamper_sqlite(
        payload,
        f"UPDATE build_meta SET sharding = '{sharding_without_hash}'",
    )
    with pytest.raises(ManifestParseError, match="hash_algorithm"):
        parse_sqlite_manifest(payload)


def test_manifest_rejects_unsupported_hash_algorithm() -> None:
    import json

    build = _make_required_build(num_dbs=2)
    shards = _make_shards(2)
    payload = _to_sqlite(build, shards)
    sharding_unknown_hash = json.dumps(
        {"strategy": "hash", "hash_algorithm": "future_hash"}
    )
    payload = _tamper_sqlite(
        payload,
        f"UPDATE build_meta SET sharding = '{sharding_unknown_hash}'",
    )
    with pytest.raises(ManifestParseError, match="Unsupported shard hash algorithm"):
        parse_sqlite_manifest(payload)


def test_manifest_rejects_unsupported_categorical_token_types() -> None:
    import json

    build = _make_cel_build(num_dbs=2, format_version=4)
    shards = _make_shards(2)
    payload = _to_sqlite(build, shards)
    # Replace routing_values with floats
    sharding_with_floats = json.dumps(
        {
            "strategy": "cel",
            "cel_expr": "region",
            "cel_columns": {"region": "string"},
            "routing_values": [1.5, 2.5],
            "hash_algorithm": "xxh3_64",
        }
    )
    payload = _tamper_sqlite(
        payload,
        f"UPDATE build_meta SET sharding = '{sharding_with_floats}'",
    )
    with pytest.raises(ManifestParseError, match="routing_values"):
        parse_sqlite_manifest(payload)


def test_categorical_manifest_num_dbs_must_match_routing_values() -> None:
    import json

    build = _make_cel_build(num_dbs=3, format_version=4)
    shards = _make_shards(3)
    payload = _to_sqlite(build, shards)
    # Set routing_values to only 2 items while num_dbs=3
    sharding_mismatch = json.dumps(
        {
            "strategy": "cel",
            "cel_expr": "region",
            "cel_columns": {"region": "string"},
            "routing_values": ["ap", "eu"],
            "hash_algorithm": "xxh3_64",
        }
    )
    payload = _tamper_sqlite(
        payload,
        f"UPDATE build_meta SET sharding = '{sharding_mismatch}'",
    )
    with pytest.raises(ManifestParseError, match="routing_values cardinality"):
        parse_sqlite_manifest(payload)
