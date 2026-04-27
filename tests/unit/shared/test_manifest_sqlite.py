"""Tests for SQLite manifest serialization and parsing."""

import sqlite3
from datetime import UTC, datetime

import pytest

from shardyfusion.errors import ManifestParseError
from shardyfusion.manifest import (
    SQLITE_MANIFEST_CONTENT_TYPE,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    SqliteManifestBuilder,
    WriterInfo,
)
from shardyfusion.manifest_store import (
    parse_manifest_payload,
    parse_sqlite_manifest,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


def _make_required_build(**overrides):
    defaults = dict(
        run_id="test-run",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_col="_key",
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
        key_encoding=KeyEncoding.U64BE,
    )
    defaults.update(overrides)
    return RequiredBuildMeta(**defaults)


def _make_shard(db_id=0, **overrides):
    defaults = dict(
        db_id=db_id,
        db_url=f"s3://bucket/prefix/shards/db={db_id:05d}",
        attempt=0,
        row_count=100,
        db_bytes=0,
    )
    defaults.update(overrides)
    return RequiredShardMeta(**defaults)


class TestSqliteManifestBuilder:
    def test_build_produces_sqlite(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build()
        shards = [_make_shard(0)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        assert artifact.content_type == SQLITE_MANIFEST_CONTENT_TYPE
        assert artifact.payload[:16] == b"SQLite format 3\x00"

    def test_roundtrip_via_parse_sqlite_manifest(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build()
        shards = [_make_shard(i) for i in range(4)]
        artifact = builder.build(
            required_build=rb, shards=shards, custom_fields={"my_field": "val"}
        )

        parsed = parse_sqlite_manifest(artifact.payload)
        assert parsed.required_build.run_id == "test-run"
        assert parsed.required_build.num_dbs == 4
        assert parsed.required_build.sharding.hash_algorithm == "xxh3_64"
        assert len(parsed.shards) == 4
        assert parsed.custom["my_field"] == "val"

    def test_auto_detect_dispatches_to_sqlite(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build()
        shards = [_make_shard(0)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        parsed = parse_manifest_payload(artifact.payload)
        assert parsed.required_build.run_id == "test-run"
        assert len(parsed.shards) == 1

    def test_cel_fields_in_manifest(self) -> None:
        builder = SqliteManifestBuilder()
        sharding = ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key % 1000",
            cel_columns={"key": "int"},
            routing_values=[250, 500, 750],
            hash_algorithm="xxh3_64",
        )
        rb = _make_required_build(sharding=sharding, format_version=4, num_dbs=3)
        shards = [_make_shard(i) for i in range(3)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        parsed = parse_sqlite_manifest(artifact.payload)
        assert parsed.required_build.sharding.strategy == ShardingStrategy.CEL
        assert parsed.required_build.sharding.cel_expr == "key % 1000"
        assert parsed.required_build.sharding.cel_columns == {"key": "int"}
        assert parsed.required_build.sharding.routing_values == [250, 500, 750]

    def test_custom_fields_merged(self) -> None:
        builder = SqliteManifestBuilder()
        builder.add_custom_field("builder_field", "from_builder")
        rb = _make_required_build()
        artifact = builder.build(
            required_build=rb,
            shards=[_make_shard(0)],
            custom_fields={"call_field": "from_call"},
        )

        parsed = parse_sqlite_manifest(artifact.payload)
        assert parsed.custom["builder_field"] == "from_builder"
        assert parsed.custom["call_field"] == "from_call"

    def test_empty_shards(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build()
        artifact = builder.build(required_build=rb, shards=[], custom_fields={})

        parsed = parse_sqlite_manifest(artifact.payload)
        assert parsed.required_build.num_dbs == 4
        assert len(parsed.shards) == 0

    def test_shard_min_max_key_int(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build(num_dbs=1)
        shards = [_make_shard(0, min_key=10, max_key=99, row_count=5)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        parsed = parse_sqlite_manifest(artifact.payload)
        assert parsed.shards[0].min_key == 10
        assert parsed.shards[0].max_key == 99
        assert isinstance(parsed.shards[0].min_key, int)

    def test_shard_min_max_key_str(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build(num_dbs=1, key_encoding=KeyEncoding.UTF8)
        shards = [_make_shard(0, min_key="aaa", max_key="zzz", row_count=5)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        parsed = parse_sqlite_manifest(artifact.payload)
        assert parsed.shards[0].min_key == "aaa"
        assert parsed.shards[0].max_key == "zzz"
        assert isinstance(parsed.shards[0].min_key, str)

    def test_shard_min_max_key_none(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build(num_dbs=1)
        shards = [_make_shard(0, min_key=None, max_key=None)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        parsed = parse_sqlite_manifest(artifact.payload)
        assert parsed.shards[0].min_key is None
        assert parsed.shards[0].max_key is None

    def test_writer_info_preserved(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build(num_dbs=1)
        wi = WriterInfo(stage_id=5, task_attempt_id=42, attempt=2, duration_ms=1500)
        shards = [_make_shard(0, writer_info=wi)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        parsed = parse_sqlite_manifest(artifact.payload)
        info = parsed.shards[0].writer_info
        assert info.stage_id == 5
        assert info.task_attempt_id == 42
        assert info.attempt == 2
        assert info.duration_ms == 1500

    def test_key_encoding_preserved(self) -> None:
        for encoding in KeyEncoding:
            builder = SqliteManifestBuilder()
            rb = _make_required_build(num_dbs=1, key_encoding=encoding)
            artifact = builder.build(
                required_build=rb, shards=[_make_shard(0)], custom_fields={}
            )
            parsed = parse_sqlite_manifest(artifact.payload)
            assert parsed.required_build.key_encoding == encoding

    def test_sqlite_tables_have_correct_schema(self) -> None:
        builder = SqliteManifestBuilder()
        rb = _make_required_build()
        shards = [_make_shard(0)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        con = sqlite3.connect(":memory:")
        con.deserialize(artifact.payload)

        # Verify tables exist
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert tables == {"build_meta", "shards"}

        # Verify shards has PRIMARY KEY on db_id
        shard_info = con.execute("PRAGMA table_info(shards)").fetchall()
        pk_cols = [row[1] for row in shard_info if row[5] == 1]
        assert pk_cols == ["db_id"]

        con.close()


class TestParseSqliteManifestRejectsInvalid:
    def test_rejects_binary_garbage(self) -> None:
        with pytest.raises(ManifestParseError):
            parse_sqlite_manifest(b"\xff\xfe\x00\x01garbage")

    def test_rejects_empty_database(self) -> None:
        """A valid SQLite DB with no manifest tables should fail."""
        con = sqlite3.connect(":memory:")
        # Python 3.13 can reject serializing a never-initialized in-memory DB.
        # Create and drop a dummy table so the payload is still a valid SQLite
        # database with no manifest tables.
        con.execute("CREATE TABLE _tmp (id INTEGER)")
        con.execute("DROP TABLE _tmp")
        con.commit()
        payload = con.serialize()
        con.close()
        with pytest.raises(ManifestParseError):
            parse_sqlite_manifest(payload)

    def test_rejects_missing_build_meta(self) -> None:
        """SQLite DB with shards table but no build_meta row."""
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE build_meta (run_id TEXT, created_at TEXT,"
            " num_dbs INTEGER, s3_prefix TEXT, key_col TEXT,"
            " sharding TEXT, db_path_template TEXT, shard_prefix TEXT,"
            " format_version INTEGER, key_encoding TEXT, custom TEXT)"
        )
        con.execute(
            "CREATE TABLE shards (db_id INTEGER PRIMARY KEY,"
            " db_url TEXT, attempt INTEGER, row_count INTEGER,"
            " min_key TEXT, max_key TEXT, checkpoint_id TEXT,"
            " writer_info TEXT)"
        )
        con.commit()
        payload = con.serialize()
        con.close()
        with pytest.raises(ManifestParseError, match="no build_meta row"):
            parse_sqlite_manifest(payload)
