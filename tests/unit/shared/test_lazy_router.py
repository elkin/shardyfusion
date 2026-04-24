"""Tests for lazy SnapshotRouter and SqliteShardLookup."""

import sqlite3
from datetime import UTC, datetime

import pytest

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    SqliteManifestBuilder,
    WriterInfo,
)
from shardyfusion.manifest_store import (
    SqliteShardLookup,
    load_sqlite_build_meta,
)
from shardyfusion.routing import SnapshotRouter
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
    )
    defaults.update(overrides)
    return RequiredShardMeta(**defaults)


def _build_sqlite_manifest_con(
    required_build: RequiredBuildMeta,
    shards: list[RequiredShardMeta],
    custom: dict | None = None,
) -> sqlite3.Connection:
    """Build a SQLite manifest and return an open in-memory connection."""
    builder = SqliteManifestBuilder()
    artifact = builder.build(
        required_build=required_build,
        shards=shards,
        custom_fields=custom or {},
    )
    con = sqlite3.connect(":memory:")
    con.deserialize(artifact.payload)
    return con


class TestSnapshotRouterGetShard:
    """Test get_shard() in eager mode (standard __init__)."""

    def test_get_shard_returns_populated(self) -> None:
        rb = _make_required_build()
        shards = [_make_shard(i) for i in range(4)]
        router = SnapshotRouter(rb, shards)

        for i in range(4):
            shard = router.get_shard(i)
            assert shard.db_id == i
            assert shard.db_url is not None
            assert shard.row_count == 100

    def test_get_shard_returns_empty_for_missing(self) -> None:
        rb = _make_required_build()
        shards = [_make_shard(0), _make_shard(2)]
        router = SnapshotRouter(rb, shards)

        # db_id 1 and 3 are empty (not in shard list)
        empty = router.get_shard(1)
        assert empty.db_url is None
        assert empty.row_count == 0
        assert empty.db_id == 1

    def test_is_lazy_false_for_eager(self) -> None:
        rb = _make_required_build(num_dbs=1)
        router = SnapshotRouter(rb, [_make_shard(0)])
        assert router.is_lazy is False


class TestSnapshotRouterLazy:
    """Test from_build_meta() lazy mode with SqliteShardLookup."""

    def test_lazy_get_shard_matches_eager(self) -> None:
        rb = _make_required_build()
        shards = [_make_shard(i, row_count=i * 10) for i in range(4)]

        con = _build_sqlite_manifest_con(rb, shards)
        lookup = SqliteShardLookup(con, rb.num_dbs)
        lazy_router = SnapshotRouter.from_build_meta(rb, shard_lookup=lookup)
        eager_router = SnapshotRouter(rb, shards)

        for i in range(4):
            lazy_shard = lazy_router.get_shard(i)
            eager_shard = eager_router.get_shard(i)
            assert lazy_shard.db_id == eager_shard.db_id
            assert lazy_shard.db_url == eager_shard.db_url
            assert lazy_shard.row_count == eager_shard.row_count

        con.close()

    def test_lazy_routing_matches_eager(self) -> None:
        rb = _make_required_build()
        shards = [_make_shard(i) for i in range(4)]

        con = _build_sqlite_manifest_con(rb, shards)
        lookup = SqliteShardLookup(con, rb.num_dbs)
        lazy_router = SnapshotRouter.from_build_meta(rb, shard_lookup=lookup)
        eager_router = SnapshotRouter(rb, shards)

        for key in [0, 42, 999, 123456]:
            assert lazy_router.route_one(key) == eager_router.route_one(key)

        con.close()

    def test_is_lazy_true(self) -> None:
        rb = _make_required_build(num_dbs=1)
        con = _build_sqlite_manifest_con(rb, [_make_shard(0)])
        lookup = SqliteShardLookup(con, rb.num_dbs)
        router = SnapshotRouter.from_build_meta(rb, shard_lookup=lookup)
        assert router.is_lazy is True
        con.close()

    def test_lazy_shards_list_is_empty(self) -> None:
        rb = _make_required_build(num_dbs=1)
        con = _build_sqlite_manifest_con(rb, [_make_shard(0)])
        lookup = SqliteShardLookup(con, rb.num_dbs)
        router = SnapshotRouter.from_build_meta(rb, shard_lookup=lookup)
        assert router.shards == []
        con.close()

    def test_lazy_empty_shard_returns_synthetic(self) -> None:
        rb = _make_required_build()
        # Only populate db_id 0 and 2
        shards = [_make_shard(0), _make_shard(2)]
        con = _build_sqlite_manifest_con(rb, shards)
        lookup = SqliteShardLookup(con, rb.num_dbs)
        router = SnapshotRouter.from_build_meta(rb, shard_lookup=lookup)

        empty = router.get_shard(1)
        assert empty.db_id == 1
        assert empty.db_url is None
        assert empty.row_count == 0

        populated = router.get_shard(0)
        assert populated.db_url is not None
        assert populated.row_count == 100

        con.close()


class TestSqliteShardLookup:
    def test_cache_hit(self) -> None:
        rb = _make_required_build(num_dbs=1)
        con = _build_sqlite_manifest_con(rb, [_make_shard(0)])
        lookup = SqliteShardLookup(con, rb.num_dbs, cache_size=10)

        # First access populates cache
        s1 = lookup.get_shard(0)
        # Second access hits cache (same object)
        s2 = lookup.get_shard(0)
        assert s1.db_id == s2.db_id == 0
        con.close()

    def test_cache_eviction(self) -> None:
        rb = _make_required_build(num_dbs=10)
        shards = [_make_shard(i) for i in range(10)]
        con = _build_sqlite_manifest_con(rb, shards)
        lookup = SqliteShardLookup(con, rb.num_dbs, cache_size=3)

        # Access 4 shards — first should be evicted
        for i in range(4):
            lookup.get_shard(i)

        # Cache should have 3 entries (1, 2, 3) — 0 was evicted
        assert len(lookup._cache) == 3
        assert 0 not in lookup._cache
        assert 3 in lookup._cache
        con.close()

    def test_cache_disabled(self) -> None:
        rb = _make_required_build(num_dbs=1)
        con = _build_sqlite_manifest_con(rb, [_make_shard(0)])
        lookup = SqliteShardLookup(con, rb.num_dbs, cache_size=0)

        s = lookup.get_shard(0)
        assert s.db_id == 0
        assert len(lookup._cache) == 0
        con.close()

    def test_writer_info_preserved(self) -> None:
        rb = _make_required_build(num_dbs=1)
        wi = WriterInfo(stage_id=5, task_attempt_id=42, attempt=2, duration_ms=1500)
        shards = [_make_shard(0, writer_info=wi)]
        con = _build_sqlite_manifest_con(rb, shards)
        lookup = SqliteShardLookup(con, rb.num_dbs)

        s = lookup.get_shard(0)
        assert s.writer_info.stage_id == 5
        assert s.writer_info.task_attempt_id == 42
        assert s.writer_info.duration_ms == 1500
        con.close()

    def test_min_max_key_types(self) -> None:
        rb = _make_required_build(num_dbs=2)
        shards = [
            _make_shard(0, min_key=10, max_key=99),
            _make_shard(1, min_key="aaa", max_key="zzz"),
        ]
        con = _build_sqlite_manifest_con(rb, shards)
        lookup = SqliteShardLookup(con, rb.num_dbs)

        s0 = lookup.get_shard(0)
        assert s0.min_key == 10
        assert isinstance(s0.min_key, int)

        s1 = lookup.get_shard(1)
        assert s1.min_key == "aaa"
        assert isinstance(s1.min_key, str)
        con.close()


class TestLoadSqliteBuildMeta:
    def test_reads_build_meta(self) -> None:
        rb = _make_required_build()
        shards = [_make_shard(0)]
        con = _build_sqlite_manifest_con(rb, shards, custom={"env": "test"})

        loaded_rb, custom = load_sqlite_build_meta(con)
        assert loaded_rb.run_id == "test-run"
        assert loaded_rb.num_dbs == 4
        assert loaded_rb.sharding.strategy == ShardingStrategy.HASH
        assert custom == {"env": "test"}
        con.close()

    def test_rejects_empty_database(self) -> None:
        from shardyfusion.errors import ManifestParseError

        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE build_meta (run_id TEXT, created_at TEXT,"
            " num_dbs INTEGER, s3_prefix TEXT, key_col TEXT,"
            " sharding TEXT, db_path_template TEXT, shard_prefix TEXT,"
            " format_version INTEGER, key_encoding TEXT, custom TEXT)"
        )
        with pytest.raises(ManifestParseError, match="no build_meta row"):
            load_sqlite_build_meta(con)
        con.close()
