from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


def _build_required(
    *, strategy: ShardingStrategy, num_dbs: int, boundaries=None
) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=strategy, boundaries=boundaries),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )


def test_hash_router_is_deterministic_and_in_range() -> None:
    required = _build_required(strategy=ShardingStrategy.HASH, num_dbs=8)
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/db={i:05d}",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        )
        for i in range(8)
    ]

    router = SnapshotRouter(required, shards)
    first = router.route_one(123)
    second = router.route_one(123)

    assert first == second
    assert 0 <= first < 8


def test_hash_router_matches_spark_xxhash64_for_integers(spark) -> None:
    num_dbs = 8
    required = _build_required(strategy=ShardingStrategy.HASH, num_dbs=num_dbs)
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/db={i:05d}",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        )
        for i in range(num_dbs)
    ]
    keys = [0, 1, 2, 7, 11, 42, 1024, 65_537]

    router = SnapshotRouter(required, shards)
    expected = {
        row["id"]: row["db_id"]
        for row in (
            spark.createDataFrame([(key,) for key in keys], ["id"])
            .select(
                "id",
                F.pmod(F.xxhash64(F.col("id").cast("long")), F.lit(num_dbs)).alias(
                    "db_id"
                ),
            )
            .collect()
        )
    }

    for key in keys:
        assert router.route_one(key) == expected[key]


def test_hash_router_treats_u64be_bytes_same_as_integer_key() -> None:
    required = _build_required(strategy=ShardingStrategy.HASH, num_dbs=8)
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/db={i:05d}",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        )
        for i in range(8)
    ]

    router = SnapshotRouter(required, shards)
    key = 123456789

    assert router.route_one(key) == router.route_one(key.to_bytes(8, byteorder="big"))


def test_range_router_with_boundaries() -> None:
    required = _build_required(
        strategy=ShardingStrategy.RANGE, num_dbs=3, boundaries=[10, 20]
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url="s3://bucket/prefix/db=00000",
            attempt=0,
            row_count=0,
            min_key=0,
            max_key=9,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=1,
            db_url="s3://bucket/prefix/db=00001",
            attempt=0,
            row_count=0,
            min_key=10,
            max_key=19,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=2,
            db_url="s3://bucket/prefix/db=00002",
            attempt=0,
            row_count=0,
            min_key=20,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        ),
    ]

    router = SnapshotRouter(required, shards)

    assert router.route_one(1) == 0
    assert router.route_one(10) == 1
    assert router.route_one(19) == 1
    assert router.route_one(20) == 2


def test_range_router_boundary_values_use_upper_shard_when_using_boundaries() -> None:
    required = _build_required(
        strategy=ShardingStrategy.RANGE, num_dbs=3, boundaries=[10, 20]
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url="s3://bucket/prefix/db=00000",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=1,
            db_url="s3://bucket/prefix/db=00001",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=2,
            db_url="s3://bucket/prefix/db=00002",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        ),
    ]

    router = SnapshotRouter(required, shards)
    assert router.route_one(9) == 0
    assert router.route_one(10) == 1
    assert router.route_one(20) == 2


# ---------------------------------------------------------------------------
# u32be encoding tests
# ---------------------------------------------------------------------------


def _build_required_u32be(
    *, strategy: ShardingStrategy, num_dbs: int, boundaries=None
) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U32BE,
        sharding=ManifestShardingSpec(strategy=strategy, boundaries=boundaries),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )


def _make_shards(num_dbs: int) -> list[RequiredShardMeta]:
    return [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/db={i:05d}",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        )
        for i in range(num_dbs)
    ]


def test_u32be_hash_routes_match_u64be_for_small_keys() -> None:
    """u32be and u64be should produce the same hash routes for keys in [0, 2^32-1]
    because both zero-extend to 8-byte little-endian before hashing."""
    num_dbs = 8
    required_u64 = _build_required(strategy=ShardingStrategy.HASH, num_dbs=num_dbs)
    required_u32 = _build_required_u32be(
        strategy=ShardingStrategy.HASH, num_dbs=num_dbs
    )
    shards = _make_shards(num_dbs)

    router_u64 = SnapshotRouter(required_u64, shards)
    router_u32 = SnapshotRouter(required_u32, shards)

    keys = [0, 1, 2, 7, 42, 1024, 65_537, (1 << 32) - 1]
    for key in keys:
        assert router_u32.route_one(key) == router_u64.route_one(key), f"key={key}"


def test_u32be_encode_lookup_key_returns_4_bytes() -> None:
    required = _build_required_u32be(strategy=ShardingStrategy.HASH, num_dbs=4)
    shards = _make_shards(4)
    router = SnapshotRouter(required, shards)

    encoded = router.encode_lookup_key(256)
    assert len(encoded) == 4
    assert encoded == b"\x00\x00\x01\x00"


def test_u32be_encode_lookup_key_bytes_passthrough() -> None:
    required = _build_required_u32be(strategy=ShardingStrategy.HASH, num_dbs=4)
    shards = _make_shards(4)
    router = SnapshotRouter(required, shards)

    raw = b"\x00\x00\x00\x2a"
    assert router.encode_lookup_key(raw) == raw


def test_u32be_encode_lookup_key_wrong_length_raises() -> None:
    required = _build_required_u32be(strategy=ShardingStrategy.HASH, num_dbs=4)
    shards = _make_shards(4)
    router = SnapshotRouter(required, shards)

    with pytest.raises(ValueError, match="u32be key bytes must have length 4"):
        router.encode_lookup_key(b"\x00\x00\x00")


def test_u32be_hash_router_bytes_key_same_as_int() -> None:
    required = _build_required_u32be(strategy=ShardingStrategy.HASH, num_dbs=8)
    shards = _make_shards(8)
    router = SnapshotRouter(required, shards)

    key = 123456
    assert router.route_one(key) == router.route_one(key.to_bytes(4, byteorder="big"))


# ---------------------------------------------------------------------------
# Range routing with empty shards
# ---------------------------------------------------------------------------


def test_range_routing_with_mixed_empty_shards_prefers_boundaries() -> None:
    """Boundaries drive routing even when a shard has no min/max keys (empty)."""
    required = _build_required(
        strategy=ShardingStrategy.RANGE, num_dbs=3, boundaries=[10, 20]
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url="s3://bucket/prefix/db=00000",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=1,
            db_url="s3://bucket/prefix/db=00001",
            attempt=0,
            row_count=10,
            min_key=10,
            max_key=19,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=2,
            db_url="s3://bucket/prefix/db=00002",
            attempt=0,
            row_count=5,
            min_key=20,
            max_key=29,
            checkpoint_id=None,
            writer_info={},
        ),
    ]

    router = SnapshotRouter(required, shards)

    assert router.route_one(5) == 0
    assert router.route_one(15) == 1
    assert router.route_one(25) == 2


def test_range_routing_empty_shards_excluded_from_intervals() -> None:
    """When no boundaries are provided, empty shards (None, None) are excluded
    from the interval list used for routing."""
    required = _build_required(
        strategy=ShardingStrategy.RANGE, num_dbs=3, boundaries=None
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url="s3://bucket/prefix/db=00000",
            attempt=0,
            row_count=5,
            min_key=0,
            max_key=9,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=1,
            db_url="s3://bucket/prefix/db=00001",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=2,
            db_url="s3://bucket/prefix/db=00002",
            attempt=0,
            row_count=5,
            min_key=20,
            max_key=29,
            checkpoint_id=None,
            writer_info={},
        ),
    ]

    router = SnapshotRouter(required, shards)

    # Only 2 intervals (shard 1 excluded because it's empty)
    assert len(router._range_intervals) == 2
    interval_db_ids = [iv.db_id for iv in router._range_intervals]
    assert 1 not in interval_db_ids
