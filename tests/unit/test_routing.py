from __future__ import annotations

import pytest

from slatedb_spark_sharded.manifest import RequiredBuildMeta, RequiredShardMeta
from slatedb_spark_sharded.routing import SnapshotRouter
from slatedb_spark_sharded.sharding import ShardingSpec, ShardingStrategy


def _build_required(
    *, strategy: ShardingStrategy, num_dbs: int, boundaries=None
) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding="u64be",
        sharding=ShardingSpec(strategy=strategy, boundaries=boundaries),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
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


def test_custom_expr_requires_routing_info() -> None:
    required = _build_required(
        strategy=ShardingStrategy.CUSTOM_EXPR, num_dbs=2, boundaries=None
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
    ]

    router = SnapshotRouter(required, shards)
    with pytest.raises(ValueError, match="not directly routable"):
        router.route_one(1)
