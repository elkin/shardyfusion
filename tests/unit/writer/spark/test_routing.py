from __future__ import annotations

import pytest

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.routing import SnapshotRouter, xxh3_db_id
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


def _build_required(*, strategy: ShardingStrategy, num_dbs: int) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(
            strategy=strategy,
            hash_algorithm="xxh3_64",
        ),
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


def test_hash_router_matches_xxh3_db_id_for_integers() -> None:
    """Router and standalone xxh3_db_id must agree for integer keys."""
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

    for key in keys:
        assert router.route_one(key) == xxh3_db_id(key, num_dbs), f"key={key}"


def test_hash_router_supports_string_and_bytes_keys() -> None:
    """Router handles str and bytes keys via canonical_bytes."""
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

    # String key
    db_id_str = router.route_one("hello")
    assert 0 <= db_id_str < 8
    assert router.route_one("hello") == db_id_str  # deterministic

    # Bytes key
    db_id_bytes = router.route_one(b"hello")
    assert 0 <= db_id_bytes < 8

    # String and its UTF-8 bytes produce the same hash
    assert router.route_one("hello") == router.route_one(b"hello")


# ---------------------------------------------------------------------------
# u32be encoding tests
# ---------------------------------------------------------------------------


def _build_required_u32be(
    *, strategy: ShardingStrategy, num_dbs: int
) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U32BE,
        sharding=ManifestShardingSpec(
            strategy=strategy,
            hash_algorithm="xxh3_64",
        ),
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


def test_hash_routing_is_encoding_independent() -> None:
    """Hash routing is the same regardless of key_encoding (u32be vs u64be)
    because routing uses canonical_bytes(), not key encoding."""
    num_dbs = 8
    required_u64 = _build_required(strategy=ShardingStrategy.HASH, num_dbs=num_dbs)
    required_u32 = _build_required_u32be(
        strategy=ShardingStrategy.HASH, num_dbs=num_dbs
    )
    shards = _make_shards(num_dbs)

    router_u64 = SnapshotRouter(required_u64, shards)
    router_u32 = SnapshotRouter(required_u32, shards)

    keys = [0, 1, 2, 7, 42, 1024, 65_537]
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


def test_u32be_hash_router_int_key_deterministic() -> None:
    """u32be router routes integer keys deterministically."""
    required = _build_required_u32be(strategy=ShardingStrategy.HASH, num_dbs=8)
    shards = _make_shards(8)
    router = SnapshotRouter(required, shards)

    key = 123456
    assert router.route_one(key) == router.route_one(key)
