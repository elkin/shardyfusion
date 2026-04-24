"""Contract tests verifying the writer-reader sharding invariant.

The core invariant: for any key K and num_dbs N,
the writer and the Python reader MUST compute the same
shard ID. A violation means reads silently go to the wrong shard.

This module contains Python-only property tests (hypothesis).
Spark-vs-Python cross-checks live in
``tests/unit/writer/spark/test_routing_contract.py``.

The ``EDGE_CASE_KEYS`` and ``U32_EDGE_CASE_KEYS`` constants are shared
with the Spark and Dask routing contract tests.
"""

from __future__ import annotations

import random
import struct

from hypothesis import given, settings
from hypothesis import strategies as st

from shardyfusion._writer_core import route_key
from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from shardyfusion.routing import (
    SnapshotRouter,
    canonical_bytes,
    hash_db_id,
    hash_digest,
    xxh3_db_id,
    xxh3_digest,
)
from shardyfusion.sharding_types import (
    KeyEncoding,
    ShardHashAlgorithm,
    ShardingSpec,
    ShardingStrategy,
)

# ---------------------------------------------------------------------------
# Shared strategies and constants
# ---------------------------------------------------------------------------

_INT64_SIGNED_MIN = -(1 << 63)
_INT64_SIGNED_MAX = (1 << 63) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1

int_keys = st.integers(min_value=_INT64_SIGNED_MIN, max_value=_INT64_SIGNED_MAX)
u64be_keys = st.integers(min_value=0, max_value=_UINT64_MAX)
u32be_keys = st.integers(min_value=0, max_value=_UINT32_MAX)
str_keys = st.text(min_size=0, max_size=128)
bytes_keys = st.binary(min_size=0, max_size=128)
num_dbs_st = st.integers(min_value=1, max_value=1024)


def _make_shards(num_dbs: int) -> list[RequiredShardMeta]:
    return [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/db={i:05d}",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info=WriterInfo(),
        )
        for i in range(num_dbs)
    ]


def _build_router(
    num_dbs: int,
    strategy: ShardingStrategy = ShardingStrategy.HASH,
    encoding: KeyEncoding = KeyEncoding.U64BE,
) -> SnapshotRouter:
    required = RequiredBuildMeta(
        run_id="contract-test",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=encoding,
        sharding=ManifestShardingSpec(
            strategy=strategy,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    return SnapshotRouter(required, _make_shards(num_dbs))


# ---------------------------------------------------------------------------
# canonical_bytes tests
# ---------------------------------------------------------------------------


@given(key=int_keys)
@settings(max_examples=500)
def test_canonical_bytes_int_is_8_bytes_signed_le(key: int) -> None:
    """Integer keys produce exactly 8 bytes in signed little-endian format."""
    result = canonical_bytes(key)
    assert len(result) == 8
    # Round-trip: unpack as signed LE int64
    assert struct.unpack("<q", result)[0] == key


@given(key=str_keys)
@settings(max_examples=500)
def test_canonical_bytes_str_is_utf8(key: str) -> None:
    """String keys produce their UTF-8 encoding."""
    result = canonical_bytes(key)
    assert result == key.encode("utf-8")


@given(key=bytes_keys)
@settings(max_examples=500)
def test_canonical_bytes_bytes_is_passthrough(key: bytes) -> None:
    """Bytes keys are returned as-is."""
    assert canonical_bytes(key) is key


def test_canonical_bytes_int_boundary_values() -> None:
    """Boundary values for signed int64 are accepted."""
    assert len(canonical_bytes(_INT64_SIGNED_MIN)) == 8
    assert len(canonical_bytes(_INT64_SIGNED_MAX)) == 8
    assert len(canonical_bytes(0)) == 8


def test_canonical_bytes_int_out_of_range() -> None:
    """Integers outside signed int64 range raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="out of range"):
        canonical_bytes(_INT64_SIGNED_MAX + 1)
    with pytest.raises(ValueError, match="out of range"):
        canonical_bytes(_INT64_SIGNED_MIN - 1)


# ---------------------------------------------------------------------------
# xxh3_db_id tests — integer keys
# ---------------------------------------------------------------------------


@given(key=int_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_deterministic_int(key: int, num_dbs: int) -> None:
    """Same integer key + num_dbs always produces the same db_id."""
    a = xxh3_db_id(key, num_dbs)
    b = xxh3_db_id(key, num_dbs)
    assert a == b


def test_hash_dispatch_matches_xxh3_helpers() -> None:
    keys = [0, 1, -1, 42, "hello", b"hello"]
    for key in keys:
        assert hash_digest(key, ShardHashAlgorithm.XXH3_64) == xxh3_digest(key)
        assert hash_db_id(key, 17, ShardHashAlgorithm.XXH3_64) == xxh3_db_id(key, 17)


def test_sharding_spec_accepts_hash_algorithm_string() -> None:
    spec = ShardingSpec(hash_algorithm="xxh3_64")  # type: ignore[arg-type]
    assert spec.hash_algorithm == ShardHashAlgorithm.XXH3_64
    assert spec.to_manifest_dict()["hash_algorithm"] == "xxh3_64"


@given(key=int_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_in_valid_range_int(key: int, num_dbs: int) -> None:
    """Result is always in [0, num_dbs) for integer keys."""
    result = xxh3_db_id(key, num_dbs)
    assert 0 <= result < num_dbs


# ---------------------------------------------------------------------------
# xxh3_db_id tests — string keys
# ---------------------------------------------------------------------------


@given(key=str_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_deterministic_str(key: str, num_dbs: int) -> None:
    """Same string key + num_dbs always produces the same db_id."""
    a = xxh3_db_id(key, num_dbs)
    b = xxh3_db_id(key, num_dbs)
    assert a == b


@given(key=str_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_in_valid_range_str(key: str, num_dbs: int) -> None:
    """Result is always in [0, num_dbs) for string keys."""
    result = xxh3_db_id(key, num_dbs)
    assert 0 <= result < num_dbs


# ---------------------------------------------------------------------------
# xxh3_db_id tests — bytes keys
# ---------------------------------------------------------------------------


@given(key=bytes_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_deterministic_bytes(key: bytes, num_dbs: int) -> None:
    """Same bytes key + num_dbs always produces the same db_id."""
    a = xxh3_db_id(key, num_dbs)
    b = xxh3_db_id(key, num_dbs)
    assert a == b


@given(key=bytes_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_in_valid_range_bytes(key: bytes, num_dbs: int) -> None:
    """Result is always in [0, num_dbs) for bytes keys."""
    result = xxh3_db_id(key, num_dbs)
    assert 0 <= result < num_dbs


# ---------------------------------------------------------------------------
# xxh3_db_id — encoding independence
# ---------------------------------------------------------------------------


@given(key=int_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_encoding_independent(key: int, num_dbs: int) -> None:
    """xxh3_db_id does not take an encoding parameter — same key always maps to
    the same shard regardless of how the key will be stored."""
    result_a = xxh3_db_id(key, num_dbs)
    result_b = xxh3_db_id(key, num_dbs)
    assert result_a == result_b


# ---------------------------------------------------------------------------
# xxh3_db_id — single-shard collapse
# ---------------------------------------------------------------------------


def test_single_shard_always_zero() -> None:
    """With num_dbs=1, every key routes to shard 0."""
    assert xxh3_db_id(0, 1) == 0
    assert xxh3_db_id(42, 1) == 0
    assert xxh3_db_id(-1, 1) == 0
    assert xxh3_db_id("hello", 1) == 0
    assert xxh3_db_id(b"\x00\xff", 1) == 0


# ---------------------------------------------------------------------------
# Writer-reader identity — integer keys
# ---------------------------------------------------------------------------


@given(key=int_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_writer_reader_routing_identity_int(key: int, num_dbs: int) -> None:
    """route_key (writer) and SnapshotRouter.route_one (reader) must agree for int keys."""
    writer_result = route_key(
        key,
        num_dbs=num_dbs,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
    )
    router = _build_router(num_dbs=num_dbs)
    reader_result = router.route_one(key)
    assert writer_result == reader_result, f"key={key}, num_dbs={num_dbs}"


# ---------------------------------------------------------------------------
# Writer-reader identity — string keys
# ---------------------------------------------------------------------------


@given(key=str_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_writer_reader_routing_identity_str(key: str, num_dbs: int) -> None:
    """route_key (writer) and SnapshotRouter.route_one (reader) must agree for str keys."""
    writer_result = route_key(
        key,
        num_dbs=num_dbs,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
    )
    router = _build_router(num_dbs=num_dbs, encoding=KeyEncoding.UTF8)
    reader_result = router.route_one(key)
    assert writer_result == reader_result, f"key={key!r}, num_dbs={num_dbs}"


# ---------------------------------------------------------------------------
# Writer-reader identity — bytes keys
# ---------------------------------------------------------------------------


@given(key=bytes_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_writer_reader_routing_identity_bytes(key: bytes, num_dbs: int) -> None:
    """route_key (writer) and SnapshotRouter.route_one (reader) must agree for bytes keys."""
    writer_result = route_key(
        key,
        num_dbs=num_dbs,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
    )
    router = _build_router(num_dbs=num_dbs, encoding=KeyEncoding.RAW)
    reader_result = router.route_one(key)
    assert writer_result == reader_result, f"key={key!r}, num_dbs={num_dbs}"


# ---------------------------------------------------------------------------
# Shared edge-case key sets (used by spark, dask, and ray routing contract tests)
# ---------------------------------------------------------------------------

_rng = random.Random(12345)
EDGE_CASE_KEYS: list[int] = sorted(
    set(
        list(range(0, 21))
        + [42, 100, 255, 256, 1023, 1024]
        + [65535, 65536, 65537]
        + [(1 << 31) - 2, (1 << 31) - 1, (1 << 31), (1 << 31) + 1]
        + [(1 << 32) - 2, (1 << 32) - 1, (1 << 32), (1 << 32) + 1]
        + [(1 << 63) - 2, (1 << 63) - 1]
        + [_rng.randint(0, (1 << 63) - 1) for _ in range(100)]
    )
)

# Keys valid for u32be (subset of EDGE_CASE_KEYS)
U32_EDGE_CASE_KEYS: list[int] = [k for k in EDGE_CASE_KEYS if k <= _UINT32_MAX]
