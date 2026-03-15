"""Contract tests verifying the writer-reader sharding invariant.

The core invariant: for any key K, num_dbs N, and encoding E,
the Spark writer and the Python reader/writer MUST compute the same
shard ID. A violation means reads silently go to the wrong shard.

This module contains Python-only property tests (hypothesis).
Spark-vs-Python cross-checks live in ``tests/unit/writer/spark/test_routing_contract.py``.

The ``EDGE_CASE_KEYS`` and ``U32_EDGE_CASE_KEYS`` constants are shared
with the Spark and Dask routing contract tests.
"""

from __future__ import annotations

import random

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
    xxhash64_db_id,
    xxhash64_payload,
    xxhash64_signed,
)
from shardyfusion.sharding_types import (
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)

# ---------------------------------------------------------------------------
# Shared strategies and constants
# ---------------------------------------------------------------------------

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1

u64be_keys = st.integers(min_value=0, max_value=_UINT64_MAX)
u32be_keys = st.integers(min_value=0, max_value=_UINT32_MAX)
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
    boundaries: list[int | float | str] | None = None,
) -> SnapshotRouter:
    required = RequiredBuildMeta(
        run_id="contract-test",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=encoding,
        sharding=ManifestShardingSpec(strategy=strategy, boundaries=boundaries),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
    )
    return SnapshotRouter(required, _make_shards(num_dbs))


@given(key=u64be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_deterministic(key: int, num_dbs: int) -> None:
    """Same key + num_dbs always produces the same db_id."""
    a = xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    b = xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    assert a == b


@given(key=u64be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_in_valid_range(key: int, num_dbs: int) -> None:
    """Result is always in [0, num_dbs)."""
    result = xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    assert 0 <= result < num_dbs


@given(payload=st.binary(min_size=1, max_size=64))
@settings(max_examples=500)
def testxxhash64_signed_in_int64_range(payload: bytes) -> None:
    """Signed conversion always stays within signed int64 range."""
    result = xxhash64_signed(payload)
    assert _INT64_MIN <= result <= _INT64_MAX


@given(payload=st.binary(min_size=8, max_size=8), num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_signed_digest_mod_non_negative(payload: bytes, num_dbs: int) -> None:
    """Python % with positive num_dbs always returns non-negative (matches pmod)."""
    digest = xxhash64_signed(payload)
    assert digest % num_dbs >= 0


@given(key=u32be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_u32be_u64be_hash_equivalence(key: int, num_dbs: int) -> None:
    """Both encodings produce identical routes for keys in [0, 2^32-1]."""
    u32 = xxhash64_db_id(key, num_dbs, KeyEncoding.U32BE)
    u64 = xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    assert u32 == u64, f"key={key}, num_dbs={num_dbs}"


@given(key=u64be_keys)
@settings(max_examples=500)
def test_payload_length_u64be(key: int) -> None:
    """xxhash64_payload returns exactly 8 bytes for u64be integer keys."""
    assert len(xxhash64_payload(key, KeyEncoding.U64BE)) == 8


@given(key=u32be_keys)
@settings(max_examples=500)
def test_payload_length_u32be(key: int) -> None:
    """xxhash64_payload returns exactly 8 bytes for u32be integer keys."""
    assert len(xxhash64_payload(key, KeyEncoding.U32BE)) == 8


@given(key=u64be_keys)
@settings(max_examples=500)
def test_bytes_key_matches_int_key_u64be(key: int) -> None:
    """Routing via int or big-endian bytes gives the same payload."""
    key_bytes = key.to_bytes(8, byteorder="big")
    assert xxhash64_payload(key_bytes, KeyEncoding.U64BE) == xxhash64_payload(
        key, KeyEncoding.U64BE
    )


@given(key=u32be_keys)
@settings(max_examples=500)
def test_bytes_key_matches_int_key_u32be(key: int) -> None:
    """Routing via int or big-endian bytes gives the same payload for u32be."""
    key_bytes = key.to_bytes(4, byteorder="big")
    assert xxhash64_payload(key_bytes, KeyEncoding.U32BE) == xxhash64_payload(
        key, KeyEncoding.U32BE
    )


@given(key=u64be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_python_writer_reader_routing_identity(key: int, num_dbs: int) -> None:
    """route_key (writer) and SnapshotRouter.route_one (reader) must agree."""
    writer_result = route_key(
        key,
        num_dbs=num_dbs,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        key_encoding=KeyEncoding.U64BE,
    )
    router = _build_router(num_dbs=num_dbs)
    reader_result = router.route_one(key)
    assert writer_result == reader_result, f"key={key}, num_dbs={num_dbs}"


# ---------------------------------------------------------------------------
# Shared edge-case key sets (used by spark and dask routing contract tests)
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
