"""Contract tests verifying the writer-reader sharding invariant.

The core invariant: for any key K, num_dbs N, and encoding E,
the Spark writer and the Python reader/writer MUST compute the same
shard ID. A violation means reads silently go to the wrong shard.

Tests are organized in two sections:

1. Python-only property tests (hypothesis) — run without Spark.
2. Spark-vs-Python cross-checks — run with a local Spark session
   and compare Spark SQL results against the Python routing functions.
"""

from __future__ import annotations

import random
from bisect import bisect_right

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slatedb_spark_sharded._writer_core import _route_key
from slatedb_spark_sharded.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from slatedb_spark_sharded.routing import (
    SnapshotRouter,
    _xxhash64_db_id,
    _xxhash64_payload,
    _xxhash64_signed,
)
from slatedb_spark_sharded.sharding_types import (
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
            writer_info={},
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


# ===================================================================
# Section 1: Python-only property tests (no Spark required)
# ===================================================================


@given(key=u64be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_deterministic(key: int, num_dbs: int) -> None:
    """Same key + num_dbs always produces the same db_id."""
    a = _xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    b = _xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    assert a == b


@given(key=u64be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_hash_routing_in_valid_range(key: int, num_dbs: int) -> None:
    """Result is always in [0, num_dbs)."""
    result = _xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    assert 0 <= result < num_dbs


@given(payload=st.binary(min_size=1, max_size=64))
@settings(max_examples=500)
def test_xxhash64_signed_in_int64_range(payload: bytes) -> None:
    """Signed conversion always stays within signed int64 range."""
    result = _xxhash64_signed(payload)
    assert _INT64_MIN <= result <= _INT64_MAX


@given(payload=st.binary(min_size=8, max_size=8), num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_signed_digest_mod_non_negative(payload: bytes, num_dbs: int) -> None:
    """Python % with positive num_dbs always returns non-negative (matches pmod)."""
    digest = _xxhash64_signed(payload)
    assert digest % num_dbs >= 0


@given(key=u32be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_u32be_u64be_hash_equivalence(key: int, num_dbs: int) -> None:
    """Both encodings produce identical routes for keys in [0, 2^32-1]."""
    u32 = _xxhash64_db_id(key, num_dbs, KeyEncoding.U32BE)
    u64 = _xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
    assert u32 == u64, f"key={key}, num_dbs={num_dbs}"


@given(key=u64be_keys)
@settings(max_examples=500)
def test_payload_length_u64be(key: int) -> None:
    """_xxhash64_payload returns exactly 8 bytes for u64be integer keys."""
    assert len(_xxhash64_payload(key, KeyEncoding.U64BE)) == 8


@given(key=u32be_keys)
@settings(max_examples=500)
def test_payload_length_u32be(key: int) -> None:
    """_xxhash64_payload returns exactly 8 bytes for u32be integer keys."""
    assert len(_xxhash64_payload(key, KeyEncoding.U32BE)) == 8


@given(key=u64be_keys)
@settings(max_examples=500)
def test_bytes_key_matches_int_key_u64be(key: int) -> None:
    """Routing via int or big-endian bytes gives the same payload."""
    key_bytes = key.to_bytes(8, byteorder="big")
    assert _xxhash64_payload(key_bytes, KeyEncoding.U64BE) == _xxhash64_payload(
        key, KeyEncoding.U64BE
    )


@given(key=u32be_keys)
@settings(max_examples=500)
def test_bytes_key_matches_int_key_u32be(key: int) -> None:
    """Routing via int or big-endian bytes gives the same payload for u32be."""
    key_bytes = key.to_bytes(4, byteorder="big")
    assert _xxhash64_payload(key_bytes, KeyEncoding.U32BE) == _xxhash64_payload(
        key, KeyEncoding.U32BE
    )


@given(key=u64be_keys, num_dbs=num_dbs_st)
@settings(max_examples=500)
def test_python_writer_reader_routing_identity(key: int, num_dbs: int) -> None:
    """_route_key (writer) and SnapshotRouter.route_one (reader) must agree."""
    writer_result = _route_key(
        key,
        num_dbs=num_dbs,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        key_encoding=KeyEncoding.U64BE,
    )
    router = _build_router(num_dbs=num_dbs)
    reader_result = router.route_one(key)
    assert writer_result == reader_result, f"key={key}, num_dbs={num_dbs}"


# ===================================================================
# Section 2: Spark-vs-Python cross-checks (requires Spark session)
# ===================================================================

# Build a static edge-case key set (~200 keys) covering type boundaries
# and randomly sampled values with a fixed seed for reproducibility.
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


@pytest.mark.spark
@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 7, 8, 16, 64, 128])
def test_spark_python_hash_agreement_u64be(spark, num_dbs: int) -> None:
    """Verify Spark and Python compute identical hash db_id for ~200 keys."""
    from pyspark.sql import functions as F

    df = spark.createDataFrame([(k,) for k in EDGE_CASE_KEYS], ["id"])
    spark_results = {
        row["id"]: row["db_id"]
        for row in df.select(
            "id",
            F.pmod(F.xxhash64(F.col("id").cast("long")), F.lit(num_dbs)).alias("db_id"),
        ).collect()
    }

    for key in EDGE_CASE_KEYS:
        python_db_id = _xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
        assert python_db_id == spark_results[key], (
            f"key={key}, num_dbs={num_dbs}: Python={python_db_id}, "
            f"Spark={spark_results[key]}"
        )


@pytest.mark.spark
@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 8, 16, 64])
def test_spark_python_hash_agreement_u32be(spark, num_dbs: int) -> None:
    """Verify Spark and Python compute identical hash db_id for u32be keys."""
    from pyspark.sql import functions as F

    df = spark.createDataFrame([(k,) for k in U32_EDGE_CASE_KEYS], ["id"])
    spark_results = {
        row["id"]: row["db_id"]
        for row in df.select(
            "id",
            F.pmod(F.xxhash64(F.col("id").cast("long")), F.lit(num_dbs)).alias("db_id"),
        ).collect()
    }

    for key in U32_EDGE_CASE_KEYS:
        python_db_id = _xxhash64_db_id(key, num_dbs, KeyEncoding.U32BE)
        assert python_db_id == spark_results[key], (
            f"key={key}, num_dbs={num_dbs}: Python(u32be)={python_db_id}, "
            f"Spark={spark_results[key]}"
        )


@pytest.mark.spark
@pytest.mark.parametrize(
    "boundaries",
    [
        [10],
        [10, 20],
        [10, 20, 35, 50],
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        [1],
        [(1 << 31) - 1, (1 << 31)],
    ],
)
def test_spark_range_expr_matches_python_bisect(spark, boundaries: list[int]) -> None:
    """Spark _range_bucket_expr output == bisect_right for integer keys."""
    from slatedb_spark_sharded.writer.spark.sharding import (
        DB_ID_COL,
        _range_bucket_expr,
    )

    keys = sorted(
        set(
            [-1, 0, 1, 5, 9, 10, 11, 15, 19, 20, 21, 35, 50, 51, 100, 500, 1000]
            + [(1 << 31) - 1, (1 << 31), (1 << 31) + 1]
            + boundaries
            + [b - 1 for b in boundaries]
            + [b + 1 for b in boundaries]
        )
    )
    df = spark.createDataFrame([(k,) for k in keys], ["id"])
    spark_df = df.withColumn(
        DB_ID_COL, _range_bucket_expr("id", boundaries).cast("int")
    )
    spark_results = {row["id"]: row[DB_ID_COL] for row in spark_df.collect()}

    for key in keys:
        python_result = bisect_right(boundaries, key)
        assert python_result == spark_results[key], (
            f"key={key}, boundaries={boundaries}: "
            f"Python={python_result}, Spark={spark_results[key]}"
        )


@pytest.mark.spark
@pytest.mark.parametrize(
    "boundaries",
    [
        [10.0, 20.0],
        [10, 20, 35, 50],
        [0, 100, 200],
    ],
)
def test_spark_bucketizer_matches_python_bisect(
    spark, boundaries: list[int | float]
) -> None:
    """Spark Bucketizer output == bisect_right for numeric boundaries."""
    from slatedb_spark_sharded.writer.spark.sharding import (
        DB_ID_COL,
        _range_bucketize_df,
    )

    keys = sorted(
        set(
            [-1, 0, 1, 5, 9, 10, 11, 15, 19, 20, 21, 35, 50, 51, 100, 500, 1000]
            + [int(b) for b in boundaries]
            + [int(b) - 1 for b in boundaries]
            + [int(b) + 1 for b in boundaries]
        )
    )
    df = spark.createDataFrame([(k,) for k in keys], ["id"])
    bucketizer_df = _range_bucketize_df(df, "id", boundaries)
    spark_results = {row["id"]: row[DB_ID_COL] for row in bucketizer_df.collect()}

    for key in keys:
        python_result = bisect_right(boundaries, key)
        assert python_result == spark_results[key], (
            f"key={key}, boundaries={boundaries}: "
            f"Python={python_result}, Spark={spark_results[key]}"
        )


@pytest.mark.spark
def test_spark_string_range_expr_matches_python_bisect(spark) -> None:
    """String boundaries: Spark SQL range expression vs Python bisect_right."""
    from slatedb_spark_sharded.writer.spark.sharding import (
        DB_ID_COL,
        _range_bucket_expr,
    )

    boundaries = ["c", "f", "m", "t"]
    keys = ["a", "b", "c", "d", "e", "f", "g", "m", "n", "t", "u", "z", "zz"]

    df = spark.createDataFrame([(k,) for k in keys], ["id"])
    spark_df = df.withColumn(
        DB_ID_COL, _range_bucket_expr("id", boundaries).cast("int")
    )
    spark_results = {row["id"]: row[DB_ID_COL] for row in spark_df.collect()}

    for key in keys:
        python_result = bisect_right(boundaries, key)
        assert python_result == spark_results[key], (
            f"key={key!r}, boundaries={boundaries}: "
            f"Python={python_result}, Spark={spark_results[key]}"
        )
