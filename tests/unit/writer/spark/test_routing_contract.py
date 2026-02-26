"""Spark-vs-Python cross-checks for the sharding invariant.

These tests run with a local Spark session and compare Spark SQL results
against the Python routing functions, verifying that both compute identical
shard IDs for ~200 edge-case keys.

The Python-only property tests (hypothesis) live in the parent
``tests/unit/writer/test_routing_contract.py``.
"""

from __future__ import annotations

from bisect import bisect_right

import pytest

from slatedb_spark_sharded.routing import xxhash64_db_id
from slatedb_spark_sharded.sharding_types import KeyEncoding
from tests.unit.writer.test_routing_contract import EDGE_CASE_KEYS, U32_EDGE_CASE_KEYS


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
        python_db_id = xxhash64_db_id(key, num_dbs, KeyEncoding.U64BE)
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
        python_db_id = xxhash64_db_id(key, num_dbs, KeyEncoding.U32BE)
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
