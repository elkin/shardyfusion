from __future__ import annotations

import pytest
from pyspark.sql import functions as F
from slatedb_spark_sharded.errors import ShardAssignmentError
from slatedb_spark_sharded.sharding import (
    DB_ID_COL,
    ShardingSpec,
    ShardingStrategy,
    _range_bucket_expr,
    _range_bucketize_df,
    add_db_id_column,
)


def test_hash_sharding_produces_db_id_in_range(spark) -> None:
    df = spark.createDataFrame([(i,) for i in range(100)], ["id"])
    with_db_id, _ = add_db_id_column(
        df,
        key_col="id",
        num_dbs=8,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
    )

    bad = with_db_id.where((F.col(DB_ID_COL) < 0) | (F.col(DB_ID_COL) >= 8)).count()
    assert bad == 0


def test_hash_sharding_rejects_non_integral_key_type(spark) -> None:
    df = spark.createDataFrame([("1",), ("2",)], ["id"])
    with pytest.raises(ShardAssignmentError, match="Hash sharding requires key column type"):
        add_db_id_column(
            df,
            key_col="id",
            num_dbs=2,
            sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        )


def test_range_sharding_with_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (5,), (10,), (15,), (20,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[10, 20])
    with_db_id, _ = add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)

    got = sorted((row["id"], row[DB_ID_COL]) for row in with_db_id.select("id", DB_ID_COL).collect())
    assert got == [(1, 0), (5, 0), (10, 1), (15, 1), (20, 2)]


def test_range_sharding_accepts_float_key_type(spark) -> None:
    df = spark.createDataFrame([(0.1,), (1.2,), (2.8,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[1.0])
    with_db_id, _ = add_db_id_column(df, key_col="id", num_dbs=2, sharding=spec)

    got = {row["id"]: row[DB_ID_COL] for row in with_db_id.select("id", DB_ID_COL).collect()}
    assert got[0.1] == 0
    assert got[1.2] == 1
    assert got[2.8] == 1


def test_range_sharding_rejects_boolean_key_type(spark) -> None:
    df = spark.createDataFrame([(False,), (True,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[0.5])
    with pytest.raises(ShardAssignmentError, match="Range sharding requires key column type one of"):
        add_db_id_column(df, key_col="id", num_dbs=2, sharding=spec)


def test_sharding_strategy_requires_enum() -> None:
    with pytest.raises(ValueError, match="strategy must be ShardingStrategy"):
        ShardingSpec(strategy="hash")  # type: ignore[arg-type]


def test_sharding_strategy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="strategy must be ShardingStrategy"):
        ShardingSpec(strategy="unknown")  # type: ignore[arg-type]


def test_range_sharding_rejects_unsorted_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[20, 10])
    with pytest.raises(ShardAssignmentError, match="strictly increasing"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_rejects_duplicate_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[10, 10])
    with pytest.raises(ShardAssignmentError, match="strictly increasing"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_rejects_null_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
    spec = ShardingSpec(
        strategy=ShardingStrategy.RANGE, boundaries=[10, None]
    )  # type: ignore[list-item]
    with pytest.raises(ShardAssignmentError, match="must not contain null"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_rejects_duplicate_quantile_boundaries(spark) -> None:
    df = spark.createDataFrame([(5,), (5,), (5,), (5,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE)
    with pytest.raises(ShardAssignmentError, match="strictly increasing"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_rejects_missing_auto_boundaries_for_empty_df(spark) -> None:
    df = spark.createDataFrame([], "id long")
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE)
    with pytest.raises(ShardAssignmentError, match="expected 2, got 0"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_many_boundaries(spark) -> None:
    boundaries = list(range(1, 100))
    df = spark.createDataFrame([(0,), (1,), (50,), (99,), (120,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=boundaries)
    with_db_id, _ = add_db_id_column(df, key_col="id", num_dbs=100, sharding=spec)

    got = {
        row["id"]: row[DB_ID_COL]
        for row in with_db_id.select("id", DB_ID_COL).collect()
    }
    assert got[0] == 0
    assert got[1] == 1
    assert got[50] == 50
    assert got[99] == 99
    assert got[120] == 99


def test_range_bucket_expr_and_bucketizer_are_equivalent(spark) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pyspark.ml.feature")

    boundaries = [10, 20, 35, 50]
    df = spark.createDataFrame(
        [(-5,), (0,), (9,), (10,), (19,), (20,), (35,), (36,), (50,), (100,)],
        ["id"],
    )

    expr_df = df.withColumn(DB_ID_COL, _range_bucket_expr("id", boundaries).cast("int"))
    bucketizer_df = _range_bucketize_df(df, "id", boundaries)

    expr_map = {row["id"]: row[DB_ID_COL] for row in expr_df.collect()}
    bucketizer_map = {row["id"]: row[DB_ID_COL] for row in bucketizer_df.collect()}

    assert expr_map == bucketizer_map


def test_range_sharding_rejects_boolean_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (2,)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[True])
    with pytest.raises(ShardAssignmentError, match="must not be boolean"):
        add_db_id_column(df, key_col="id", num_dbs=2, sharding=spec)


def test_range_sharding_string_boundaries_non_numeric_path(spark) -> None:
    df = spark.createDataFrame([("aa",), ("ba",), ("zz",)], ["id"])
    spec = ShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=["m"])
    with_db_id, _ = add_db_id_column(df, key_col="id", num_dbs=2, sharding=spec)
    got = {row["id"]: row[DB_ID_COL] for row in with_db_id.select("id", DB_ID_COL).collect()}
    assert got["aa"] == 0
    assert got["ba"] == 0
    assert got["zz"] == 1
