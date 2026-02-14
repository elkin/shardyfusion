from __future__ import annotations

import pytest
from pyspark.sql import functions as F
from slatedb_spark_sharded.errors import ShardAssignmentError
from slatedb_spark_sharded.sharding import (
    DB_ID_COL,
    ShardingSpec,
    ShardingStrategy,
    add_db_id_column,
)


def test_hash_sharding_produces_db_id_in_range(spark) -> None:
    df = spark.createDataFrame([(i,) for i in range(100)], ["id"])
    with_db_id, _ = add_db_id_column(
        df,
        key_col="id",
        num_dbs=8,
        sharding=ShardingSpec(strategy="hash"),
    )

    bad = with_db_id.where((F.col(DB_ID_COL) < 0) | (F.col(DB_ID_COL) >= 8)).count()
    assert bad == 0


def test_range_sharding_with_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (5,), (10,), (15,), (20,)], ["id"])
    spec = ShardingSpec(strategy="range", boundaries=[10, 20])
    with_db_id, _ = add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)

    got = sorted((row["id"], row[DB_ID_COL]) for row in with_db_id.select("id", DB_ID_COL).collect())
    assert got == [(1, 0), (5, 0), (10, 1), (15, 1), (20, 2)]


def test_sharding_strategy_enum_coercion() -> None:
    spec = ShardingSpec(strategy="hash")
    assert spec.strategy == ShardingStrategy.HASH


def test_sharding_strategy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported sharding strategy"):
        ShardingSpec(strategy="unknown")


def test_range_sharding_rejects_unsorted_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
    spec = ShardingSpec(strategy="range", boundaries=[20, 10])
    with pytest.raises(ShardAssignmentError, match="strictly increasing"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_rejects_duplicate_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
    spec = ShardingSpec(strategy="range", boundaries=[10, 10])
    with pytest.raises(ShardAssignmentError, match="strictly increasing"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_rejects_null_boundaries(spark) -> None:
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
    spec = ShardingSpec(strategy="range", boundaries=[10, None])  # type: ignore[list-item]
    with pytest.raises(ShardAssignmentError, match="must not contain null"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)


def test_range_sharding_rejects_duplicate_quantile_boundaries(spark) -> None:
    df = spark.createDataFrame([(5,), (5,), (5,), (5,)], ["id"])
    spec = ShardingSpec(strategy="range")
    with pytest.raises(ShardAssignmentError, match="strictly increasing"):
        add_db_id_column(df, key_col="id", num_dbs=3, sharding=spec)
