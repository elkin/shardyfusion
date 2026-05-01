from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from shardyfusion.errors import ShardAssignmentError
from shardyfusion.sharding_types import (
    KeyEncoding,
    ShardHashAlgorithm,
)
from shardyfusion.writer.spark.sharding import DB_ID_COL
from shardyfusion.writer.spark.writer import (
    _add_db_id_column_hash,
    _verify_hash_routing_agreement,
)


def test_hash_sharding_produces_db_id_in_range(spark) -> None:
    df = spark.createDataFrame([(i,) for i in range(100)], ["id"])
    with_db_id = _add_db_id_column_hash(
        df,
        key_col="id",
        num_dbs=8,
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
    )

    bad = with_db_id.where((F.col(DB_ID_COL) < 0) | (F.col(DB_ID_COL) >= 8)).count()
    assert bad == 0


def test_hash_sharding_accepts_string_key(spark) -> None:
    """HASH sharding now supports all key types (via mapInArrow + xxh3)."""
    df = spark.createDataFrame([("foo",), ("bar",), ("baz",)], ["id"])
    with_db_id = _add_db_id_column_hash(
        df,
        key_col="id",
        num_dbs=4,
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
    )

    bad = with_db_id.where((F.col(DB_ID_COL) < 0) | (F.col(DB_ID_COL) >= 4)).count()
    assert bad == 0


def test_hash_sharding_is_deterministic(spark) -> None:
    """Running _add_db_id_column_hash twice produces the same shard assignments."""
    df = spark.createDataFrame([(i,) for i in range(50)], ["id"])

    first = _add_db_id_column_hash(
        df, key_col="id", num_dbs=8, hash_algorithm=ShardHashAlgorithm.XXH3_64
    )
    second = _add_db_id_column_hash(
        df, key_col="id", num_dbs=8, hash_algorithm=ShardHashAlgorithm.XXH3_64
    )

    first_map = {row["id"]: row[DB_ID_COL] for row in first.collect()}
    second_map = {row["id"]: row[DB_ID_COL] for row in second.collect()}
    assert first_map == second_map


# ---------------------------------------------------------------------------
# Runtime spot-check tests (_verify_hash_routing_agreement)
# ---------------------------------------------------------------------------


def test_verify_hash_routing_agreement_passes_for_hash(spark) -> None:
    """Spot-check should pass for a correctly hash-sharded DataFrame."""
    df = spark.createDataFrame([(i,) for i in range(50)], ["id"])
    with_db_id = _add_db_id_column_hash(
        df,
        key_col="id",
        num_dbs=8,
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
    )
    # Should not raise
    _verify_hash_routing_agreement(
        with_db_id,
        key_col="id",
        num_dbs=8,
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
        key_encoding=KeyEncoding.U64BE,
    )


def test_verify_hash_routing_agreement_catches_wrong_db_ids(spark) -> None:
    """Spot-check should raise when db_ids are deliberately wrong."""

    df = spark.createDataFrame([(i,) for i in range(20)], ["id"])
    # Assign wrong db_ids: invert the correct shard assignment
    correct = _add_db_id_column_hash(
        df,
        key_col="id",
        num_dbs=8,
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
    )
    wrong_df = correct.withColumn(DB_ID_COL, (F.lit(7) - F.col(DB_ID_COL)).cast("int"))
    with pytest.raises(ShardAssignmentError, match="Spark/Python routing mismatch"):
        _verify_hash_routing_agreement(
            wrong_df,
            key_col="id",
            num_dbs=8,
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
            key_encoding=KeyEncoding.U64BE,
        )
