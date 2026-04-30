from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from shardyfusion.errors import ShardAssignmentError
from shardyfusion.sharding_types import HashShardingSpec, KeyEncoding
from shardyfusion.writer.spark.sharding import (
    DB_ID_COL,
    add_db_id_column,
)
from shardyfusion.writer.spark.writer import verify_routing_agreement


def test_hash_sharding_produces_db_id_in_range(spark) -> None:
    df = spark.createDataFrame([(i,) for i in range(100)], ["id"])
    with_db_id, _ = add_db_id_column(
        df,
        key_col="id",
        num_dbs=8,
        sharding=HashShardingSpec(),
    )

    bad = with_db_id.where((F.col(DB_ID_COL) < 0) | (F.col(DB_ID_COL) >= 8)).count()
    assert bad == 0


def test_hash_sharding_accepts_string_key(spark) -> None:
    """HASH sharding now supports all key types (via mapInArrow + xxh3)."""
    df = spark.createDataFrame([("foo",), ("bar",), ("baz",)], ["id"])
    with_db_id, _ = add_db_id_column(
        df,
        key_col="id",
        num_dbs=4,
        sharding=HashShardingSpec(),
    )

    bad = with_db_id.where((F.col(DB_ID_COL) < 0) | (F.col(DB_ID_COL) >= 4)).count()
    assert bad == 0


def test_hash_sharding_is_deterministic(spark) -> None:
    """Running add_db_id_column twice produces the same shard assignments."""
    df = spark.createDataFrame([(i,) for i in range(50)], ["id"])
    spec = HashShardingSpec()

    first, _ = add_db_id_column(df, key_col="id", num_dbs=8, sharding=spec)
    second, _ = add_db_id_column(df, key_col="id", num_dbs=8, sharding=spec)

    first_map = {row["id"]: row[DB_ID_COL] for row in first.collect()}
    second_map = {row["id"]: row[DB_ID_COL] for row in second.collect()}
    assert first_map == second_map


# ---------------------------------------------------------------------------
# Runtime spot-check tests (verify_routing_agreement)
# ---------------------------------------------------------------------------


def test_verify_routing_agreement_passes_for_hash(spark) -> None:
    """Spot-check should pass for a correctly hash-sharded DataFrame."""
    df = spark.createDataFrame([(i,) for i in range(50)], ["id"])
    with_db_id, resolved = add_db_id_column(
        df,
        key_col="id",
        num_dbs=8,
        sharding=HashShardingSpec(),
    )
    # Should not raise
    verify_routing_agreement(
        with_db_id,
        key_col="id",
        num_dbs=8,
        resolved_sharding=resolved,
        key_encoding=KeyEncoding.U64BE,
    )


def test_verify_routing_agreement_catches_wrong_db_ids(spark) -> None:
    """Spot-check should raise when db_ids are deliberately wrong."""

    df = spark.createDataFrame([(i,) for i in range(20)], ["id"])
    # Assign wrong db_ids: invert the correct shard assignment
    correct, _ = add_db_id_column(
        df,
        key_col="id",
        num_dbs=8,
        sharding=HashShardingSpec(),
    )
    wrong_df = correct.withColumn(DB_ID_COL, (F.lit(7) - F.col(DB_ID_COL)).cast("int"))
    sharding = HashShardingSpec()
    with pytest.raises(ShardAssignmentError, match="Spark/Python routing mismatch"):
        verify_routing_agreement(
            wrong_df,
            key_col="id",
            num_dbs=8,
            resolved_sharding=sharding,
            key_encoding=KeyEncoding.U64BE,
        )
