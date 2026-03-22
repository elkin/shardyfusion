"""Spark-vs-Python cross-checks for the sharding invariant.

Since the sharding rework, ALL writers (including Spark) use Python-based
``mapInArrow`` for hashing via ``xxh3_db_id()``.  There is no longer a
Spark SQL xxhash64 path to cross-check, but these tests verify that the
full ``add_db_id_column()`` pipeline (Arrow serialisation round-trip,
partition assignment, etc.) produces db_ids identical to calling
``xxh3_db_id()`` directly.

The Python-only property tests (hypothesis) live in
``tests/unit/writer/core/test_routing_contract.py``.
"""

from __future__ import annotations

import pytest

from shardyfusion.routing import xxh3_db_id
from shardyfusion.sharding_types import (
    DB_ID_COL,
    ShardingSpec,
    ShardingStrategy,
)
from shardyfusion.writer.spark.sharding import add_db_id_column
from tests.unit.writer.core.test_routing_contract import EDGE_CASE_KEYS

# String keys for cross-checking add_db_id_column with string columns.
_STRING_EDGE_CASE_KEYS: list[str] = [
    "",
    "a",
    "z",
    "hello",
    "Hello",
    "HELLO",
    "key with spaces",
    "key\twith\ttabs",
    "\x00null_byte",
    "emoji\U0001f600face",
    "\u00e9\u00e8\u00ea",
    "a" * 256,
    "0",
    "42",
    "9999999999",
    "/slashes/in/path",
    "key=value&foo=bar",
]


@pytest.mark.spark
@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 7, 8, 16, 64, 128])
def test_spark_hash_agreement_int_keys(spark, num_dbs: int) -> None:
    """Spark add_db_id_column matches xxh3_db_id for integer edge-case keys."""

    df = spark.createDataFrame([(k,) for k in EDGE_CASE_KEYS], ["id"])
    sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    result_df, _ = add_db_id_column(
        df,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
    )

    for row in result_df.collect():
        key = row["id"]
        spark_db_id = row[DB_ID_COL]
        python_db_id = xxh3_db_id(key, num_dbs)
        assert spark_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Spark={spark_db_id}, Python={python_db_id}"
        )


@pytest.mark.spark
@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 7, 8, 16, 64, 128])
def test_spark_hash_agreement_string_keys(spark, num_dbs: int) -> None:
    """Spark add_db_id_column matches xxh3_db_id for string edge-case keys."""

    df = spark.createDataFrame([(k,) for k in _STRING_EDGE_CASE_KEYS], ["id"])
    sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    result_df, _ = add_db_id_column(
        df,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
    )

    for row in result_df.collect():
        key = row["id"]
        spark_db_id = row[DB_ID_COL]
        python_db_id = xxh3_db_id(key, num_dbs)
        assert spark_db_id == python_db_id, (
            f"key={key!r}, num_dbs={num_dbs}: Spark={spark_db_id}, Python={python_db_id}"
        )
