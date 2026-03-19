from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.smoke_scenarios import (
    run_smoke_cel_scenario,
    run_smoke_write_then_read_scenario,
)


def _s3(svc):
    return {
        "credential_provider": credential_provider_from_service(svc),
        "s3_connection_options": s3_connection_options_from_service(svc),
    }


def _make_write_fn(spark):
    """Return a write callback that creates a Spark DataFrame from SMOKE_DATA."""

    def write_fn(data, config):
        from shardyfusion.serde import ValueSpec
        from shardyfusion.writer.spark import write_sharded

        df = spark.createDataFrame(data, ["key", "value", "group"])
        return write_sharded(
            df, config, key_col="key", value_spec=ValueSpec.binary_col("value")
        )

    return write_fn


# ---------------------------------------------------------------------------
# HASH sharding
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.spark
def test_smoke_hash(spark, garage_s3_service, tmp_path) -> None:
    run_smoke_write_then_read_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        num_dbs=3,
        expected_num_shards=3,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.spark
def test_smoke_hash_num_dbs_2(spark, garage_s3_service, tmp_path) -> None:
    run_smoke_write_then_read_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        num_dbs=2,
        expected_num_shards=2,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.spark
def test_smoke_hash_max_keys_per_shard(spark, garage_s3_service, tmp_path) -> None:
    run_smoke_write_then_read_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        num_dbs=0,
        max_keys_per_shard=5,
        expected_num_shards=2,
        **_s3(garage_s3_service),
    )


# ---------------------------------------------------------------------------
# CEL sharding
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.spark
@pytest.mark.cel
def test_smoke_cel_key_modulo(spark, garage_s3_service, tmp_path) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        cel_expr="key % 3",
        cel_columns={"key": "int"},
        boundaries=[1, 2],
        expected_num_shards=3,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.spark
@pytest.mark.cel
def test_smoke_cel_shard_hash(spark, garage_s3_service, tmp_path) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        cel_expr="shard_hash(key) % 3u",
        cel_columns={"key": "int"},
        expected_num_shards=3,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.spark
@pytest.mark.cel
def test_smoke_cel_key_identity(spark, garage_s3_service, tmp_path) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        cel_expr="uint(key)",
        cel_columns={"key": "int"},
        expected_num_shards=10,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.spark
@pytest.mark.cel
def test_smoke_cel_routing_context(spark, garage_s3_service, tmp_path) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        cel_expr="group",
        cel_columns={"group": "string"},
        boundaries=["b"],
        expected_num_shards=2,
        routing_context_fn=lambda row: {"group": row[2]},
        **_s3(garage_s3_service),
    )
