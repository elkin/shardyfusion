from __future__ import annotations

from datetime import timedelta

import pytest

from shardyfusion.testing import FailOnceAdapterFactory
from shardyfusion.type_defs import RetryConfig
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


def _write_fn(data, config):
    import ray.data

    from shardyfusion.config import CelWriteConfig, HashWriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.writer.ray import write_sharded_by_cel, write_sharded_by_hash

    items = [{"key": k, "value": v, "group": g} for k, v, g in data]
    ds = ray.data.from_items(items, override_num_blocks=2)
    if isinstance(config, CelWriteConfig):
        return write_sharded_by_cel(
            ds, config, key_col="key", value_spec=ValueSpec.binary_col("value")
        )
    return write_sharded_by_hash(
        ds, config, key_col="key", value_spec=ValueSpec.binary_col("value")
    )


def _retry_write_fn(data, config):
    import ray.data

    from shardyfusion.config import CelWriteConfig, HashWriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.writer.ray import write_sharded_by_cel, write_sharded_by_hash

    config.shard_retry = RetryConfig(
        max_retries=1,
        initial_backoff=timedelta(seconds=0),
    )
    items = [{"key": k, "value": v, "group": g} for k, v, g in data]
    ds = ray.data.from_items(items, override_num_blocks=2)
    if isinstance(config, CelWriteConfig):
        return write_sharded_by_cel(
            ds, config, key_col="key", value_spec=ValueSpec.binary_col("value")
        )
    return write_sharded_by_hash(
        ds, config, key_col="key", value_spec=ValueSpec.binary_col("value")
    )


# ---------------------------------------------------------------------------
# HASH sharding
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.ray
def test_smoke_hash(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        num_dbs=3,
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.ray
def test_smoke_hash_num_dbs_2(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        num_dbs=2,
        expected_num_shards=2,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.ray
def test_smoke_hash_retry_success(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _retry_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        expect_retry=True,
        num_dbs=3,
        expected_num_shards=3,
        adapter_factory=FailOnceAdapterFactory(
            backend.adapter_factory,
            marker_root=str(tmp_path / "retry-markers"),
            fail_db_ids=(0,),
        ),
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.ray
def test_smoke_hash_max_keys_per_shard(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        num_dbs=None,
        max_keys_per_shard=5,
        expected_num_shards=2,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


# ---------------------------------------------------------------------------
# CEL sharding
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.ray
@pytest.mark.cel
def test_smoke_cel_key_modulo(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        cel_expr="key % 3",
        cel_columns={"key": "int"},
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.ray
@pytest.mark.cel
def test_smoke_cel_shard_hash(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        cel_expr="shard_hash(key) % 3u",
        cel_columns={"key": "int"},
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.ray
@pytest.mark.cel
def test_smoke_cel_key_identity(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        cel_expr="uint(key)",
        cel_columns={"key": "int"},
        expected_num_shards=10,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.ray
@pytest.mark.cel
def test_smoke_cel_routing_context(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="ray",
        cel_expr="group",
        cel_columns={"group": "string"},
        routing_values=["a", "b"],
        expected_num_shards=2,
        routing_context_fn=lambda row: {"group": row[2]},
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )
