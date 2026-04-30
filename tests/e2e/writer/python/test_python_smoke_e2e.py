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


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _hash_write_fn(data, config):
    from shardyfusion.writer.python import write_sharded_by_hash

    return write_sharded_by_hash(
        data, config, key_fn=lambda r: r[0], value_fn=lambda r: r[1]
    )


def _cel_context_write_fn(data, config):
    """Like _hash_write_fn but provides the ``group`` column for CEL routing."""
    from shardyfusion.writer.python import write_sharded_by_cel

    return write_sharded_by_cel(
        data,
        config,
        key_fn=lambda r: r[0],
        value_fn=lambda r: r[1],
        columns_fn=lambda r: {"group": r[2]},
    )


def _hash_retry_write_fn(data, config):
    from shardyfusion.writer.python import write_sharded_by_hash

    config.shard_retry = RetryConfig(
        max_retries=1,
        initial_backoff=timedelta(seconds=0),
    )
    return write_sharded_by_hash(
        data,
        config,
        key_fn=lambda r: r[0],
        value_fn=lambda r: r[1],
        parallel=True,
    )


# ---------------------------------------------------------------------------
# HASH sharding
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_smoke_hash(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _hash_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
        num_dbs=3,
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
def test_smoke_hash_num_dbs_2(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _hash_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
        num_dbs=2,
        expected_num_shards=2,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
def test_smoke_hash_retry_success(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _hash_retry_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
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
def test_smoke_hash_max_keys_per_shard(garage_s3_service, tmp_path, backend) -> None:
    # 10 rows / 5 per shard → ceil(10/5) = 2 shards
    run_smoke_write_then_read_scenario(
        _hash_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
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
@pytest.mark.cel
def test_smoke_cel_key_modulo(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _hash_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
        cel_expr="key % 3",
        cel_columns={"key": "int"},
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.cel
def test_smoke_cel_shard_hash(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    # shard_hash(key) % 3u → direct mode, 3 shards
    run_smoke_cel_scenario(
        _hash_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
        cel_expr="shard_hash(key) % 3u",
        cel_columns={"key": "int"},
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.cel
def test_smoke_cel_key_identity(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    # key itself as shard id (direct mode) → 10 shards (keys 0-9)
    run_smoke_cel_scenario(
        _hash_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
        cel_expr="uint(key)",
        cel_columns={"key": "int"},
        expected_num_shards=10,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.cel
def test_smoke_cel_routing_context(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    # Route by "group" column: "a" → shard 0, "b" → shard 1
    run_smoke_cel_scenario(
        _cel_context_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="python",
        cel_expr="group",
        cel_columns={"group": "string"},
        routing_values=["a", "b"],
        expected_num_shards=2,
        routing_context_fn=lambda row: {"group": row[2]},
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )
