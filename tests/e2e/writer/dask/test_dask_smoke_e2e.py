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
    import dask
    import dask.dataframe as dd
    import pandas as pd

    from shardyfusion.serde import ValueSpec
    from tests.helpers.writer_api import write_dask_cel_sharded as write_cel_sharded
    from tests.helpers.writer_api import write_dask_hash_sharded as write_hash_sharded

    pdf = pd.DataFrame(data, columns=["key", "value", "group"])
    ddf = dd.from_pandas(pdf, npartitions=2)
    with dask.config.set(scheduler="synchronous"):
        if hasattr(config, "cel_expr"):
            return write_cel_sharded(
                ddf,
                config,
                key_col="key",
                value_spec=ValueSpec.binary_col("value"),
            )
        return write_hash_sharded(
            ddf,
            config,
            key_col="key",
            value_spec=ValueSpec.binary_col("value"),
        )


def _retry_write_fn(data, config):
    import dask
    import dask.dataframe as dd
    import pandas as pd

    from shardyfusion.serde import ValueSpec
    from tests.helpers.writer_api import write_dask_cel_sharded as write_cel_sharded
    from tests.helpers.writer_api import write_dask_hash_sharded as write_hash_sharded

    config.shard_retry = RetryConfig(
        max_retries=1,
        initial_backoff=timedelta(seconds=0),
    )
    pdf = pd.DataFrame(data, columns=["key", "value", "group"])
    ddf = dd.from_pandas(pdf, npartitions=2)
    with dask.config.set(scheduler="synchronous"):
        if hasattr(config, "cel_expr"):
            return write_cel_sharded(
                ddf,
                config,
                key_col="key",
                value_spec=ValueSpec.binary_col("value"),
            )
        return write_hash_sharded(
            ddf,
            config,
            key_col="key",
            value_spec=ValueSpec.binary_col("value"),
        )


# ---------------------------------------------------------------------------
# HASH sharding
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.dask
def test_smoke_hash(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
        num_dbs=3,
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.dask
def test_smoke_hash_num_dbs_2(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
        num_dbs=2,
        expected_num_shards=2,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.dask
def test_smoke_hash_retry_success(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _retry_write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
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
@pytest.mark.dask
def test_smoke_hash_max_keys_per_shard(garage_s3_service, tmp_path, backend) -> None:
    run_smoke_write_then_read_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
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
@pytest.mark.dask
@pytest.mark.cel
def test_smoke_cel_key_modulo(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
        cel_expr="key % 3",
        cel_columns={"key": "int"},
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.dask
@pytest.mark.cel
def test_smoke_cel_shard_hash(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
        cel_expr="shard_hash(key) % 3u",
        cel_columns={"key": "int"},
        expected_num_shards=3,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.dask
@pytest.mark.cel
def test_smoke_cel_key_identity(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
        cel_expr="uint(key)",
        cel_columns={"key": "int"},
        expected_num_shards=10,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )


@pytest.mark.e2e
@pytest.mark.dask
@pytest.mark.cel
def test_smoke_cel_routing_context(garage_s3_service, tmp_path, backend) -> None:
    pytest.importorskip("cel_expr_python")
    run_smoke_cel_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        writer_type="dask",
        cel_expr="group",
        cel_columns={"group": "string"},
        routing_values=["a", "b"],
        expected_num_shards=2,
        routing_context_fn=lambda row: {"group": row[2]},
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        **_s3(garage_s3_service),
    )
