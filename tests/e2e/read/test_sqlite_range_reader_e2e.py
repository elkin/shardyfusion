"""E2E tests for the SQLite range-read VFS reader against Garage.

These tests exercise the APSW VFS path that serves point lookups via S3 Range
requests — the shard DB is never downloaded in full.  Unlike the parameterized
backend tests, these are SQLite-range-specific and always use
``SqliteFactory`` for writes and ``SqliteRangeReaderFactory`` for reads.
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.s3_test_scenarios import run_reader_loads_manifest_scenario
from tests.helpers.smoke_scenarios import run_smoke_write_then_read_scenario


def _s3(svc):
    return {
        "credential_provider": credential_provider_from_service(svc),
        "s3_connection_options": s3_connection_options_from_service(svc),
    }


def _sqlite_range_factories(svc):
    """Build SQLite write factory + range-read reader factory."""
    pytest.importorskip("apsw")
    from shardyfusion.sqlite_adapter import SqliteFactory, SqliteRangeReaderFactory

    opts = s3_connection_options_from_service(svc)
    creds = credential_provider_from_service(svc)
    return (
        SqliteFactory(s3_connection_options=opts, credential_provider=creds),
        SqliteRangeReaderFactory(s3_connection_options=opts, credential_provider=creds),
    )


# ---------------------------------------------------------------------------
# Smoke: HASH sharding roundtrip via range-read VFS
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_smoke_hash_range_reader(garage_s3_service, tmp_path) -> None:
    adapter_factory, reader_factory = _sqlite_range_factories(garage_s3_service)

    def _write_fn(data, config):
        from shardyfusion.writer.python import write_sharded

        return write_sharded(
            data, config, key_fn=lambda r: r[0], value_fn=lambda r: r[1]
        )

    run_smoke_write_then_read_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        num_dbs=3,
        expected_num_shards=3,
        adapter_factory=adapter_factory,
        reader_factory=reader_factory,
        **_s3(garage_s3_service),
    )


# ---------------------------------------------------------------------------
# Reader loads manifest via range-read VFS
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_reader_loads_manifest_range_reader(garage_s3_service, tmp_path) -> None:
    adapter_factory, reader_factory = _sqlite_range_factories(garage_s3_service)

    run_reader_loads_manifest_scenario(
        garage_s3_service,
        tmp_path,
        adapter_factory=adapter_factory,
        reader_factory=reader_factory,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
