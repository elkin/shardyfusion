"""Shared fixtures and helpers for CLI end-to-end tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click.testing
import pytest

from shardyfusion.cli.app import cli
from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.writer.python import write_sharded

if TYPE_CHECKING:
    from tests.conftest import LocalS3Service
    from tests.e2e.conftest import BackendFixture


# Override the parametrized backend fixture so CLI e2e tests only run
# against sqlite.  The existing e2e suite uses file-backed SlateDB
# (map_s3_db_url_to_file_url) which never hits real S3; the CLI
# constructs SlateDbReaderFactory with real S3 URLs, and SlateDB's
# object_store cannot resolve AWS credentials from the environment
# against Garage, causing a ~12 s IMDS timeout per test that makes
# the suite appear to hang.
@pytest.fixture
def backend(garage_s3_service: LocalS3Service, tmp_path: Path) -> BackendFixture:
    from tests.e2e.conftest import _sqlite_backend

    return _sqlite_backend(garage_s3_service)


def _s3(svc: LocalS3Service) -> dict[str, Any]:
    """Build credential_provider + s3_connection_options from a service dict."""
    from tests.e2e.conftest import (
        credential_provider_from_service,
        s3_connection_options_from_service,
    )

    return {
        "credential_provider": credential_provider_from_service(svc),
        "s3_connection_options": s3_connection_options_from_service(svc),
    }


def _write_cli_configs(
    tmp_path: Path,
    s3_service: LocalS3Service,
    current_url: str,
    *,
    reader_backend: str = "slatedb",
) -> tuple[Path, Path]:
    """Write reader.toml + credentials.toml into *tmp_path*.

    Returns (reader_config_path, credentials_path).
    """
    reader_toml = f"""\
[reader]
current_url = {current_url!r}
reader_backend = {reader_backend!r}

[output]
format = "jsonl"
"""
    creds_toml = f"""\
[default]
endpoint_url = {s3_service["endpoint_url"]!r}
region = {s3_service["region_name"]!r}
access_key_id = {s3_service["access_key_id"]!r}
secret_access_key = {s3_service["secret_access_key"]!r}
addressing_style = "path"
"""
    reader_path = tmp_path / "reader.toml"
    creds_path = tmp_path / "credentials.toml"
    reader_path.write_text(reader_toml)
    creds_path.write_text(creds_toml)
    return reader_path, creds_path


def _invoke_cli(
    tmp_path: Path,
    args: list[str],
    input: str | None = None,  # noqa: A002
    env: dict[str, str] | None = None,
) -> click.testing.Result:
    """Invoke the shardy CLI with config files resolved from *tmp_path*."""
    runner = click.testing.CliRunner()
    base_env = {"SHARDY_CONFIG": str(tmp_path / "reader.toml")}
    if env:
        base_env.update(env)
    return runner.invoke(cli, args, input=input, env=base_env)


def _write_kv_data(
    s3_service: LocalS3Service,
    tmp_path: Path,
    backend: BackendFixture,
    num_dbs: int = 3,
) -> str:
    """Write a small KV snapshot via the Python writer and return the S3 prefix."""
    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/cli-e2e-{tmp_path.name}"
    local_root = str(tmp_path / "writer-local")

    records = [
        (0, b"zero"),
        (1, b"one"),
        (2, b"two"),
        (3, b"three"),
        (4, b"four"),
        (5, b"five"),
        (6, b"six"),
        (7, b"seven"),
        (8, b"eight"),
        (9, b"nine"),
    ]

    s3_kwargs = _s3(s3_service)
    config = WriteConfig(
        num_dbs=num_dbs,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        adapter_factory=backend.adapter_factory,
        manifest=ManifestOptions(**s3_kwargs),
        output=OutputOptions(run_id="cli-e2e-run", local_root=local_root),
    )

    write_sharded(
        records,
        config,
        key_fn=lambda r: r[0],
        value_fn=lambda r: r[1],
    )
    return s3_prefix


def _write_two_kv_manifests(
    s3_service: LocalS3Service,
    tmp_path: Path,
    backend: BackendFixture,
) -> str:
    """Publish two manifests so history/rollback tests have data to work with."""
    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/cli-e2e-rollback-{tmp_path.name}"
    local_root = str(tmp_path / "writer-local")

    s3_kwargs = _s3(s3_service)

    def _build_config(run_id: str) -> WriteConfig:
        return WriteConfig(
            num_dbs=2,
            s3_prefix=s3_prefix,
            sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
            adapter_factory=backend.adapter_factory,
            manifest=ManifestOptions(**s3_kwargs),
            output=OutputOptions(run_id=run_id, local_root=local_root),
        )

    write_sharded(
        [(i, f"old-{i}".encode()) for i in range(8)],
        _build_config("cli-e2e-run-v1"),
        key_fn=lambda r: r[0],
        value_fn=lambda r: r[1],
    )
    write_sharded(
        [(i, f"new-{i}".encode()) for i in range(8)],
        _build_config("cli-e2e-run-v2"),
        key_fn=lambda r: r[0],
        value_fn=lambda r: r[1],
    )
    return s3_prefix


@pytest.fixture
def cli_kv_prefix(
    garage_s3_service: LocalS3Service,
    tmp_path: Path,
    backend: BackendFixture,
) -> str:
    """Yield an S3 prefix with a single published KV snapshot."""
    return _write_kv_data(garage_s3_service, tmp_path, backend)


@pytest.fixture
def cli_rollback_prefix(
    garage_s3_service: LocalS3Service,
    tmp_path: Path,
    backend: BackendFixture,
) -> str:
    """Yield an S3 prefix with two published KV snapshots."""
    return _write_two_kv_manifests(garage_s3_service, tmp_path, backend)
