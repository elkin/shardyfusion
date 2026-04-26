"""End-to-end CLI read tests against Garage S3."""

from __future__ import annotations

import json

import pytest

from tests.e2e.cli.conftest import _invoke_cli, _write_cli_configs


@pytest.mark.e2e
class TestCliGet:
    def test_get_found(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["get", "1"])
        assert result.exit_code == 0, result.output + (result.stderr or "")
        parsed = json.loads(result.output)
        assert parsed["op"] == "get"
        assert parsed["found"] is True
        assert parsed["value"] == "b25l"  # base64 of b"one"

    def test_get_not_found(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["get", "999"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["found"] is False

    def test_get_strict_not_found(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["get", "--strict", "999"])
        assert result.exit_code == 1, f"stdout={result.output!r} stderr={result.stderr!r}"
        parsed = json.loads(result.output)
        assert parsed.get("found") is False, f"unexpected output: {parsed!r}"


@pytest.mark.e2e
class TestCliMultiget:
    def test_multiget(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["multiget", "1", "2", "3"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "multiget"
        assert len(parsed["results"]) == 3
        assert all(r["found"] is True for r in parsed["results"])

    def test_multiget_stdin(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["multiget", "-"], input="1\n2\n3\n")
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "multiget"
        assert len(parsed["results"]) == 3


@pytest.mark.e2e
class TestCliRoute:
    def test_route(self, garage_s3_service, tmp_path, backend, cli_kv_prefix) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["route", "5"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "route"
        assert "db_id" in parsed


@pytest.mark.e2e
class TestCliInfo:
    def test_info(self, garage_s3_service, tmp_path, backend, cli_kv_prefix) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["info"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "info"
        assert parsed["run_id"] == "cli-e2e-run"


@pytest.mark.e2e
class TestCliShards:
    def test_shards(self, garage_s3_service, tmp_path, backend, cli_kv_prefix) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["shards"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "shards"
        assert len(parsed["shards"]) == 3


@pytest.mark.e2e
class TestCliHealth:
    def test_health_healthy(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["health"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "health"
        assert parsed["status"] == "healthy"

    def test_health_degraded(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["health", "--staleness-threshold", "0.001"])
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["status"] == "degraded"
