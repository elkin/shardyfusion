"""End-to-end CLI cleanup tests against Garage S3."""

from __future__ import annotations

import json

import pytest

from tests.e2e.cli.conftest import _invoke_cli_with_retry, _write_cli_configs


@pytest.mark.e2e
class TestCliCleanup:
    def test_cleanup_dry_run(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli_with_retry(tmp_path, ["cleanup", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "cleanup"
        assert parsed["dry_run"] is True

    def test_cleanup_older_than(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli_with_retry(
            tmp_path, ["cleanup", "--older-than", "7d", "--dry-run"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "cleanup"
