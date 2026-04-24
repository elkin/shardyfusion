"""End-to-end CLI output format tests against Garage S3."""

from __future__ import annotations

import json

import pytest

from tests.e2e.cli.conftest import _invoke_cli, _write_cli_configs


@pytest.mark.e2e
class TestCliFormats:
    def test_format_jsonl_default(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["get", "1"])
        assert result.exit_code == 0
        # jsonl is single-line JSON
        lines = result.output.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["op"] == "get"

    def test_format_json(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["--output-format", "json", "get", "1"])
        assert result.exit_code == 0
        # json is pretty-printed (multi-line)
        assert "\n" in result.output
        parsed = json.loads(result.output)
        assert parsed["op"] == "get"

    def test_format_text_get(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["--output-format", "text", "get", "1"])
        assert result.exit_code == 0
        assert "1=" in result.output

    def test_format_table_multiget(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(
            tmp_path, ["--output-format", "table", "multiget", "1", "2"]
        )
        assert result.exit_code == 0
        assert "KEY" in result.output
        assert "VALUE" in result.output

    def test_format_text_shards(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["--output-format", "text", "shards"])
        assert result.exit_code == 0
        assert "db_id=" in result.output

    def test_format_text_health(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["--output-format", "text", "health"])
        assert result.exit_code == 0
        assert "status=" in result.output
