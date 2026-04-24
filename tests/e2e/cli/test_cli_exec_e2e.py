"""End-to-end CLI batch exec tests against Garage S3."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.cli.conftest import _invoke_cli, _write_cli_configs


def _write_script(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "script.yaml"
    path.write_text(content)
    return path


@pytest.mark.e2e
class TestCliExec:
    def test_exec_batch_script(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        script = _write_script(
            tmp_path,
            """\
commands:
  - op: get
    key: 1
  - op: info
""",
        )
        result = _invoke_cli(tmp_path, ["exec", "--script", str(script)])
        assert result.exit_code == 0, result.output + (result.stderr or "")
        lines = result.output.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["op"] == "get"
        assert json.loads(lines[1])["op"] == "info"

    def test_exec_on_error_stop(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        script = _write_script(
            tmp_path,
            """\
on_error: stop
commands:
  - op: bogus
  - op: info
""",
        )
        result = _invoke_cli(tmp_path, ["exec", "--script", str(script)])
        assert result.exit_code == 1
        lines = result.output.strip().split("\n")
        assert len(lines) == 1
        assert "error" in json.loads(lines[0])

    def test_exec_output_file(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        script = _write_script(
            tmp_path,
            """\
commands:
  - op: get
    key: 1
""",
        )
        out_path = tmp_path / "out.jsonl"
        result = _invoke_cli(
            tmp_path, ["exec", "--script", str(script), "--output", str(out_path)]
        )
        assert result.exit_code == 0
        assert result.output.strip() == ""
        assert out_path.exists()
        assert json.loads(out_path.read_text())["op"] == "get"
