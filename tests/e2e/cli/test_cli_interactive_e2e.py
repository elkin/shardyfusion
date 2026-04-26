"""End-to-end CLI interactive REPL tests against Garage S3."""

from __future__ import annotations

import pytest

from tests.e2e.cli.conftest import _invoke_cli_with_retry, _write_cli_configs


@pytest.mark.e2e
class TestCliInteractive:
    def test_interactive_get(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli_with_retry(tmp_path, [], input="get 1\nquit\n")
        assert result.exit_code == 0, result.output + (result.stderr or "")
        assert "b25l" in result.output  # base64 of b"one"

    def test_interactive_multiget(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli_with_retry(tmp_path, [], input="multiget 1 2\nquit\n")
        assert result.exit_code == 0, (
            f"exit_code={result.exit_code} stdout={result.output!r} stderr={result.stderr!r}"
        )
        # Interactive mode uses pretty-printed JSON which spans multiple lines,
        # so parse by extracting the JSON block containing "multiget".
        output = result.output
        assert '"op": "multiget"' in output
        assert '"key": "1"' in output
        assert '"key": "2"' in output
        assert '"found": true' in output

    def test_interactive_info(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli_with_retry(tmp_path, [], input="info\nquit\n")
        assert result.exit_code == 0, (
            f"exit_code={result.exit_code} stdout={result.output!r} stderr={result.stderr!r}"
        )
        # The banner + info output should contain the run_id
        assert "cli-e2e-run" in result.output

    def test_interactive_use_offset(
        self, garage_s3_service, tmp_path, backend, cli_rollback_prefix
    ) -> None:
        current_url = f"{cli_rollback_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli_with_retry(
            tmp_path,
            [],
            input="use --offset 1\ninfo\nquit\n",
            check_output=lambda r: "cli-e2e-run-v1" in r.output,
        )
        assert result.exit_code == 0, (
            f"exit_code={result.exit_code} stdout={result.output!r} stderr={result.stderr!r}"
        )
        # After use --offset 1, info should show the older run_id
        assert "cli-e2e-run-v1" in result.output

    def test_interactive_eof_exits(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli_with_retry(tmp_path, [], input="get 1\n")
        assert result.exit_code == 0, (
            f"exit_code={result.exit_code} stdout={result.output!r} stderr={result.stderr!r}"
        )
        assert "b25l" in result.output
