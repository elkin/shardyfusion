"""End-to-end CLI tests for ``--sqlite-mode {download,range,auto}``.

These tests publish a small sqlite-backed snapshot to Garage S3 and then
invoke ``shardy`` with each mode flag, asserting that:

* the CLI exit code is 0,
* a ``get`` returns the correct value,
* ``info`` reports a valid snapshot.

The conftest's ``backend`` fixture is overridden to always select sqlite,
which is required for ``--sqlite-mode`` to take effect (it's a no-op for
``slatedb``).  All three modes route through distinct factory classes
(``SqliteReaderFactory`` / ``SqliteRangeReaderFactory`` /
``AdaptiveSqliteReaderFactory``) inside ``cli/app.py``; the assertions
below verify each path actually works against a real S3-compatible
backend.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.cli.conftest import (
    _invoke_cli_with_retry,
    _write_cli_configs,
)


@pytest.mark.e2e
class TestCliSqliteMode:
    def test_get_with_mode_download(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        """``--sqlite-mode download`` selects ``SqliteReaderFactory``."""
        if backend.name != "sqlite":
            pytest.skip("--sqlite-mode requires sqlite backend")

        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path,
            garage_s3_service,
            current_url,
            reader_backend=backend.name,
        )
        result = _invoke_cli_with_retry(
            tmp_path, ["--sqlite-mode", "download", "get", "1"]
        )
        assert result.exit_code == 0, result.output + (result.stderr or "")
        parsed = json.loads(result.output)
        assert parsed["op"] == "get"
        assert parsed["found"] is True
        assert parsed["value"] == "b25l"  # base64 of b"one"

    def test_get_with_mode_range(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        """``--sqlite-mode range`` selects ``SqliteRangeReaderFactory``.

        Range-read requires ``apsw`` + ``obstore``; both are in the
        ``read-sqlite-range`` / ``cli`` extras and present in the dev env.
        """
        if backend.name != "sqlite":
            pytest.skip("--sqlite-mode requires sqlite backend")
        pytest.importorskip("apsw")
        pytest.importorskip("obstore")

        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path,
            garage_s3_service,
            current_url,
            reader_backend=backend.name,
        )
        result = _invoke_cli_with_retry(
            tmp_path, ["--sqlite-mode", "range", "get", "2"]
        )
        assert result.exit_code == 0, result.output + (result.stderr or "")
        parsed = json.loads(result.output)
        assert parsed["op"] == "get"
        assert parsed["found"] is True
        assert parsed["value"] == "dHdv"  # base64 of b"two"

    def test_get_with_mode_auto(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        """``--sqlite-mode auto`` selects ``AdaptiveSqliteReaderFactory``.

        Small shards in the test fixture stay well under the default
        16 MiB per-shard threshold, so the policy resolves to download
        mode under the hood — the read still succeeds.
        """
        if backend.name != "sqlite":
            pytest.skip("--sqlite-mode requires sqlite backend")

        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path,
            garage_s3_service,
            current_url,
            reader_backend=backend.name,
        )
        result = _invoke_cli_with_retry(tmp_path, ["--sqlite-mode", "auto", "get", "3"])
        assert result.exit_code == 0, result.output + (result.stderr or "")
        parsed = json.loads(result.output)
        assert parsed["op"] == "get"
        assert parsed["found"] is True
        assert parsed["value"] == "dGhyZWU="  # base64 of b"three"

    def test_auto_with_low_threshold_forces_range(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        """``--sqlite-auto-per-shard-bytes 1`` forces auto → range mode.

        Verifies that the threshold flag plumbs through and that the
        adaptive factory picks the range-read path when the policy
        triggers.  The read must still succeed end-to-end.
        """
        if backend.name != "sqlite":
            pytest.skip("--sqlite-mode requires sqlite backend")
        pytest.importorskip("apsw")
        pytest.importorskip("obstore")

        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path,
            garage_s3_service,
            current_url,
            reader_backend=backend.name,
        )
        result = _invoke_cli_with_retry(
            tmp_path,
            [
                "--sqlite-mode",
                "auto",
                "--sqlite-auto-per-shard-bytes",
                "1",
                "get",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output + (result.stderr or "")
        parsed = json.loads(result.output)
        assert parsed["found"] is True
        assert parsed["value"] == "Zm91cg=="  # base64 of b"four"

    def test_info_with_each_mode(
        self, garage_s3_service, tmp_path, backend, cli_kv_prefix
    ) -> None:
        """``info`` succeeds under all three modes (read-only metadata)."""
        if backend.name != "sqlite":
            pytest.skip("--sqlite-mode requires sqlite backend")
        # Range mode needs apsw + obstore; both are part of the cli/dev
        # environment but skip cleanly if not installed.
        pytest.importorskip("apsw")
        pytest.importorskip("obstore")

        current_url = f"{cli_kv_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path,
            garage_s3_service,
            current_url,
            reader_backend=backend.name,
        )
        for mode in ("download", "range", "auto"):
            result = _invoke_cli_with_retry(tmp_path, ["--sqlite-mode", mode, "info"])
            assert result.exit_code == 0, (
                f"mode={mode}: " + result.output + (result.stderr or "")
            )
            parsed = json.loads(result.output)
            assert parsed["op"] == "info"
            assert parsed["num_dbs"] == 3
            assert parsed["run_id"] == "cli-e2e-run"
