"""End-to-end CLI history and rollback tests against Garage S3."""

from __future__ import annotations

import json

import pytest

from tests.e2e.cli.conftest import _invoke_cli, _invoke_cli_with_retry, _write_cli_configs


@pytest.mark.e2e
class TestCliHistory:
    def test_history_lists_manifests(
        self, garage_s3_service, tmp_path, backend, cli_rollback_prefix
    ) -> None:
        current_url = f"{cli_rollback_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["history"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "history"
        assert len(parsed["manifests"]) >= 1

    def test_history_limit(
        self, garage_s3_service, tmp_path, backend, cli_rollback_prefix
    ) -> None:
        current_url = f"{cli_rollback_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )
        result = _invoke_cli(tmp_path, ["history", "--limit", "1"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed["manifests"]) == 1


@pytest.mark.e2e
class TestCliRollback:
    def test_rollback_by_offset(
        self, garage_s3_service, tmp_path, backend, cli_rollback_prefix
    ) -> None:
        current_url = f"{cli_rollback_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )

        # Verify latest is v2
        result = _invoke_cli_with_retry(tmp_path, ["get", "0"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["value"] == "bmV3LTA="  # base64 of b"new-0"

        # Rollback to previous (v1)
        result = _invoke_cli(tmp_path, ["rollback", "--offset", "1"])
        assert result.exit_code == 0
        assert "Rolled back _CURRENT" in result.output

        # Verify rolled back (retry for S3 eventual consistency)
        parsed = json.loads(_invoke_cli_with_retry(tmp_path, ["get", "0"]).output)
        assert parsed["value"] == "b2xkLTA="  # base64 of b"old-0"


@pytest.mark.e2e
class TestCliRefDoesNotMutateCurrent:
    def test_ref_pin_does_not_mutate(
        self, garage_s3_service, tmp_path, backend, cli_rollback_prefix
    ) -> None:
        current_url = f"{cli_rollback_prefix}/_CURRENT"
        _write_cli_configs(
            tmp_path, garage_s3_service, current_url, reader_backend=backend.name
        )

        # Get latest info (should be v2)
        result = _invoke_cli_with_retry(tmp_path, ["info"])
        assert result.exit_code == 0
        latest_run_id = json.loads(result.output)["run_id"]

        # Use --ref to read an older manifest
        result = _invoke_cli(tmp_path, ["--offset", "1", "info"])
        assert result.exit_code == 0
        older_run_id = json.loads(result.output)["run_id"]
        assert older_run_id != latest_run_id

        # Verify un-pinned info still shows latest
        result = _invoke_cli_with_retry(tmp_path, ["info"])
        assert result.exit_code == 0
        assert json.loads(result.output)["run_id"] == latest_run_id
