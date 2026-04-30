"""Unit tests for the shardy cleanup CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import click.testing
import pytest

from shardyfusion._writer_core import CleanupAction
from shardyfusion.cli.app import _parse_duration, cli
from shardyfusion.manifest import ManifestRef


def _make_manifest_ref(run_id: str, age_hours: int = 0) -> ManifestRef:
    """Create a ManifestRef with published_at relative to now."""
    return ManifestRef(
        ref=f"s3://bucket/prefix/manifests/{run_id}/manifest",
        run_id=run_id,
        published_at=datetime.now(UTC) - timedelta(hours=age_hours),
    )


def _make_manifest_mock(
    run_id: str = "run-current",
    num_shards: int = 2,
    s3_prefix: str = "s3://bucket/prefix",
    shard_prefix: str = "shards",
) -> MagicMock:
    """Build a mock ParsedManifest."""
    m = MagicMock()
    m.required_build.run_id = run_id
    m.required_build.s3_prefix = s3_prefix
    m.required_build.shard_prefix = shard_prefix
    m.required_build.db_path_template = "db={db_id:05d}"
    shards = []
    for db_id in range(num_shards):
        shard = MagicMock()
        shard.db_id = db_id
        shard.db_url = (
            f"{s3_prefix}/{shard_prefix}/run_id={run_id}/db={db_id:05d}/attempt=00"
        )
        shards.append(shard)
    m.shards = shards
    return m


def _invoke_cleanup(
    args: list[str],
    *,
    stale_actions: list[CleanupAction] | None = None,
    old_run_actions: list[CleanupAction] | None = None,
    manifest: MagicMock | None = None,
    manifest_refs: list[ManifestRef] | None = None,
    env: dict[str, str] | None = None,
) -> click.testing.Result:
    """Invoke 'shardy cleanup' with mocked internals."""
    if manifest is None:
        manifest = _make_manifest_mock()

    store = MagicMock()
    current_ref = MagicMock()
    current_ref.ref = "s3://bucket/prefix/manifests/current/manifest"
    store.load_current.return_value = current_ref
    store.load_manifest.return_value = manifest
    store.list_manifests.return_value = manifest_refs or []

    effective_env = {"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"}
    if env:
        effective_env.update(env)

    with (
        patch("shardyfusion.cli.app._build_manifest_store", return_value=store),
        patch(
            "shardyfusion.cli.app.cleanup_stale_attempts",
            return_value=stale_actions or [],
        ),
        patch(
            "shardyfusion.cli.app.cleanup_old_runs",
            return_value=old_run_actions or [],
        ),
    ):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, ["cleanup", *args], env=effective_env)


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


class TestParseDuration:
    def test_days(self) -> None:
        assert _parse_duration("7d") == timedelta(days=7)

    def test_hours(self) -> None:
        assert _parse_duration("24h") == timedelta(hours=24)

    def test_invalid_unit(self) -> None:
        with pytest.raises(click.BadParameter, match="Unknown duration unit"):
            _parse_duration("5m")

    def test_invalid_number(self) -> None:
        with pytest.raises(click.BadParameter, match="Invalid duration"):
            _parse_duration("abcd")

    def test_empty(self) -> None:
        with pytest.raises(click.BadParameter, match="must not be empty"):
            _parse_duration("")


# ---------------------------------------------------------------------------
# Basic cleanup
# ---------------------------------------------------------------------------


class TestCleanupBasic:
    def test_no_stale_attempts(self) -> None:
        """No losers found → empty result."""
        result = _invoke_cleanup([])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "cleanup"
        assert parsed["dry_run"] is False
        assert parsed["stale_attempts"] == []
        assert parsed["total_objects_deleted"] == 0
        assert parsed["total_prefixes_removed"] == 0

    def test_stale_attempts_found(self) -> None:
        """Stale attempts are reported."""
        actions = [
            CleanupAction(
                kind="stale_attempt",
                prefix_url="s3://bucket/prefix/shards/run_id=run-current/db=00000/attempt=01/",
                db_id=0,
                run_id="run-current",
                objects_deleted=5,
            ),
        ]
        result = _invoke_cleanup([], stale_actions=actions)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed["stale_attempts"]) == 1
        assert parsed["stale_attempts"][0]["db_id"] == 0
        assert parsed["stale_attempts"][0]["objects_deleted"] == 5
        assert parsed["total_objects_deleted"] == 5
        assert parsed["total_prefixes_removed"] == 1

    def test_dry_run(self) -> None:
        """Dry-run flag is passed through."""
        actions = [
            CleanupAction(
                kind="stale_attempt",
                prefix_url="s3://b/p/attempt=01/",
                db_id=0,
                run_id="run-current",
                objects_deleted=0,
            ),
        ]
        result = _invoke_cleanup(["--dry-run"], stale_actions=actions)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["dry_run"] is True


# ---------------------------------------------------------------------------
# Old runs cleanup
# ---------------------------------------------------------------------------


class TestCleanupOldRuns:
    def test_include_old_runs(self) -> None:
        """--include-old-runs triggers old_run cleanup."""
        old_actions = [
            CleanupAction(
                kind="old_run",
                prefix_url="s3://bucket/prefix/shards/run_id=orphan/",
                db_id=None,
                run_id="orphan",
                objects_deleted=20,
            ),
        ]
        refs = [_make_manifest_ref("run-current", age_hours=0)]
        result = _invoke_cleanup(
            ["--include-old-runs"],
            old_run_actions=old_actions,
            manifest_refs=refs,
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "old_runs" in parsed
        assert len(parsed["old_runs"]) == 1
        assert parsed["old_runs"][0]["run_id"] == "orphan"

    def test_older_than(self) -> None:
        """--older-than triggers old_run cleanup."""
        result = _invoke_cleanup(
            ["--older-than", "7d"],
            manifest_refs=[_make_manifest_ref("run-current", age_hours=0)],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "cleanup"

    def test_keep_last(self) -> None:
        """--keep-last triggers old_run cleanup."""
        refs = [
            _make_manifest_ref("run-1", age_hours=0),
            _make_manifest_ref("run-2", age_hours=24),
            _make_manifest_ref("run-3", age_hours=48),
        ]
        result = _invoke_cleanup(
            ["--keep-last", "2"],
            manifest_refs=refs,
        )
        assert result.exit_code == 0

    def test_keep_last_zero_rejected(self) -> None:
        """--keep-last 0 is rejected."""
        refs = [_make_manifest_ref("run-1")]
        result = _invoke_cleanup(["--keep-last", "0"], manifest_refs=refs)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Max retries
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
# Text output format
# ---------------------------------------------------------------------------


class TestCleanupTextOutput:
    def test_text_format(self) -> None:
        """Text format shows human-readable summary."""
        actions = [
            CleanupAction(
                kind="stale_attempt",
                prefix_url="s3://b/p/attempt=01/",
                db_id=0,
                run_id="run-current",
                objects_deleted=5,
            ),
        ]
        result = _invoke_cleanup(
            ["--dry-run"],
            stale_actions=actions,
            env={
                "SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT",
            },
        )
        assert result.exit_code == 0
        # Text format should mention cleanup info in JSON (default format)
        parsed = json.loads(result.output)
        assert parsed["dry_run"] is True

    def test_text_output_format_explicit(self) -> None:
        """When --output-format=text is used, the text formatter fires."""
        actions = [
            CleanupAction(
                kind="stale_attempt",
                prefix_url="s3://b/p/attempt=01/",
                db_id=0,
                run_id="run-current",
                objects_deleted=5,
            ),
        ]
        manifest = _make_manifest_mock()
        store = MagicMock()
        current_ref = MagicMock()
        current_ref.ref = "ref"
        store.load_current.return_value = current_ref
        store.load_manifest.return_value = manifest

        with (
            patch("shardyfusion.cli.app._build_manifest_store", return_value=store),
            patch(
                "shardyfusion.cli.app.cleanup_stale_attempts",
                return_value=actions,
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                ["--output-format", "text", "cleanup"],
                env={"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"},
            )

        assert result.exit_code == 0
        assert "Cleanup for run_id=" in result.output
        assert "stale attempt" in result.output
        assert "Total:" in result.output


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCleanupErrors:
    def test_no_current_manifest(self) -> None:
        """Error when no current manifest exists."""
        store = MagicMock()
        store.load_current.return_value = None

        with patch("shardyfusion.cli.app._build_manifest_store", return_value=store):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                ["cleanup"],
                env={"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"},
            )

        assert result.exit_code != 0
