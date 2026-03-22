"""Tests for cleanup_stale_attempts() and cleanup_old_runs() in _writer_core.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from shardyfusion._writer_core import (
    cleanup_old_runs,
    cleanup_stale_attempts,
)

# Patch targets: _writer_core imports these at module level via
# ``from .storage import ..., list_prefixes, delete_prefix``.
# We patch at the import site so the bound names are replaced.
_LP = "shardyfusion._writer_core.list_prefixes"
_DP = "shardyfusion._writer_core.delete_prefix"


def _fake_manifest(
    *,
    run_id: str = "run-abc",
    num_shards: int = 2,
    s3_prefix: str = "s3://bucket/prefix",
    shard_prefix: str = "shards",
    db_path_template: str = "db={db_id:05d}",
) -> MagicMock:
    """Build a mock ParsedManifest with the given structure."""
    manifest = MagicMock()
    manifest.required_build.run_id = run_id
    manifest.required_build.s3_prefix = s3_prefix
    manifest.required_build.shard_prefix = shard_prefix
    manifest.required_build.db_path_template = db_path_template

    shards = []
    for db_id in range(num_shards):
        shard = MagicMock()
        shard.db_id = db_id
        shard.db_url = (
            f"{s3_prefix}/{shard_prefix}/run_id={run_id}/"
            f"{db_path_template.format(db_id=db_id)}/attempt=00"
        )
        shards.append(shard)
    manifest.shards = shards
    return manifest


class TestCleanupStaleAttempts:
    def test_no_stale_attempts(self) -> None:
        """When only winner attempts exist, nothing is deleted."""
        manifest = _fake_manifest(num_shards=2)

        with patch(_LP, return_value=[]):
            actions = cleanup_stale_attempts(manifest, s3_client=MagicMock())

        assert actions == []

    def test_identifies_losers(self) -> None:
        """Non-winning attempt dirs are identified and deleted."""
        manifest = _fake_manifest(num_shards=1)
        winner_url = manifest.shards[0].db_url
        loser_url = winner_url.replace("attempt=00", "attempt=01")

        with (
            patch(_LP, return_value=[f"{winner_url}/", f"{loser_url}/"]),
            patch(_DP, return_value=5) as mock_delete,
        ):
            actions = cleanup_stale_attempts(manifest, s3_client=MagicMock())

        assert len(actions) == 1
        assert actions[0].kind == "stale_attempt"
        assert actions[0].db_id == 0
        assert actions[0].objects_deleted == 5
        assert "attempt=01" in actions[0].prefix_url
        mock_delete.assert_called_once()

    def test_dry_run_skips_deletion(self) -> None:
        """Dry-run mode records actions but does not call delete_prefix."""
        manifest = _fake_manifest(num_shards=1)
        winner_url = manifest.shards[0].db_url
        loser_url = winner_url.replace("attempt=00", "attempt=01")

        with (
            patch(_LP, return_value=[f"{winner_url}/", f"{loser_url}/"]),
            patch(_DP) as mock_delete,
        ):
            actions = cleanup_stale_attempts(
                manifest, s3_client=MagicMock(), dry_run=True
            )

        assert len(actions) == 1
        assert actions[0].objects_deleted == 0
        mock_delete.assert_not_called()

    def test_winner_url_trailing_slash_normalization(self) -> None:
        """Winner URLs are compared after stripping trailing slashes."""
        manifest = _fake_manifest(num_shards=1)
        winner_url_with_slash = f"{manifest.shards[0].db_url}/"

        with patch(_LP, return_value=[winner_url_with_slash]):
            actions = cleanup_stale_attempts(manifest, s3_client=MagicMock())

        assert actions == []

    def test_multiple_shards_multiple_losers(self) -> None:
        """Each shard's losers are found independently."""
        manifest = _fake_manifest(num_shards=2)

        def side_effect(prefix, **kwargs):
            for shard in manifest.shards:
                expected = (
                    f"s3://bucket/prefix/shards/run_id=run-abc/db={shard.db_id:05d}/"
                )
                if prefix == expected:
                    loser = shard.db_url.replace("attempt=00", "attempt=01")
                    return [f"{shard.db_url}/", f"{loser}/"]
            return []

        with (
            patch(_LP, side_effect=side_effect),
            patch(_DP, return_value=3),
        ):
            actions = cleanup_stale_attempts(manifest, s3_client=MagicMock())

        assert len(actions) == 2
        assert {a.db_id for a in actions} == {0, 1}


class TestCleanupOldRuns:
    def test_deletes_unprotected_runs(self) -> None:
        """Runs not in protected_run_ids are deleted."""
        run_dirs = [
            "s3://bucket/prefix/shards/run_id=run-old/",
            "s3://bucket/prefix/shards/run_id=run-keep/",
        ]

        with (
            patch(_LP, return_value=run_dirs),
            patch(_DP, return_value=10) as mock_delete,
        ):
            actions = cleanup_old_runs(
                "s3://bucket/prefix",
                "shards",
                protected_run_ids={"run-keep"},
                s3_client=MagicMock(),
            )

        assert len(actions) == 1
        assert actions[0].kind == "old_run"
        assert actions[0].run_id == "run-old"
        assert actions[0].objects_deleted == 10
        mock_delete.assert_called_once()

    def test_preserves_all_protected(self) -> None:
        """When all runs are protected, nothing is deleted."""
        run_dirs = [
            "s3://bucket/prefix/shards/run_id=run-a/",
            "s3://bucket/prefix/shards/run_id=run-b/",
        ]

        with patch(_LP, return_value=run_dirs):
            actions = cleanup_old_runs(
                "s3://bucket/prefix",
                "shards",
                protected_run_ids={"run-a", "run-b"},
                s3_client=MagicMock(),
            )

        assert actions == []

    def test_dry_run(self) -> None:
        """Dry-run records actions without deleting."""
        run_dirs = ["s3://bucket/prefix/shards/run_id=run-old/"]

        with (
            patch(_LP, return_value=run_dirs),
            patch(_DP) as mock_delete,
        ):
            actions = cleanup_old_runs(
                "s3://bucket/prefix",
                "shards",
                protected_run_ids=set(),
                s3_client=MagicMock(),
                dry_run=True,
            )

        assert len(actions) == 1
        assert actions[0].objects_deleted == 0
        mock_delete.assert_not_called()

    def test_skips_non_run_id_prefixes(self) -> None:
        """Directories that don't match run_id= pattern are skipped."""
        run_dirs = [
            "s3://bucket/prefix/shards/run_id=run-a/",
            "s3://bucket/prefix/shards/metadata/",
        ]

        with (
            patch(_LP, return_value=run_dirs),
            patch(_DP, return_value=1),
        ):
            actions = cleanup_old_runs(
                "s3://bucket/prefix",
                "shards",
                protected_run_ids=set(),
                s3_client=MagicMock(),
            )

        assert len(actions) == 1
        assert actions[0].run_id == "run-a"

    def test_empty_prefix(self) -> None:
        """No run directories at all."""
        with patch(_LP, return_value=[]):
            actions = cleanup_old_runs(
                "s3://bucket/prefix",
                "shards",
                protected_run_ids=set(),
                s3_client=MagicMock(),
            )

        assert actions == []
