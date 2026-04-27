"""Tests for cleanup_stale_attempts() and cleanup_old_runs() in _writer_core.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from shardyfusion._writer_core import (
    cleanup_old_runs,
    cleanup_stale_attempts,
)
from shardyfusion.storage import MemoryBackend


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
        backend = MemoryBackend()
        # Pre-populate with winner attempt keys so list_prefixes finds them
        for shard in manifest.shards:
            backend.put(f"{shard.db_url}/some_object", b"x", "application/octet-stream")

        actions = cleanup_stale_attempts(manifest, backend=backend)
        assert actions == []

    def test_identifies_losers(self) -> None:
        """Non-winning attempt dirs are identified and deleted."""
        manifest = _fake_manifest(num_shards=1)
        winner_url = manifest.shards[0].db_url
        loser_url = winner_url.replace("attempt=00", "attempt=01")

        backend = MemoryBackend()
        backend.put(f"{winner_url}/obj", b"w", "application/octet-stream")
        backend.put(f"{loser_url}/obj", b"l", "application/octet-stream")

        actions = cleanup_stale_attempts(manifest, backend=backend)

        assert len(actions) == 1
        assert actions[0].kind == "stale_attempt"
        assert actions[0].db_id == 0
        assert actions[0].objects_deleted == 1
        assert "attempt=01" in actions[0].prefix_url
        # Loser data removed, winner preserved
        assert backend.get(f"{winner_url}/obj") == b"w"
        assert backend.try_get(f"{loser_url}/obj") is None

    def test_dry_run_skips_deletion(self) -> None:
        """Dry-run mode records actions but does not delete."""
        manifest = _fake_manifest(num_shards=1)
        winner_url = manifest.shards[0].db_url
        loser_url = winner_url.replace("attempt=00", "attempt=01")

        backend = MemoryBackend()
        backend.put(f"{winner_url}/obj", b"w", "application/octet-stream")
        backend.put(f"{loser_url}/obj", b"l", "application/octet-stream")

        actions = cleanup_stale_attempts(manifest, backend=backend, dry_run=True)

        assert len(actions) == 1
        assert actions[0].objects_deleted == 0
        # Data still present
        assert backend.get(f"{loser_url}/obj") == b"l"

    def test_winner_url_trailing_slash_normalization(self) -> None:
        """Winner URLs are compared after stripping trailing slashes."""
        manifest = _fake_manifest(num_shards=1)
        winner_url = manifest.shards[0].db_url

        backend = MemoryBackend()
        backend.put(f"{winner_url}/obj", b"w", "application/octet-stream")

        actions = cleanup_stale_attempts(manifest, backend=backend)
        assert actions == []

    def test_multiple_shards_multiple_losers(self) -> None:
        """Each shard's losers are found independently."""
        manifest = _fake_manifest(num_shards=2)

        backend = MemoryBackend()
        for shard in manifest.shards:
            winner = shard.db_url
            loser = winner.replace("attempt=00", "attempt=01")
            backend.put(f"{winner}/obj", b"w", "application/octet-stream")
            backend.put(f"{loser}/obj", b"l", "application/octet-stream")

        actions = cleanup_stale_attempts(manifest, backend=backend)

        assert len(actions) == 2
        assert {a.db_id for a in actions} == {0, 1}


class TestCleanupOldRuns:
    def test_deletes_unprotected_runs(self) -> None:
        """Runs not in protected_run_ids are deleted."""
        backend = MemoryBackend()
        backend.put(
            "s3://bucket/prefix/shards/run_id=run-old/obj",
            b"x",
            "application/octet-stream",
        )
        backend.put(
            "s3://bucket/prefix/shards/run_id=run-keep/obj",
            b"x",
            "application/octet-stream",
        )

        actions = cleanup_old_runs(
            "s3://bucket/prefix",
            "shards",
            protected_run_ids={"run-keep"},
            backend=backend,
        )

        assert len(actions) == 1
        assert actions[0].kind == "old_run"
        assert actions[0].run_id == "run-old"
        assert actions[0].objects_deleted == 1
        # Verify deletion
        assert backend.try_get("s3://bucket/prefix/shards/run_id=run-old/obj") is None
        assert backend.get("s3://bucket/prefix/shards/run_id=run-keep/obj") == b"x"

    def test_preserves_all_protected(self) -> None:
        """When all runs are protected, nothing is deleted."""
        backend = MemoryBackend()
        backend.put(
            "s3://bucket/prefix/shards/run_id=run-a/obj",
            b"x",
            "application/octet-stream",
        )
        backend.put(
            "s3://bucket/prefix/shards/run_id=run-b/obj",
            b"x",
            "application/octet-stream",
        )

        actions = cleanup_old_runs(
            "s3://bucket/prefix",
            "shards",
            protected_run_ids={"run-a", "run-b"},
            backend=backend,
        )

        assert actions == []

    def test_dry_run(self) -> None:
        """Dry-run records actions without deleting."""
        backend = MemoryBackend()
        backend.put(
            "s3://bucket/prefix/shards/run_id=run-old/obj",
            b"x",
            "application/octet-stream",
        )

        actions = cleanup_old_runs(
            "s3://bucket/prefix",
            "shards",
            protected_run_ids=set(),
            backend=backend,
            dry_run=True,
        )

        assert len(actions) == 1
        assert actions[0].objects_deleted == 0
        assert backend.get("s3://bucket/prefix/shards/run_id=run-old/obj") == b"x"

    def test_skips_non_run_id_prefixes(self) -> None:
        """Directories that don't match run_id= pattern are skipped."""
        backend = MemoryBackend()
        backend.put(
            "s3://bucket/prefix/shards/run_id=run-a/obj",
            b"x",
            "application/octet-stream",
        )
        backend.put(
            "s3://bucket/prefix/shards/metadata/obj", b"x", "application/octet-stream"
        )

        actions = cleanup_old_runs(
            "s3://bucket/prefix",
            "shards",
            protected_run_ids=set(),
            backend=backend,
        )

        assert len(actions) == 1
        assert actions[0].run_id == "run-a"

    def test_empty_prefix(self) -> None:
        """No run directories at all."""
        backend = MemoryBackend()
        actions = cleanup_old_runs(
            "s3://bucket/prefix",
            "shards",
            protected_run_ids=set(),
            backend=backend,
        )
        assert actions == []
