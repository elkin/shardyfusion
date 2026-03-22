"""Tests for empty-shard handling in select_winners, cleanup_losers, cleanup_stale_attempts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from shardyfusion._writer_core import (
    ShardAttemptResult,
    cleanup_losers,
    cleanup_stale_attempts,
    select_winners,
)
from shardyfusion.manifest import RequiredShardMeta, WriterInfo


def _attempt(
    db_id: int, *, db_url: str | None = None, row_count: int = 10
) -> ShardAttemptResult:
    return ShardAttemptResult(
        db_id=db_id,
        db_url=db_url or f"s3://b/p/db={db_id:05d}/attempt=00",
        attempt=0,
        row_count=row_count,
        min_key=1 if row_count > 0 else None,
        max_key=row_count if row_count > 0 else None,
        checkpoint_id="ckpt" if row_count > 0 else None,
        writer_info=WriterInfo(),
    )


class TestSelectWinnersWithEmptyShards:
    def test_missing_db_ids_omitted_from_winners(self) -> None:
        """select_winners does not require all db_ids; missing ones are just absent."""
        attempts = [_attempt(0), _attempt(2)]
        winners, num_attempts, all_urls = select_winners(attempts, num_dbs=4)
        assert len(winners) == 2
        assert {w.db_id for w in winners} == {0, 2}
        assert num_attempts == 2

    def test_db_url_none_filtered_from_winners(self) -> None:
        """Attempts with db_url=None (empty shards) are not included in winners."""
        attempts = [
            _attempt(0),
            ShardAttemptResult(
                db_id=1,
                db_url=None,
                attempt=0,
                row_count=0,
                min_key=None,
                max_key=None,
                checkpoint_id=None,
                writer_info=WriterInfo(),
            ),
        ]
        winners, num_attempts, all_urls = select_winners(attempts, num_dbs=2)
        assert len(winners) == 1
        assert winners[0].db_id == 0
        assert num_attempts == 2
        # None URLs are excluded from all_attempt_urls
        assert len(all_urls) == 1

    def test_all_empty_produces_no_winners(self) -> None:
        """When all shards have db_url=None, winners is empty."""
        attempts = [
            ShardAttemptResult(
                db_id=i,
                db_url=None,
                attempt=0,
                row_count=0,
                min_key=None,
                max_key=None,
                checkpoint_id=None,
                writer_info=WriterInfo(),
            )
            for i in range(3)
        ]
        winners, num_attempts, all_urls = select_winners(attempts, num_dbs=3)
        assert len(winners) == 0
        assert num_attempts == 3
        assert all_urls == []


class TestCleanupLosersWithNoneDbUrl:
    @patch("shardyfusion._writer_core.delete_prefix")
    @patch("shardyfusion._writer_core.create_s3_client")
    def test_none_winner_urls_excluded(
        self, mock_create: MagicMock, mock_delete: MagicMock
    ) -> None:
        """cleanup_losers skips None db_urls in winner set."""
        mock_delete.return_value = 3
        mock_create.return_value = MagicMock()

        winners = [
            RequiredShardMeta(
                db_id=0, db_url="s3://b/p/db=0/attempt=00", attempt=0, row_count=5
            ),
            RequiredShardMeta(db_id=1, db_url=None, attempt=0, row_count=0),
        ]
        all_urls = ["s3://b/p/db=0/attempt=00", "s3://b/p/db=0/attempt=01"]

        deleted = cleanup_losers(all_urls, winners)
        assert deleted == 3
        mock_delete.assert_called_once()


class TestCleanupStaleAttemptsWithNoneDbUrl:
    def test_skips_none_db_url_shards(self) -> None:
        """cleanup_stale_attempts skips shards with db_url=None."""
        manifest = MagicMock()
        manifest.required_build.run_id = "run-abc"
        manifest.required_build.s3_prefix = "s3://bucket/prefix"
        manifest.required_build.shard_prefix = "shards"
        manifest.required_build.db_path_template = "db={db_id:05d}"

        real_shard = MagicMock()
        real_shard.db_id = 0
        real_shard.db_url = (
            "s3://bucket/prefix/shards/run_id=run-abc/db=00000/attempt=00"
        )

        empty_shard = MagicMock()
        empty_shard.db_id = 1
        empty_shard.db_url = None

        manifest.shards = [real_shard, empty_shard]

        with patch("shardyfusion._writer_core.list_prefixes", return_value=[]):
            actions = cleanup_stale_attempts(manifest, s3_client=MagicMock())

        # Should only scan shard 0, not shard 1 (which has db_url=None)
        assert actions == []
