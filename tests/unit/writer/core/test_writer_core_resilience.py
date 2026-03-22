"""Tests for _writer_core resilience changes (H4, C1, M1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from shardyfusion._writer_core import ShardAttemptResult, cleanup_losers, select_winners
from shardyfusion.manifest import RequiredShardMeta, WriterInfo


def _attempt(db_id: int, attempt: int = 0) -> ShardAttemptResult:
    return ShardAttemptResult(
        db_id=db_id,
        db_url=f"s3://b/p/db={db_id:05d}/attempt={attempt:02d}",
        attempt=attempt,
        row_count=10,
        min_key=1,
        max_key=10,
        checkpoint_id=None,
        writer_info=WriterInfo(task_attempt_id=attempt),
    )


def _winner(db_id: int, attempt: int = 0) -> RequiredShardMeta:
    return RequiredShardMeta(
        db_id=db_id,
        db_url=f"s3://b/p/db={db_id:05d}/attempt={attempt:02d}",
        attempt=attempt,
        row_count=10,
        min_key=1,
        max_key=10,
        checkpoint_id=None,
        writer_info=WriterInfo(task_attempt_id=attempt),
    )


class TestSelectWinnersIterable:
    """H4: select_winners accepts any Iterable, not just list."""

    def test_accepts_generator(self) -> None:
        """select_winners works with a generator (not just list)."""

        def gen():
            yield _attempt(0)
            yield _attempt(1)

        winners, num_attempts, urls = select_winners(gen(), num_dbs=2)
        assert len(winners) == 2
        assert num_attempts == 2
        assert len(urls) == 2

    def test_accepts_iterator(self) -> None:
        """select_winners works with an iterator."""
        attempts = [_attempt(0), _attempt(1, attempt=0), _attempt(1, attempt=1)]
        winners, num_attempts, urls = select_winners(iter(attempts), num_dbs=2)
        assert len(winners) == 2
        assert num_attempts == 3
        assert len(urls) == 3

    def test_returns_all_attempt_urls(self) -> None:
        """All attempt URLs are returned, including losers."""
        attempts = [
            _attempt(0, attempt=0),
            _attempt(0, attempt=1),  # loser
            _attempt(1, attempt=0),
        ]
        winners, num_attempts, urls = select_winners(attempts, num_dbs=2)
        assert num_attempts == 3
        assert len(urls) == 3
        # Winner URLs should be a subset of all URLs
        winner_urls = {w.db_url for w in winners}
        assert winner_urls.issubset(set(urls))

    def test_returns_embedded_attempt_urls_from_retry_result(self) -> None:
        """Retry metadata carried on the winner contributes loser URLs."""
        attempts = [
            ShardAttemptResult(
                db_id=0,
                db_url="s3://b/p/db=00000/attempt=01",
                attempt=1,
                row_count=10,
                min_key=1,
                max_key=10,
                checkpoint_id=None,
                writer_info=WriterInfo(task_attempt_id=1),
                all_attempt_urls=(
                    "s3://b/p/db=00000/attempt=00",
                    "s3://b/p/db=00000/attempt=01",
                ),
            )
        ]

        winners, num_attempts, urls = select_winners(attempts, num_dbs=1)
        assert len(winners) == 1
        assert num_attempts == 1
        assert urls == [
            "s3://b/p/db=00000/attempt=00",
            "s3://b/p/db=00000/attempt=01",
        ]

    def test_num_attempts_matches_count(self) -> None:
        """num_attempts accurately reflects total attempts processed."""
        attempts = [_attempt(i) for i in range(5)]
        _, num_attempts, _ = select_winners(attempts, num_dbs=5)
        assert num_attempts == 5


class TestCleanupLosers:
    """cleanup_losers deletes non-winning attempt paths (best-effort)."""

    @patch("shardyfusion._writer_core.delete_prefix")
    @patch("shardyfusion._writer_core.create_s3_client")
    def test_losers_deleted_winners_preserved(
        self, mock_create_client: MagicMock, mock_delete: MagicMock
    ) -> None:
        """Only loser URLs are deleted; winner URLs are skipped."""
        mock_delete.return_value = 5
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        winners = [_winner(0, attempt=0), _winner(1, attempt=0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",  # winner
            "s3://b/p/db=00000/attempt=01",  # loser
            "s3://b/p/db=00001/attempt=00",  # winner
            "s3://b/p/db=00001/attempt=01",  # loser
        ]

        deleted = cleanup_losers(all_urls, winners)
        assert deleted == 10  # 5 objects x 2 losers

        # Only loser URLs should be passed to delete_prefix
        deleted_urls = [call.args[0] for call in mock_delete.call_args_list]
        assert "s3://b/p/db=00000/attempt=01" in deleted_urls
        assert "s3://b/p/db=00001/attempt=01" in deleted_urls
        assert "s3://b/p/db=00000/attempt=00" not in deleted_urls
        assert "s3://b/p/db=00001/attempt=00" not in deleted_urls

    @patch("shardyfusion._writer_core.delete_prefix")
    @patch("shardyfusion._writer_core.create_s3_client")
    def test_returns_total_deleted_count(
        self, mock_create_client: MagicMock, mock_delete: MagicMock
    ) -> None:
        """Returns the total number of S3 objects deleted."""
        mock_delete.side_effect = [3, 7]
        mock_create_client.return_value = MagicMock()

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",  # winner
            "s3://b/p/db=00000/attempt=01",  # loser (3 objects)
            "s3://b/p/db=00000/attempt=02",  # loser (7 objects)
        ]

        deleted = cleanup_losers(all_urls, winners)
        assert deleted == 10

    @patch("shardyfusion._writer_core.delete_prefix")
    @patch("shardyfusion._writer_core.create_s3_client")
    def test_no_losers(
        self, mock_create_client: MagicMock, mock_delete: MagicMock
    ) -> None:
        """When all URLs are winners, nothing is deleted."""
        mock_create_client.return_value = MagicMock()

        winners = [_winner(0), _winner(1)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",
            "s3://b/p/db=00001/attempt=00",
        ]

        deleted = cleanup_losers(all_urls, winners)
        assert deleted == 0
        mock_delete.assert_not_called()

    @patch("shardyfusion._writer_core.delete_prefix")
    @patch("shardyfusion._writer_core.create_s3_client")
    def test_best_effort_s3_errors_not_raised(
        self, mock_create_client: MagicMock, mock_delete: MagicMock
    ) -> None:
        """S3 errors in delete_prefix are handled internally (returns 0 for that URL)."""
        # delete_prefix catches exceptions internally and returns partial count
        mock_delete.return_value = 0
        mock_create_client.return_value = MagicMock()

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",  # winner
            "s3://b/p/db=00000/attempt=01",  # loser - delete returns 0
        ]

        # Should not raise even when delete_prefix returns 0
        deleted = cleanup_losers(all_urls, winners)
        assert deleted == 0

    @patch("shardyfusion._writer_core.delete_prefix")
    @patch("shardyfusion._writer_core.create_s3_client")
    def test_metrics_collector_passed_through(
        self, mock_create_client: MagicMock, mock_delete: MagicMock
    ) -> None:
        """metrics_collector is forwarded to delete_prefix."""
        mock_delete.return_value = 1
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mc = MagicMock()

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",
            "s3://b/p/db=00000/attempt=01",
        ]

        cleanup_losers(all_urls, winners, metrics_collector=mc)
        mock_delete.assert_called_once_with(
            "s3://b/p/db=00000/attempt=01",
            s3_client=mock_client,
            metrics_collector=mc,
        )
