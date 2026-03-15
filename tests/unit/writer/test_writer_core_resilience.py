"""Tests for _writer_core resilience changes (H4, C1, M1)."""

from __future__ import annotations

from shardyfusion._writer_core import ShardAttemptResult, select_winners
from shardyfusion.manifest import WriterInfo


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

    def test_num_attempts_matches_count(self) -> None:
        """num_attempts accurately reflects total attempts processed."""
        attempts = [_attempt(i) for i in range(5)]
        _, num_attempts, _ = select_winners(attempts, num_dbs=5)
        assert num_attempts == 5
