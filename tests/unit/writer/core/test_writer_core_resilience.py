"""Tests for _writer_core resilience changes (H4, C1, M1)."""

from __future__ import annotations

from shardyfusion._writer_core import ShardAttemptResult, cleanup_losers, select_winners
from shardyfusion.manifest import RequiredShardMeta, WriterInfo
from shardyfusion.storage import MemoryBackend


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
        db_bytes=0,
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

    def test_losers_deleted_winners_preserved(self) -> None:
        """Only loser URLs are deleted; winner URLs are skipped."""
        backend = MemoryBackend()
        # Pre-populate with objects
        for url in [
            "s3://b/p/db=00000/attempt=00/obj",
            "s3://b/p/db=00000/attempt=01/obj",
            "s3://b/p/db=00001/attempt=00/obj",
            "s3://b/p/db=00001/attempt=01/obj",
        ]:
            backend.put(url, b"x", "application/octet-stream")

        winners = [_winner(0, attempt=0), _winner(1, attempt=0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",  # winner
            "s3://b/p/db=00000/attempt=01",  # loser
            "s3://b/p/db=00001/attempt=00",  # winner
            "s3://b/p/db=00001/attempt=01",  # loser
        ]

        deleted = cleanup_losers(all_urls, winners, backend=backend)
        assert deleted == 2  # 2 loser prefixes, 1 object each

        # Winners preserved, losers deleted
        assert backend.try_get("s3://b/p/db=00000/attempt=00/obj") == b"x"
        assert backend.try_get("s3://b/p/db=00001/attempt=00/obj") == b"x"
        assert backend.try_get("s3://b/p/db=00000/attempt=01/obj") is None
        assert backend.try_get("s3://b/p/db=00001/attempt=01/obj") is None

    def test_returns_total_deleted_count(self) -> None:
        """Returns the total number of S3 objects deleted."""
        backend = MemoryBackend()
        backend.put(
            "s3://b/p/db=00000/attempt=00/obj", b"x", "application/octet-stream"
        )
        backend.put("s3://b/p/db=00000/attempt=01/a", b"x", "application/octet-stream")
        backend.put("s3://b/p/db=00000/attempt=01/b", b"x", "application/octet-stream")
        backend.put(
            "s3://b/p/db=00000/attempt=02/obj", b"x", "application/octet-stream"
        )

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",  # winner
            "s3://b/p/db=00000/attempt=01",  # loser (2 objects)
            "s3://b/p/db=00000/attempt=02",  # loser (1 object)
        ]

        deleted = cleanup_losers(all_urls, winners, backend=backend)
        assert deleted == 3

    def test_no_losers(self) -> None:
        """When all URLs are winners, nothing is deleted."""
        backend = MemoryBackend()
        backend.put(
            "s3://b/p/db=00000/attempt=00/obj", b"x", "application/octet-stream"
        )
        backend.put(
            "s3://b/p/db=00001/attempt=00/obj", b"x", "application/octet-stream"
        )

        winners = [_winner(0), _winner(1)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",
            "s3://b/p/db=00001/attempt=00",
        ]

        deleted = cleanup_losers(all_urls, winners, backend=backend)
        assert deleted == 0
        assert backend.get("s3://b/p/db=00000/attempt=00/obj") == b"x"
        assert backend.get("s3://b/p/db=00001/attempt=00/obj") == b"x"

    def test_best_effort_empty_prefix(self) -> None:
        """Empty prefixes return 0 deleted."""
        backend = MemoryBackend()
        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",  # winner
            "s3://b/p/db=00000/attempt=01",  # loser - but empty
        ]

        deleted = cleanup_losers(all_urls, winners, backend=backend)
        assert deleted == 0
