from __future__ import annotations

from slatedb_spark_sharded._writer_core import ShardAttemptResult, select_winners


def test_winner_selection_is_deterministic() -> None:
    attempts = [
        ShardAttemptResult(
            db_id=0,
            db_url="s3://b/p/db=0/attempt=01",
            attempt=1,
            row_count=10,
            min_key=1,
            max_key=10,
            checkpoint_id=None,
            writer_info={"task_attempt_id": 2},
        ),
        ShardAttemptResult(
            db_id=0,
            db_url="s3://b/p/db=0/attempt=00",
            attempt=0,
            row_count=10,
            min_key=1,
            max_key=10,
            checkpoint_id=None,
            writer_info={"task_attempt_id": 5},
        ),
        ShardAttemptResult(
            db_id=1,
            db_url="s3://b/p/db=1/attempt=00",
            attempt=0,
            row_count=7,
            min_key=11,
            max_key=17,
            checkpoint_id=None,
            writer_info={"task_attempt_id": 1},
        ),
    ]

    winners = select_winners(attempts, num_dbs=2)
    assert [w.db_id for w in winners] == [0, 1]
    assert winners[0].attempt == 0
    assert winners[0].db_url.endswith("attempt=00")
