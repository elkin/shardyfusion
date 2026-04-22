# Retry and cleanup

Writers may attempt the same shard multiple times — explicitly via `shard_retry`, or implicitly via engine-level task restarts (Spark speculative execution, Dask retries, Ray task restarts, Python writer's `shard_retry`). The system is designed so that **multiple successful attempts at the same shard are normal**, and exactly one is selected as the winner per run.

## Engine-level retry

| Engine | Retry mechanism | Notes |
|---|---|---|
| Python | `shard_retry: int` | Spawned subprocess per shard; on failure, a fresh subprocess retries up to N times. |
| Spark | Task speculation + `spark.task.maxFailures` | Driven by Spark scheduler; writer is idempotent at the task level. |
| Dask | `dask.config retries` | Per-task retry. |
| Ray | `max_retries` on `@ray.remote` | Per-task retry. |

In all cases, each retry produces a fresh `ShardAttemptResult` (`_writer_core.py:73`) with a distinct `task_attempt_id` (or `db_url` if no attempt ID is available).

## Winner selection (the invariant)

After all attempts complete, `select_winners` (`_writer_core.py:296`) uses the comparator from `_winner_sort_key` (`_writer_core.py:363`):

```
(attempt, task_attempt_id_or_INT64_MAX, db_url_or_empty)  # ascending
```

The smallest tuple wins. This is **identical across all four engines**, which is what makes the writer behavior consistent. See [`history/design-decisions/adr-004-consistent-writer-retry.md`](../history/design-decisions/adr-004-consistent-writer-retry.md).

If any shard `db_id` in `[0, num_dbs)` has zero successful attempts, the build aborts with `ShardCoverageError` (`errors.py:47`).

## Three cleanup phases

After publish, the writer runs cleanup in this order:

### 1. `cleanup_losers` (`_writer_core.py:602`)

Deletes all losing attempts from the *current* run's published shards. The winner has been recorded in the manifest; everything else under `data/<run_id>/shard=NNNN/attempt=*` other than the winning attempt is removed.

This cleanup is best-effort: a failure here does not invalidate the published manifest. See [`history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md`](../history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md) for outstanding work to make this more robust under partial failure.

### 2. `cleanup_stale_attempts` (`_writer_core.py:665`)

Deletes uncommitted attempts from prior runs older than `stale_attempt_age`. This is the recovery mechanism for runs that crashed before publishing — their data lives on under `data/<run_id>/` and would otherwise leak.

Gated by the run registry: only attempts belonging to runs in `FAILED` status (or runs older than the threshold and not in `SUCCEEDED`) are eligible. See [`run-registry.md`](run-registry.md).

### 3. `cleanup_old_runs` (`_writer_core.py:725`)

Deletes runs older than `keep_runs` policy — both their data and their manifest objects. This bounds storage growth at the cost of giving up the ability to roll back to those snapshots.

Operates on `SUCCEEDED` runs only (in registry terms). The current `_CURRENT` run is never deleted regardless of age.

## CleanupAction

`CleanupAction` (`_writer_core.py:655`) is the data class describing one planned cleanup operation. The cleanup functions return lists of these so callers can log/inspect them (used heavily in tests and by the `cleanup` CLI subcommand — see [`use-cases/operate-cli.md`](../use-cases/operate-cli.md)).

## Failure modes during cleanup

| Failure | Effect |
|---|---|
| S3 transient error during loser cleanup | Loser data leaks; next `cleanup_stale_attempts` pass collects it. |
| Run registry write fails after publish | Manifest is published but registry shows `RUNNING`; future cleanup will refuse to touch it; next writer succeeds and overwrites `_CURRENT`. |
| Network partition mid-cleanup | Partial deletion; remaining objects collected by next stale-attempts pass. |

The system tolerates cleanup failures by design: nothing in the read path depends on cleanup having completed. Storage cost is the only consequence.

## See also

- [`writer-core.md`](writer-core.md) — `select_winners`, all three cleanup functions.
- [`run-registry.md`](run-registry.md) — gating semantics.
- [`history/design-decisions/adr-004-consistent-writer-retry.md`](../history/design-decisions/adr-004-consistent-writer-retry.md) — why one comparator across all engines.
- [`history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md`](../history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md) — open work.
