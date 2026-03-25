# 2026-03-25 Consistent Writer Retry

- Status: `implemented`
- Date: `2026-03-25`
- Baseline repo commit before this change series: `0097fdfbad1099c3ad494532b423132e71ac8b49`
- Baseline commit summary: `Remove duplicate code coverage limit from pyproject.toml`
- Initial implementation commits:
  - `37fe4f7` `feat: add retryable single-db writer attempts`
  - `1b922dd` `feat: add retryable python parallel spool writer`
  - `3ab8255` `docs: describe consistent writer retry behavior`

## Summary

This change series made retry behavior more consistent across writer backends and added built-in retry support to the Python parallel writer.

Two core updates were implemented:

1. Writers that perform built-in retries now write each retry attempt to a fresh `attempt=NN` path.
2. The Python parallel writer now supports optional per-shard retry using durable file-backed spool files.

The goal was not to make shard materialization exactly-once. The goal was to make retries safe, deterministic at publish time, and easier to reason about across backends.

## Previous State

Before this change series:

- Dask and Ray sharded writers already retried shard writes into fresh `attempt=NN` paths.
- Spark sharded writer already used attempt-isolated paths via Spark task attempts.
- `write_single_db()` paths did not use built-in retry and effectively wrote only `attempt=00`.
- Python parallel writer had no built-in retry and relied only on shared memory for parent-to-worker transfer.

That meant retry semantics were inconsistent across backends, and the Python parallel writer had no durable replay source if a worker died or a shard write failed late in the attempt.

## What Was Implemented

### Shared Retry Semantics Across Retrying Writers

After this change:

- Dask and Ray sharded writers continue to retry into fresh attempt paths.
- Spark sharded writer remains Spark-attempt-driven and continues to use attempt-isolated paths.
- Spark, Dask, and Ray `write_single_db()` now use built-in retry and advance through `attempt=00`, `attempt=01`, and so on.
- Python parallel writer now optionally retries per shard when `WriteConfig.shard_retry` is configured.

### Python Parallel Writer Retry

The Python parallel writer now has two transport modes.

#### Default Mode

Used when `shard_retry is None`.

- Parent serializes chunks into shared memory.
- Workers decode shared-memory chunks and write them directly.
- No shard replay support is provided.

This preserves the previous fast path and avoids unnecessary disk usage when retry is not requested.

#### Retry-Enabled Mode

Used when `shard_retry` is configured.

- Parent routes each record to a shard.
- Parent batches records per shard and serializes them into a durable spool file.
- Parent records chunk metadata: file path, offset, size, row count.
- Parent sends only chunk metadata over `multiprocessing.Queue`.
- Worker reconstructs batches by reading those file ranges.
- On retryable failure, the parent respawns the worker and replays prior chunks for that shard into a new attempt path.

## Why These Changes Were Made

### Consistency

Retry behavior should mean the same thing across backends:

- a retryable failure should not overwrite the previous attempt
- each retry should produce a distinct attempt path
- cleanup and winner selection should work the same way across implementations

Extending the existing attempt model to single-db writers aligns them with the existing sharded Dask/Ray behavior.

### Safety

Retrying into the same output path is unsafe. It makes it hard to distinguish:

- a clean retry
- a partial rewrite
- a mixed state after an interrupted write

Fresh attempt IDs avoid path reuse and preserve the current winner-selection model.

### Replayability for Python Parallel Mode

Shared memory alone cannot provide durable replay because it is:

- transient
- reclaimed as workers progress
- lost if the worker dies after consuming the chunk

The file-backed spool design solves that by making shard input durable until the parent accepts a successful result.

## Design Details

### Shared Retry Orchestration

Retry attempt orchestration is centralized in the shared shard writer logic.

Responsibilities of the shared retry loop:

- assign attempt numbers
- derive `db_url` and `local_dir` for each attempt
- preserve `all_attempt_urls`
- apply exponential backoff
- emit retry metrics
- stop immediately on non-retryable failures

This keeps attempt numbering and retry bookkeeping uniform across Dask, Ray, and single-db writers.

### Single-DB Writers

`write_single_db()` in Spark, Dask, and Ray now wraps the write phase in the shared retry loop.

Important scope choice:

- The expensive distributed sort or preparation phase is still outside the retry loop.
- Retry covers the adapter write attempt itself.
- Each retry reopens a fresh adapter against a new `attempt=NN` path.

This keeps retry focused on transient write failures and avoids repeating more work than necessary.

### Python Parallel Writer

The retry-enabled Python parallel mode persists shard input locally and replays it by shard attempt.

Important protocol details in the current implementation:

- A chunk is appended to the replay history only after it has been successfully queued to the current worker attempt.
- Worker replay during respawn uses the same attempt-aware queue helper as live dispatch, so replay does not block forever on a dead worker queue.
- Retry result collection timeouts include the configured retry backoff budget instead of using only a fixed per-shard deadline.

Those details were tightened after review because the first version had three problems:

- a chunk could be marked replayable before it had actually been queued, which could duplicate rows after a respawn
- replay used blocking queue writes with no liveness polling, which could hang indefinitely
- result collection used a fixed timeout that could expire before legitimate retry backoff windows elapsed

## Failure Handling Model

The implemented design targets safe retry, not exact-once attempt creation.

That distinction matters for the crash window after:

- `flush()`
- `checkpoint()`
- but before the worker reports success back to the parent

If the worker crashes in that window:

- the shard attempt may already exist and be valid
- the parent may still retry because success was not observed

The design accepts that outcome. The retry writes to a fresh attempt path, and manifest winner selection keeps publication deterministic.

This means duplicate successful attempts are still possible in rare crash cases, but mixed-path corruption is avoided.

## Implementation Areas

Primary implementation areas:

- `shardyfusion/_shard_writer.py`
  - added shared retry attempt orchestration
- `shardyfusion/writer/dask/single_db_writer.py`
  - added single-db retry support
- `shardyfusion/writer/ray/single_db_writer.py`
  - added single-db retry support
- `shardyfusion/writer/spark/single_db_writer.py`
  - added single-db retry support
- `shardyfusion/writer/python/_parallel_writer.py`
  - added retry-enabled file-spool transport
  - added worker respawn and replay logic
  - preserved shared-memory mode as the default fast path
  - later tightened queue publication ordering, replay liveness, and timeout sizing

Additional updates:

- `shardyfusion/config.py`
  - broadened `shard_retry` documentation to match actual behavior
- writer and observability docs
  - updated retry descriptions and backend coverage
- unit tests
  - added retry-specific coverage for single-db writers
  - added retry and replay coverage for Python parallel mode

## Pros

### Consistent Attempt Semantics

All built-in retry paths now use fresh attempt IDs. That makes output paths, cleanup, and debugging much more predictable.

### Safe Retry Paths

Retries no longer risk writing over the same attempt directory in single-db writers or Python parallel retry mode.

### Better Python Parallel Reliability

The Python parallel writer can now recover from:

- retryable adapter failures
- unexpected worker exits

without requiring the caller to rerun the entire write from scratch.

### Minimal Public API Change

No new public retry config was added. The feature uses the existing `WriteConfig.shard_retry` switch.

### Preserves Fast Path

Users who do not opt into retry keep the original shared-memory Python parallel behavior and do not pay the disk overhead.

## Cons

### Higher Disk Usage in Retry-Enabled Python Parallel Mode

Durable replay requires storing shard input on disk until the parent accepts success. Disk usage grows with the serialized size of unfinished shard inputs.

### More Complex Parent/Worker Protocol

The retry-enabled Python parallel mode is materially more complex than the original shared-memory-only path:

- spool file management
- chunk metadata tracking
- worker respawn
- replay ordering
- stale result filtering
- attempt-aware queue handoff

That increases maintenance cost.

### Orphan Attempts Are Still Possible

Without a separate success marker or transactional commit point, a worker can crash after checkpointing but before reporting success. In that case, a valid earlier attempt may remain orphaned after the retry succeeds.

This is acceptable for correctness because winner selection stays deterministic, but it is not perfect cleanup behavior.

### Retry Scope Is Still Selective

This change does not add retry everywhere:

- Spark sharded writes still rely on Spark task retry and speculation.
- Python single-process mode still has no built-in retry.

That is intentional, but it means retry behavior is still backend- and mode-dependent.

## Why This Solution Was Chosen Over Alternatives

### Reusing Shared Memory for Retry

Rejected because shared memory is not durable and cannot survive worker death in a replayable way.

### Retrying Only the Failing Batch

Rejected because a shard attempt can fail after many successful `write_batch()` calls, including during `flush()` or `checkpoint()`. Safe retry requires replaying the whole shard attempt into a fresh path, not just replaying the last batch.

### Success Markers

Not implemented because they would introduce a commit convention that other writer paths do not currently use. The design stays aligned with the existing winner-selection and loser-cleanup model.

## Resulting Behavior

With this change series, the retry story is now:

- Dask and Ray sharded writes: built-in per-shard retry, fresh attempt paths
- Spark sharded writes: Spark-managed attempts, fresh attempt paths
- Spark, Dask, and Ray single-db writes: built-in whole-db retry, fresh attempt paths
- Python parallel writes: optional per-shard retry via durable spool files, fresh attempt paths
- Python single-process writes: no built-in retry

This is a meaningful improvement in consistency and reliability while keeping the existing attempt-based publish model intact.
