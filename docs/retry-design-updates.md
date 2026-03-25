# Retry Design Updates

## Summary

This change set makes retry behavior more consistent across writer backends and adds built-in retry support to the Python parallel writer.

Two design updates were implemented:

1. Writers that perform built-in retries now write each retry attempt to a fresh `attempt=NN` path.
2. The Python parallel writer now supports optional per-shard retry using durable file-backed spool files.

The goal was not to make writes exactly-once at the shard-materialization level. The goal was to make retries safe, deterministic at publish time, and consistent across backends.

## What Changed

### 1. Shared Retry Semantics Across Retrying Writers

Before this change:

- Dask and Ray sharded writers already retried shard writes into fresh `attempt=NN` paths.
- Spark sharded writer already had attempt-isolated paths through Spark task attempts.
- `write_single_db()` paths did not use built-in retry and effectively wrote only `attempt=00`.
- Python parallel writer had no built-in retry.

After this change:

- Dask and Ray sharded writers continue to retry into fresh attempt paths.
- Spark sharded writer remains Spark-attempt-driven and continues to use attempt-isolated paths.
- Spark, Dask, and Ray `write_single_db()` now use built-in retry and advance through `attempt=00`, `attempt=01`, and so on.
- Python parallel writer now optionally retries per shard when `WriteConfig.shard_retry` is configured.

### 2. Python Parallel Writer Retry

Before this change, the Python parallel writer used only shared memory for parent-to-worker transfer. That was efficient, but it was not replayable:

- If adapter operations failed late in the shard attempt, there was no durable replay source.
- If a worker process died after consuming shared-memory chunks, the parent could not reconstruct the shard input.

After this change:

- Default Python parallel mode still uses shared memory when `shard_retry` is `None`.
- Retry-enabled Python parallel mode switches to a durable per-shard spool file.
- The parent appends serialized chunks to disk and sends only chunk metadata to workers.
- If a worker fails or exits unexpectedly, the parent respawns it and replays that shard from the spool file into a fresh attempt path.

## Why These Changes Were Made

## Consistency

Retry behavior should mean the same thing across backends:

- a retryable failure should not overwrite the previous attempt
- each retry should produce a distinct attempt path
- cleanup and winner selection should work the same way across implementations

Extending the existing attempt model to single-db writers makes the system easier to reason about and aligns them with the existing sharded Dask/Ray behavior.

## Safety

Retrying into the same output path is unsafe. It makes it hard to distinguish:

- a clean retry
- a partial rewrite
- a mixed state after an interrupted write

Fresh attempt IDs avoid path reuse and preserve the current winner-selection model.

## Replayability for Python Parallel Mode

The Python parallel writer needed a durable replay source to support retry correctly.

Shared memory alone cannot provide that because it is:

- transient
- reclaimed as workers progress
- lost if the worker dies after consuming the chunk

The file-backed spool design solves that by making shard input durable until the parent accepts a successful result.

## Design Details

### Shared Retry Orchestration

Retry attempt orchestration is now centralized in the shared shard writer logic.

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

- The expensive distributed sort/preparation phase is still outside the retry loop.
- Retry covers the adapter write attempt itself.
- Each retry reopens a fresh adapter against a new `attempt=NN` path.

This keeps retry focused on transient write failures and avoids repeating more work than necessary.

### Python Parallel Writer

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
- On retryable failure, the parent respawns the worker and replays all prior chunks for that shard into a new attempt path.

### Failure Handling Model

The implemented design targets safe retry, not exact-once attempt creation.

That distinction matters for the crash window after:

- `flush()`
- `checkpoint()`
- but before the worker reports success back to the parent

If the worker crashes in that window:

- the shard attempt may already exist and be valid
- the parent may still retry because success was not observed

The implemented design accepts that outcome. The retry writes to a fresh attempt path, and manifest winner selection keeps publication deterministic.

This means duplicate successful attempts are possible in rare crash cases, but mixed-path corruption is avoided.

## Implementation Notes

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
  - added worker respawn/replay logic
  - preserved shared-memory mode as the default fast path

Additional updates:

- `shardyfusion/config.py`
  - broadened `shard_retry` documentation to match actual behavior
- writer and observability docs
  - updated retry descriptions and backend coverage
- unit tests
  - added retry-specific coverage for single-db writers
  - added retry/replay coverage for Python parallel mode

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

That increases maintenance cost.

### Orphan Attempts Are Still Possible

Without a separate success marker or transactional commit point, a worker can crash after checkpointing but before reporting success. In that case, a valid earlier attempt may remain orphaned after the retry succeeds.

This is acceptable for correctness because winner selection stays deterministic, but it is not perfect cleanup behavior.

### Retry Scope Is Still Selective

This change does not add retry everywhere:

- Spark sharded writes still rely on Spark task retry/speculation.
- Python single-process mode still has no built-in retry.

That is intentional, but it means retry behavior is still backend- and mode-dependent.

## Why This Solution Was Chosen Over Alternatives

### Reusing Shared Memory for Retry

Rejected because shared memory is not durable and cannot survive worker death in a replayable way.

### Retrying Only the Failing Batch

Rejected because a shard attempt can fail after many successful `write_batch()` calls, including during `flush()` or `checkpoint()`. Safe retry requires replaying the whole shard attempt into a fresh path, not just replaying the last batch.

### Success Markers

Not implemented because they would introduce a commit convention that other writer paths do not currently use. The implemented design stays aligned with the existing winner-selection and loser-cleanup model.

## Resulting Behavior

With this change, the retry story is now:

- Dask/Ray sharded writes: built-in per-shard retry, fresh attempt paths
- Spark sharded writes: Spark-managed attempts, fresh attempt paths
- Spark/Dask/Ray single-db writes: built-in whole-db retry, fresh attempt paths
- Python parallel writes: optional per-shard retry via durable spool files, fresh attempt paths
- Python single-process writes: no built-in retry

This is a meaningful improvement in consistency and reliability, while keeping the existing attempt-based publish model intact.
