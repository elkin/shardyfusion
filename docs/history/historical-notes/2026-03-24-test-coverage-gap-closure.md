# 2026-03-24 Test Coverage Gap Closure

- Status: `implemented`
- Date: `2026-03-24`
- Baseline repo commit before this note's documented state: `a30f01a93d0f9e2136033f2498618dab236a01a6`
- Baseline commit summary: `test: update async reader tests to use timedelta for delays`
- This coverage note was recorded in commit:
  - `c1fc90f714c7cd13d0daf9a077f1394a5adaf69c` `docs: reflect timedelta and renamed parameters in documentation`

## Summary

This note records a focused test-coverage closure pass that added 73 tests across 12 files, targeting states and error paths that had previously been uncovered.

The emphasis was not broad percentage improvement. It was concentrated coverage in areas where missing tests could hide correctness, lifecycle, or error-propagation bugs.

## What Was Added

73 new tests across 12 files, organized by severity.

## Critical — Data Corruption / Contract Violations

### `test_publish_error_paths.py` (7 tests)

`shardyfusion/_writer_core.py:publish_to_store()`

The two-phase publish (manifest PUT → `_CURRENT` pointer PUT) had its entire error-handling path marked `# pragma: no cover`. These tests cover:

- `PublishCurrentError` raised by `store.publish()` with a `manifest_ref` — triggers the retry loop that calls `store.set_current()` directly
- Retry succeeds on the 2nd attempt (first `set_current` fails, second succeeds)
- All 3 `set_current` retries exhausted — raises `PublishManifestError`
- `PublishCurrentError` with `manifest_ref=None` — re-raised immediately (no recovery possible)
- Generic exception during `store.publish()` — wrapped in `PublishManifestError`
- Happy-path: `MANIFEST_PUBLISHED` metric emitted on success

### `test_cel_routing_reader.py` (6 tests)

`shardyfusion/reader/reader.py` and `concurrent_reader.py`

All three reader types accept `routing_context` for CEL sharding, but no reader test had ever used CEL. These tests cover:

- `ShardedReader.get()` and `route_key()` with a CEL key-only manifest (`key % 4`)
- `ConcurrentShardedReader.route_key()` with CEL
- `ShardedReader.get()` / `multi_get()` / `route_key()` with multi-column CEL + boundaries (`region` → bisect_right shard selection)
- Calling `route_key()` without `routing_context` on a multi-column CEL manifest — raises `ValueError` because the router cannot function without the extra columns

### `test_adapter_lifecycle.py` (5 tests)

`shardyfusion/_shard_writer.py:write_shard_core()`

No test had simulated an adapter that fails mid-write. These tests cover:

- `adapter.flush()` raises `OSError` — `write_shard_core` wraps it in `ShardWriteError`
- `adapter.checkpoint()` returns `None` — shard result is written but `checkpoint_id` is `None`
- `adapter.__exit__()` raises — error propagates out
- Factory `__call__` raises `ConnectionError` — propagates, possibly wrapped
- Zero rows written — `row_count == 0`, no crash

## High — Reliability / Correctness

### `test_router_transitions.py` (4 tests)

`shardyfusion/routing.py`, `shardyfusion/reader/reader.py`

When a reader refreshes, `num_dbs` or the shard layout can change. Tests cover:

- `num_dbs` increases (2→4) on refresh — `snapshot_info().num_dbs` reflects the new count
- `num_dbs` decreases (4→2) on refresh — `route_key()` uses the new modulus
- Manifest with missing `db_id`s, such as only shards 0 and 2 out of 3 — `SnapshotRouter` pads the gap with a synthetic null shard; `shard_details()` shows `row_count=0` for shard 1
- Refresh from sparse to full manifest — previously-null shard now has real data

### `test_multi_get_failures.py` (5 tests)

`shardyfusion/reader/reader.py` and `concurrent_reader.py:multi_get()`

When `multi_get` fans out to multiple shards and one fails, the error contract was unspecified. Tests cover:

- `ShardedReader`: one shard fails — exception raised, not silent partial result
- `ConcurrentShardedReader`: one shard fails — exception raised
- All shards fail — `SlateDbApiError` raised
- All keys on the same failing shard — original exception preserved
- `ConcurrentShardedReader` with `max_workers=2` (executor path) — `SlateDbApiError` propagated from the failed future

### `test_async_deferred_cleanup.py` (4 tests)

`shardyfusion/reader/async_reader.py:_release_state()`

When `refresh()` retires old state, cleanup is deferred via `loop.create_task()` until borrows drop to zero. Tests cover:

- Borrow held during refresh, then released — deferred cleanup task is scheduled
- `close()` called while a borrow handle is still outstanding — releasing the handle afterwards does not crash
- Two rapid refreshes — reader lands on the correct final manifest
- `get()` after refresh — returns data from the new state

### `test_value_spec.py` (14 tests)

`shardyfusion/serde.py:ValueSpec`

`ValueSpec` was only tested indirectly via integration tests. These are dedicated unit tests:

- `binary_col`: `None` value → `b""`; `bytearray` → converted to `bytes`; `str` → UTF-8; unsupported type → `ConfigValidationError`
- `json_cols`: selected columns only; all columns; `None` value in column; nested dict; Unicode round-trips; missing column → `None` in output; `description` field
- `callable_encoder`: basic callable; encoder that raises; `description` uses `__name__`

## Medium — Robustness

### `test_ordering.py` (10 tests)

`shardyfusion/ordering.py:compare_ordered()`

Previously only exercised indirectly through boundary validation. Direct tests cover:

- Integers and strings: less-than, equal, greater-than
- Incompatible types (`"a"` vs `1`) — raises `ValueError` with the caller's message
- `None` vs `int` — raises `ValueError`
- `bytes` and `float` comparisons

### `test_prometheus_completeness.py` (15 tests) and `test_otel_completeness.py` (6 tests)

`shardyfusion/metrics/prometheus.py`, `shardyfusion/metrics/otel.py`

Several `MetricEvent` values handled by the collectors had no tests. Added coverage for:

- Handled events: `SHARD_WRITE_COMPLETED` (counter + histogram), `BATCH_WRITTEN`, `READER_MULTI_GET` (counter + histogram), `S3_RETRY_EXHAUSTED`, `RATE_LIMITER_DENIED`
- Silently ignored events, by design and not mapped to instruments: `SHARDING_COMPLETED`, `MANIFEST_PUBLISHED`, `CURRENT_PUBLISHED`, `READER_INITIALIZED`, `READER_REFRESHED`, `READER_CLOSED`, `SHARD_WRITE_RETRIED`, `SHARD_WRITE_RETRY_EXHAUSTED` — verified they do not raise

### `test_batch_errors.py` (10 tests)

`shardyfusion/cli/batch.py`

Error-path coverage for the batch script runner:

- `commands` key is a string or int, not a list — `ValueError`
- Empty `commands: []` — parses successfully, runs nothing
- Command dict missing `op` field — reported as an error
- `get` without `key`, `multiget` with empty or non-list `keys`, `route` without `key` — each reported as an error
- Reader raises during `get` — caught, reported as error, not a crash
- `on_error: continue` — all remaining commands still run after an error
- `on_error: stop` — execution halts after the first error; exactly one output line emitted

### `test_python_writer_edge_cases.py` (3 tests)

`shardyfusion/writer/python/writer.py:_write_parallel()`

Edge cases for parallel mode not covered by the existing crash test:

- Empty record list — all shard results have `row_count == 0`
- Single shard (`num_dbs=1`) — one result with all rows
- 4 shards, 100 records — total row count across all shards equals 100
