# Writer core

`shardyfusion/_writer_core.py` is the engine-agnostic core that every writer flavor (Python, Spark, Dask, Ray) calls into. It owns routing, winner selection, manifest publication, and cleanup. Engines only contribute parallelism — they do not contain business logic.

## Responsibilities

| Concern | Function | Source |
|---|---|---|
| Resolve `num_dbs` (HASH or CEL) | `resolve_num_dbs` | `_writer_core.py:159` |
| Discover CEL `num_dbs` from data | `discover_cel_num_dbs` | `_writer_core.py:243` |
| Build categorical CEL routing values | `build_categorical_routing_values` | `_writer_core.py:275` |
| Route a row to a shard id | `route_key` | `_writer_core.py:122` |
| Pick winner attempt per shard | `select_winners` | `_writer_core.py:296` |
| Publish manifest + `_CURRENT` pointer | `publish_to_store` | `_writer_core.py:456` |
| Assemble user-facing `BuildResult` | `assemble_build_result` | `_writer_core.py:566` |
| Delete losing attempts | `cleanup_losers` | `_writer_core.py:602` |
| Delete stale uncommitted attempts | `cleanup_stale_attempts` | `_writer_core.py:665` |
| Delete old completed runs | `cleanup_old_runs` | `_writer_core.py:725` |

## Per-shard outcome model

Each shard attempt produces a `ShardAttemptResult` (`_writer_core.py:73`) carrying `db_id`, `attempt`, `task_attempt_id`, `db_url`, row/byte counts, and per-shard min/max key.

Per-partition outcome is `PartitionWriteOutcome` (`_writer_core.py:103`) — a list of `ShardAttemptResult`s plus the partition index.

## Winner selection

`select_winners` consumes all `ShardAttemptResult`s across all partitions, groups by `db_id`, and selects exactly one winner per shard using the comparator below. The invariant: every shard in `[0, num_dbs)` must have a winner, otherwise the build fails with `ShardCoverageError`.

The sort key is defined in `_winner_sort_key` (`_writer_core.py:363`):

```
(attempt, task_attempt_id_or_INT64_SIGNED_MAX, db_url_or_empty)
```

ascending. The smallest tuple wins. `task_attempt_id` defaults to `2**63 - 1` (signed int64 max) when missing, so explicit IDs always sort before unset ones; `db_url` is the deterministic tiebreaker.

This comparator is **identical** in every engine's writer. See [`history/design-decisions/adr-004-consistent-writer-retry.md`](../history/design-decisions/adr-004-consistent-writer-retry.md).

## Two-phase publish

`publish_to_store` (`_writer_core.py:456`) implements the two-phase commit:

1. Build SQLite manifest payload via `SqliteManifestBuilder` (`manifest.py:202`).
2. Write timestamped manifest object (`manifests/<run_id>/<timestamp>.sqlite`).
3. Overwrite `_CURRENT` pointer object (small JSON pointing at the manifest).

Readers atomically observe either the old `_CURRENT` or the new one. Manifest objects accumulate in `manifests/`; `_CURRENT` is a single mutable pointer. See [`manifest-and-current.md`](manifest-and-current.md) and [`history/design-decisions/adr-001-two-phase-publish.md`](../history/design-decisions/adr-001-two-phase-publish.md).

## Cleanup phases

After a successful publish, the writer runs three cleanup operations (any of them can be disabled by config):

- **`cleanup_losers`** — delete shard attempts that lost winner selection in *this* build.
- **`cleanup_stale_attempts`** — delete uncommitted attempts from *prior* runs older than `stale_attempt_age`.
- **`cleanup_old_runs`** — delete completed runs older than `keep_runs` to bound storage growth.

All three respect the `RunRecordLifecycle` (see [`run-registry.md`](run-registry.md)) and refuse to delete data belonging to runs in non-terminal states.

## What this module does *not* do

- It does not open shard databases. Adapters do that ([`adapters.md`](adapters.md)).
- It does not write data. Engines call `DbAdapter.put_batch` per shard.
- It does not parallelize. Engines do (`writer/python`, `writer/spark`, `writer/dask`, `writer/ray`).
- It does not implement retry logic. That's the engine's responsibility, but every engine uses the same comparator at the end so the invariant holds. See [`retry-and-cleanup.md`](retry-and-cleanup.md).

## See also

- [`sharding.md`](sharding.md) — `ShardingSpec`, key encoding, HASH vs CEL.
- [`routing.md`](routing.md) — `xxh3_db_id`, `SnapshotRouter`.
- [`manifest-and-current.md`](manifest-and-current.md) — manifest format, `_CURRENT` semantics.
- [`run-registry.md`](run-registry.md) — run lifecycle, `RunRecordLifecycle`.
