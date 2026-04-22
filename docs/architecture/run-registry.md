# Run registry

A **run** is one writer invocation. It has a unique `run_id`, a status, and a record stored in a `RunRegistry` (`shardyfusion/run_registry.py:58`). The registry is the source of truth for "is this run still in flight, did it succeed, did it fail" — and it gates cleanup so we never delete data belonging to an active run.

## Run lifecycle

`RunStatus` (`run_registry.py:30`):

| Status | Meaning | Terminal? |
|---|---|---|
| `RUNNING` | Writer is in progress. | No |
| `SUCCEEDED` | Manifest published, run complete. | Yes |
| `FAILED` | Writer aborted. | Yes |

`RunRecord` (`run_registry.py:38`) carries `run_id`, `status`, timestamps, writer info, and configuration fingerprint.

## Registry implementations

| Implementation | Backend | Module |
|---|---|---|
| `S3RunRegistry` | S3 (one object per run record) | `run_registry.py:108` |
| `InMemoryRunRegistry` | RAM (test fixture) | `run_registry.py:154` |

`resolve_run_registry` (`run_registry.py:172`) picks the right one based on writer config.

## RunRecordLifecycle

`RunRecordLifecycle` (`run_registry.py:194`) is the state machine that owns transitions:

- `RunRecordLifecycle.start(...)` — *classmethod* (`run_registry.py:229`) that constructs the lifecycle and writes the initial `RUNNING` record.
- `set_manifest_ref(manifest_ref)` (`:273`) — attach the published manifest reference.
- `mark_succeeded()` (`:277`) — flip to `SUCCEEDED`.
- `mark_failed(exc: BaseException)` (`:280`) — flip to `FAILED`, recording the exception.
- `close()` (`:287`) — stop the heartbeat thread.
- `run_record_ref` (property, `:225`) — the storage reference of the current record.

A background heartbeat thread (`_heartbeat_loop`, `:294`) periodically refreshes the `RUNNING` record so that `cleanup_stale_attempts` can distinguish "still alive" from "dead" runs.

`managed_run_record` (`run_registry.py:344`) is a `@contextmanager` that wraps a writer call and guarantees one of `mark_succeeded()` / `mark_failed()` runs before returning, even on exception, then calls `close()`.

## Why cleanup is gated on the registry

Three cleanup operations interact with the registry:

- `cleanup_losers` — reads the *current* run's record to confirm it owns the losing attempts.
- `cleanup_stale_attempts` — only deletes attempts belonging to runs in `FAILED` status (or `RUNNING` runs older than `stale_attempt_age`, which is a separate policy decision).
- `cleanup_old_runs` — deletes runs in terminal states only, never `RUNNING`.

This is why the registry exists at all: without it, a slow concurrent writer's data could be deleted by a fast cleanup pass. See [`history/design-decisions/adr-003-run-registry-deferred-cleanup.md`](../history/design-decisions/adr-003-run-registry-deferred-cleanup.md).

## What the registry does *not* do

- It does not store manifests. That's the manifest store ([`manifest-stores.md`](manifest-stores.md)).
- It does not store data. Data lives under `data/<run_id>/`.
- It does not coordinate writers. Two writers can have overlapping `RUNNING` records — they will compete via the winner-selection comparator at publish time, and the loser's data will be cleaned up.

## Format

S3 records are stored as **YAML** under `runs/<timestamp>_run_id=<run_id>_<uuidhex>/run.yaml` (default `run_registry_prefix="runs"`, `RUN_RECORD_NAME = "run.yaml"`). Format helpers:

- `_format_run_timestamp` (`run_registry.py:88`) — `RUN_REGISTRY_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"`.
- `_build_run_payload` (`run_registry.py:92`) — `yaml.safe_dump(..., sort_keys=True)`.
- `parse_run_record` (`run_registry.py:100`) — parses YAML; raises `ValueError` if the payload isn't a mapping.

## See also

- [`writer-core.md`](writer-core.md) — cleanup functions.
- [`retry-and-cleanup.md`](retry-and-cleanup.md) — interaction with engine-level retry.
- [`history/design-decisions/adr-003-run-registry-deferred-cleanup.md`](../history/design-decisions/adr-003-run-registry-deferred-cleanup.md).
- [`history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md`](../history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md) — open work item.
