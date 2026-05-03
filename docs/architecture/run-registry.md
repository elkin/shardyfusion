# Run registry

A **run** is one writer invocation. It has a unique `run_id`, a status, and a record stored in a `RunRegistry` (`shardyfusion/run_registry.py`). The registry is the source of truth for "is this run still in flight, did it succeed, did it fail" — and it gates cleanup so we never delete data belonging to an active run.

## Run lifecycle

`RunStatus` (`run_registry.py`):

| Status | Meaning | Terminal? |
|---|---|---|
| `RUNNING` | Writer is in progress. | No |
| `SUCCEEDED` | Manifest published, run complete. | Yes |
| `FAILED` | Writer aborted. | Yes |

`RunRecord` (`run_registry.py`) carries `run_id`, `status`, timestamps, writer info, and configuration fingerprint.

## Registry implementations

| Implementation | Backend | Module |
|---|---|---|
| `S3RunRegistry` | S3 (one object per run record) | `run_registry.py` |
| `InMemoryRunRegistry` | RAM (test fixture) | `run_registry.py` |

`resolve_run_registry` (`run_registry.py`) picks the right one based on writer config. If `config.lifecycle.run_registry` is set on a KV or vector writer config, that registry is used directly. If the manifest store is in-memory, the run registry is also in-memory. Otherwise the default is `S3RunRegistry` under `storage.s3_prefix/output.run_registry_prefix`.

Default S3 run-registry resolution uses writer-level `credential_provider` and `s3_connection_options`. Manifest-level credentials/options are scoped to the default S3 manifest store and are not used for run-registry S3 access. If the run registry needs different settings, pass an explicit `run_registry`.

## RunRecordLifecycle

`RunRecordLifecycle` (`run_registry.py`) is the state machine that owns transitions:

- `RunRecordLifecycle.start(...)` — *classmethod* that constructs the lifecycle and writes the initial `RUNNING` record.
- `set_manifest_ref(manifest_ref)` — attach the published manifest reference.
- `mark_succeeded()` — flip to `SUCCEEDED`.
- `mark_failed(exc: BaseException)` — flip to `FAILED`, recording the exception.
- `close()` — stop the heartbeat thread.
- `run_record_ref` — the storage reference of the current record.

A background heartbeat thread (`_heartbeat_loop`) periodically refreshes the `RUNNING` record's `updated_at` and `lease_expires_at` fields. This gives cleanup jobs and external operators a persisted liveness deadline for distinguishing an actively heartbeating writer from a stale `RUNNING` record; the current `_writer_core.cleanup_stale_attempts()` helper itself only removes non-winning attempts for the current manifest run.

`RunRecordLifecycle` implements the context manager protocol. Use `with RunRecordLifecycle.start(...) as run_record:` to mark the run failed on exception and call `close()` on exit unless the run was already terminal.

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

- `_format_run_timestamp` — `RUN_REGISTRY_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"`.
- `_build_run_payload` — `yaml.safe_dump(..., sort_keys=True)`.
- `parse_run_record` — parses YAML; raises `ValueError` if the payload isn't a mapping.

## See also

- [`writer-core.md`](writer-core.md) — cleanup functions.
- [`retry-and-cleanup.md`](retry-and-cleanup.md) — interaction with engine-level retry.
- [`history/design-decisions/adr-003-run-registry-deferred-cleanup.md`](../history/design-decisions/adr-003-run-registry-deferred-cleanup.md).
- [`history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md`](../history/implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md) — open work item.
