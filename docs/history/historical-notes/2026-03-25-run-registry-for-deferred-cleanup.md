# 2026-03-25 Run Registry For Deferred Cleanup

- Status: `implemented`
- Date: `2026-03-25`
- Baseline repo commit before this change series: `3dba87b6e126a46fdedd48609c7d0d016cee436f`
- Baseline commit summary: `chore: remove codecov patch target`
- Initial implementation commits:
  - `d37d7d7` `feat: add shared run registry primitives`
  - `7a50b87` `feat: track writer runs in the registry`
  - `6e56637` `test: cover run registry lifecycle`

## Summary

This change series added a writer-owned run registry that stores one run-scoped YAML record per writer invocation under a dedicated S3 prefix.

The run record is operational metadata for writers and future cleanup workflows. It is not part of the reader contract and is not embedded in the manifest body.

The core idea is:

1. Create one run record when the write starts.
2. Keep it fresh with a lease heartbeat while the writer is active.
3. Mark it `succeeded` or `failed` at the end.
4. Let a later sweeper derive cleanup targets by combining the run record, the manifest if present, and the existing `shards/run_id=...` layout.

This replaced the earlier direction of writing per-attempt cleanup artifacts to S3.

## Why This Change Was Needed

The existing best-effort `cleanup_losers()` behavior is useful, but it is not a complete contract for deferred cleanup:

- it runs only inline during the write
- it does not leave a durable writer-owned record that a periodic cleanup job can discover later
- if the writer process dies or cleanup fails transiently, a later sweeper needs some other durable signal to know which run should be inspected

At the same time, the solution had to satisfy a few constraints:

- work across all writers: Python, Dask, Ray, Spark, and single-db modes
- work for both HASH and CEL sharding
- avoid adding reader-only overhead to the manifest
- avoid generating one S3 object per partition or per attempt
- leave a clean migration path to a future SQLite-backed implementation

## Problem With The Earlier Direction

The earlier plan was to write failed-attempt or attempt-journal YAML objects to S3 and let a later cleanup job process them.

That direction had three problems.

### Object Fan-Out

One object per attempt or per partition does not scale well.

For large future runs, the cardinality can be very high. A run with hundreds of thousands or millions of partitions would produce an unacceptable number of tiny operational objects.

### Distributed Capture Semantics

A precise per-attempt journal is easy for a driver-owned writer, but much harder for distributed systems:

- Spark executor attempts can fail before returning structured results
- Dask and Ray have framework-specific event paths, but they are not the same abstraction
- a shared mutable run journal on S3 is awkward because S3 is an object store, not a transactional append log

### Reader Contract Pollution

Putting cleanup telemetry directly in the manifest would make a reader-facing artifact carry operational details that readers do not need.

That is the wrong ownership boundary. Cleanup metadata belongs to the writer side.

## Main Design Decisions

### 1. Separate Operational Metadata From The Manifest

The manifest remains the reader-facing snapshot contract.

The run registry is a separate writer-owned artifact. On successful runs, the run record stores `manifest_ref`. The manifest does not need to point back to the run record.

This keeps readers decoupled from cleanup mechanics.

### 2. Store One Record Per Run, Not Per Attempt

The run registry stores one YAML object per writer invocation under:

`s3://.../<run_registry_prefix>/<timestamp>_run_id=<run_id>_<uuid>/run.yaml`

That gives cleanup jobs one stable entrypoint per run and avoids S3 object explosions.

### 3. Derive Loser Cleanup From Existing Shard Layout

The solution does not try to persist a complete list of failed attempts.

Instead, a future sweeper derives losers from what already exists:

- the run record says whether the run is still active, succeeded, or failed
- the manifest identifies winner shard URLs when a successful publish exists
- the shard layout already isolates attempts under `shards/run_id=<run_id>/.../attempt=NN`

So the sweeper can:

- load the manifest and keep winner `db_url`s
- list all prefixes under the run's shard tree
- delete everything else

For failed runs without a manifest, the sweeper can delete the whole run shard tree once the run is terminal or its lease expires.

### 4. Use A Lease-Based Liveness Model

The run record is created as `running` and refreshed with a heartbeat.

Fields include:

- `status`
- `started_at`
- `updated_at`
- `lease_expires_at`
- `s3_prefix`
- `shard_prefix`
- `db_path_template`
- `manifest_ref`
- optional failure context

This gives a future sweeper a simple rule:

- active `running` record with valid lease: leave it alone
- `failed` record: eligible for failed-run cleanup
- stale `running` record with expired lease: treat as abandoned

### 5. Keep The Interface Storage-Agnostic

The implementation introduced a `RunRegistry` abstraction with:

- `S3RunRegistry` now
- `InMemoryRunRegistry` for tests

This is intentionally shaped so a future SQLite-backed implementation can replace the storage backend without changing the writer lifecycle.

## Alternatives Considered

### Alternative 1: Failed-Attempt YAML Object Per Run

Idea:

- write one YAML report at the end containing failed attempt prefixes and run identity

Rejected because:

- failed Spark executor attempts may not be observable at the driver
- it still requires precise attempt bookkeeping across all backends
- it is better than per-attempt objects, but still pushes the design toward explicit attempt journaling instead of deriving from stable layout

### Alternative 2: Immutable Per-Attempt Journal Objects Plus Summary

Idea:

- each attempt writes an immutable record
- the driver later writes one summary object

Rejected because:

- S3 object fan-out is too high
- a million-partition future makes this design operationally expensive
- framework-specific attempt signaling becomes part of the core design

### Alternative 3: Put Attempt Metadata In The Manifest

Idea:

- extend manifest payload with attempt journal or cleanup metadata

Rejected because:

- readers do not need it
- failed runs may have no manifest at all
- it mixes operational cleanup metadata into the snapshot contract

### Alternative 4: Driver-Owned SQLite Attempt Journal Right Now

Idea:

- record all attempts into one SQLite DB and publish it at the end

This is still a plausible future direction, but it was not chosen for the current implementation because:

- it is a larger design jump than needed for the immediate cleanup problem
- distributed attempt capture remains tricky unless all frameworks can reliably report attempt starts to a shared durable journal
- the simpler run-registry contract already solves discovery and cleanup ownership without committing to per-attempt persistence

### Alternative 5: Framework-Specific Event Collection

Idea:

- Spark listeners
- Dask scheduler events
- Ray actors or queues

Rejected as the primary design because:

- it is not uniform across frameworks
- it solves detailed attempt telemetry, not the simpler question of "which runs need cleanup inspection?"
- it adds framework-specific infrastructure to a problem that can be solved from existing object layout

## Final Solution

The implemented solution is a driver-owned run registry.

### Writer Lifecycle

At run start:

- create one run record
- set `status=running`
- set `lease_expires_at`

While the run is active:

- refresh `updated_at`
- refresh `lease_expires_at`

At successful completion:

- publish the manifest
- update `manifest_ref`
- mark the run record `succeeded`

At handled failure:

- mark the run record `failed`
- include `error_type` and `error_message` when available

If the process dies before terminal update:

- the record stays `running`
- the lease eventually expires
- a sweeper can treat it as abandoned

### Cleanup Model

The current implementation does not add the sweeper itself. It establishes the contract the sweeper will use later.

The intended cleanup behavior is:

#### Successful Run With Manifest

- read the run record
- load `manifest_ref`
- keep winner `db_url`s from the manifest
- list all attempt prefixes under `shards/run_id=<run_id>/`
- delete non-winner attempts

#### Failed Run Without Manifest

- read the run record
- if `status=failed`, or `status=running` with expired lease, delete the whole run shard tree

This works for CEL as well because CEL still resolves into concrete `db_id` shards and writes into the same run-scoped attempt layout.

## Implementation Areas

Primary implementation areas:

- `shardyfusion/run_registry.py`
  - run record model
  - S3 and in-memory registries
  - lifecycle helper with heartbeat and terminal state updates
- `shardyfusion/config.py`
  - `output.run_registry_prefix`
  - optional injected `run_registry`
- `shardyfusion/manifest.py`
  - `BuildResult.run_record_ref`
- writer entrypoints
  - Python, Dask, Ray, Spark, and single-db writers now create and complete a run record

Test coverage was added across:

- shared unit tests for run-record lifecycle
- writer unit tests for success and retry-success paths
- local-S3 integration tests that validate run-record contents
- e2e smoke scenarios that validate run-record contents when the Garage environment is available

## Pros

### One Durable Artifact Per Run

The operational metadata footprint stays small and predictable.

### Uniform Across Writers

The same contract works for:

- Python
- Dask
- Ray
- Spark
- single-db modes
- HASH and CEL sharding

### Readers Stay Unchanged

No reader behavior depends on the run registry.

### Cleanup Can Be Derived, Not Logged Explicitly

The solution reuses the existing attempt-isolated shard layout instead of inventing a separate attempt journal.

### Good Fit For Future SQLite

The abstraction is already separated from the storage backend, so a future SQLite-backed run registry can preserve the same writer lifecycle.

## Cons

### No Per-Attempt Provenance

The run registry tells the sweeper which run to inspect, not the exact list of attempts that failed.

That is a deliberate tradeoff.

### Sweeper Must List S3

Deferred cleanup now depends on listing the run shard tree and comparing it to the manifest. That is simple, but it shifts some work to the sweeper.

### Abandoned Run Detection Is Lease-Based

Hard crashes are detected by expired heartbeat lease, not by an explicit crash marker.

### Spark Retry Telemetry Is Still Implicit

The design does not try to record every Spark task attempt. It only guarantees that abandoned or completed runs are discoverable for cleanup.

## Why This Solution Was Chosen

This solution was chosen because it is the smallest design that satisfies the real operational need:

- discover unfinished or completed runs later
- keep cleanup metadata out of reader-facing artifacts
- work uniformly across all writers
- scale better than per-attempt S3 records

It does not try to solve exact per-attempt journaling. That was an intentional boundary.

The run registry gives the system one durable source of truth for run lifecycle, while loser detection continues to come from the existing shard layout and manifest winner selection.

## Follow-On Work

The run registry is the discovery layer. A future cleanup implementation still needs to:

1. list run records under the configured registry prefix
2. evaluate terminal or expired runs
3. load manifests when present
4. delete non-winner or abandoned attempt prefixes
5. report cleanup results

If a later SQLite-based writer journal is introduced, it should implement the same conceptual contract:

- one logical run record
- terminal state
- manifest pointer when available
- enough metadata for deferred cleanup discovery
