# ADR-003: Run registry for deferred cleanup

**Status:** Accepted (2026-03-25)
**Source:** [`historical-notes/2026-03-25-run-registry-for-deferred-cleanup.md`](../historical-notes/2026-03-25-run-registry-for-deferred-cleanup.md)

## Context

Two-phase publish (ADR-001) keeps history but lets dead data accumulate. Naive "delete anything not in `_CURRENT`" is unsafe:

- Concurrent writes from a parallel run may have produced shard files that aren't yet in any manifest.
- A losing writer in a publish race may have uploaded shards that become orphans.
- A reader may still hold a reference to an older manifest.

We need a **registry of in-flight and completed runs** to decide what is safe to delete.

## Decision

Introduce a **run registry**:

- Each writer run creates a `RunRecord` at `runs/<timestamp>_run_id=<run_id>_<uuidhex>/run.yaml`.
- `RunRecordLifecycle` exposes `start` (classmethod), `set_manifest_ref`, `mark_succeeded`, `mark_failed(exc)`, `close`.
- `RunStatus` ∈ {`RUNNING`, `SUCCEEDED`, `FAILED`}.
- Cleanup consults the registry: a shard file is reapable iff no `RUNNING` run could plausibly own it AND no published manifest references it.

## Consequences

- `cleanup` is safe by construction (deferred until runs settle).
- Failed runs are observable for debugging.
- Run records are independent of manifests — rollback does not delete run records.
- Adds a small per-run YAML write.

## Related

- [`architecture/run-registry.md`](../../architecture/run-registry.md)
- [`historical-notes/2026-03-15-reliable-loser-cleanup-plan.md`](../historical-notes/2026-03-15-reliable-loser-cleanup-plan.md) (still-open work).
