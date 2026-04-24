# ADR-001: Two-phase manifest publish

**Status:** Accepted (2026-03-14)
**Source:** [`historical-notes/2026-03-14-manifest-history-and-rollback.md`](../historical-notes/2026-03-14-manifest-history-and-rollback.md)

## Context

Writers produce per-shard databases plus a manifest enumerating them. Readers need:

- Atomic visibility: a snapshot is either fully visible or not at all.
- Rollback: revert `_CURRENT` to a previous manifest without re-uploading data.
- History: keep prior manifests addressable for audit.

A naive single-write publish (overwrite a `manifest` key) loses history and races readers.

## Decision

Publish in two phases:

1. **Phase 1 — Write manifest** at `manifests/<timestamp>_run_id=<run_id>/manifest`. Path is content-addressable by run; prior manifests remain.
2. **Phase 2 — Swap `_CURRENT`** to point at the new manifest ref. `_CURRENT` is the only mutable pointer.

`_CURRENT` carries its own `format_version: int = 1` (separate from manifest format version).

## Consequences

- Atomic visibility flips on the `_CURRENT` swap.
- Rollback = `set_current(<old ref>)`. No data movement.
- History = list of `manifests/*` keys, ordered by timestamp (`%Y-%m-%dT%H:%M:%S.%fZ`).
- Old manifest data accumulates in S3 until `cleanup` reaps unreferenced shards.

## Related

- [`architecture/manifest-and-current.md`](../../architecture/manifest-and-current.md)
- ADR-003 (run registry — required for safe cleanup).
