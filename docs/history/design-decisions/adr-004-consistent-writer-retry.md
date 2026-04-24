# ADR-004: Consistent writer retry model

**Status:** Accepted (2026-03-25)
**Source:** [`historical-notes/2026-03-25-consistent-writer-retry.md`](../historical-notes/2026-03-25-consistent-writer-retry.md)

## Context

Writers across frameworks (Spark, Dask, Ray, Python) have their own retry semantics for failed tasks. Without a shared model:

- Behavior diverges across frameworks (e.g. Ray retries a task; Dask doesn't; Python writer aborts).
- Per-shard transient failures cause whole-build aborts.
- Operators can't tune retry budget centrally.

## Decision

Define a **writer-level retry contract** independent of the framework:

- `WriteConfig.shard_retry` controls per-shard retry budget for transient adapter failures.
- The `_writer_core` shared module owns the retry loop; framework writers delegate to it.
- Framework-native retries (Spark task retry, Ray task retry) are **additive** — they handle worker-level failures; `shard_retry` handles adapter-level failures (S3 transient, SlateDB I/O).
- File-spool fallback exists in the Python writer when `shard_retry` is set, to allow re-running a single shard without re-feeding the input iterator.

## Consequences

- Consistent shard-level retry semantics across frameworks.
- `ShardCoverageError` raised only after `shard_retry` is exhausted.
- Operators tune one knob (`shard_retry`) regardless of framework.

## Related

- [`architecture/retry-and-cleanup.md`](../../architecture/retry-and-cleanup.md)
- [`architecture/writer-core.md`](../../architecture/writer-core.md)
