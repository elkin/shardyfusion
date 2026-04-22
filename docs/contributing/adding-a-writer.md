# Adding a writer

shardyfusion ships four writer flavors: **Spark**, **Dask**, **Ray**, and **Python** (single-process / multi-process). Each is a thin orchestration layer over `_writer_core`. This page describes the pattern so you can add a fifth (e.g. Beam, Flink, plain `concurrent.futures`).

Before starting, read:

- [`architecture/writer-core.md`](../architecture/writer-core.md) — the shared shard-attempt pipeline.
- [`architecture/sharding.md`](../architecture/sharding.md) — how rows are routed.
- [`architecture/manifest-and-current.md`](../architecture/manifest-and-current.md) — two-phase publish.

## What "adding a writer" means

A writer is responsible for:

1. **Distributing records to shards** — using the chosen `ShardingStrategy`.
2. **Driving `_writer_core` per shard attempt** — open adapter, write batches, checkpoint, report `ShardAttemptResult`.
3. **Selecting winners** — call `select_winners` once all attempts are reported.
4. **Building and publishing the manifest** — call `assemble_build_result` then `publish_to_store`.
5. **Cleaning up losers** — call `cleanup_losers`.
6. **Run-registry lifecycle** — `managed_run_record(...)` context manager bookends the whole thing.

Steps 2–6 are framework-agnostic and live in `_writer_core`. Step 1 is what makes each writer different.

## Anatomy

```
shardyfusion/writer/<flavor>/
├── __init__.py
├── writer.py          # public entry point: build() or write_sharded()
├── sharding.py        # framework-specific row distribution
└── (helpers)
```

Read `shardyfusion/writer/python/writer.py` first — it's the simplest and uses no framework primitives.

## Pattern

### 1. Public entry point

The signature should be familiar to users of the framework. Keep `WriteConfig` as the canonical configuration container; framework-specific knobs live on the function signature.

```python
def write_sharded(
    records: <framework-native collection>,
    config: WriteConfig,
    *,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    vector_fn: Callable[[T], tuple[int | str, Any, dict[str, Any] | None]] | None = None,
    # framework-specific knobs ...
) -> BuildResult: ...
```

Spark uses a DataFrame; Dask uses a Bag/DataFrame; Ray uses Datasets; Python uses an iterable. The internal pipeline is the same.

### 2. Routing

Use `route_key(key, sharding, num_dbs)` from `_writer_core` for the per-row routing decision. Don't duplicate the HASH/CEL logic.

For frameworks that pre-partition (Spark, Dask), partition by `route_key` then hand each partition to a worker that runs the per-shard write loop. For frameworks without pre-partitioning (Ray Datasets, plain Python), groupby on the route key.

### 3. Per-shard write loop

For each shard attempt:

1. Construct the adapter via `config.adapter_factory(db_url=..., local_dir=...)`.
2. Buffer records into batches of `config.batch_size`.
3. `adapter.write_batch(pairs)` per batch.
4. After all records: `adapter.flush()`, then `adapter.checkpoint()`.
5. Report a `ShardAttemptResult` (db_id, attempt index, row count, byte count, min/max keys, db_url returned by checkpoint).

### 4. Winner selection + publish

Once all attempts have reported (success **or** failure):

```python
winners = select_winners(attempts)
build = assemble_build_result(...)
manifest_ref = publish_to_store(build, manifest_options)
cleanup_losers(attempts, winners, ...)
```

Wrap everything in `managed_run_record(config)` so the run-registry lifecycle (start → set_manifest_ref → mark_succeeded/mark_failed → close) is correctly emitted.

### 5. Retries

Per-shard retry is opt-in via `config.shard_retry`. Implement it as: catch exceptions inside the per-shard worker, increment the attempt counter, write under a fresh `db_url`, report the new attempt. Winner selection naturally picks the latest successful attempt because the sort key is `(attempt, ...)`.

The Python parallel writer file-spool pattern (`_parallel_writer.py:1405`) is a reference implementation.

## Sharding extras

If the framework needs framework-specific sharding helpers (e.g. Spark `repartitionByRange`), put them in `writer/<flavor>/sharding.py`. The Spark writer's `sharding.py` is the canonical example.

## Optional dependency gating

The framework dependency goes behind an extra:

```toml
# pyproject.toml
[project.optional-dependencies]
writer-myframework = [
  "shardyfusion[slatedb]",
  "myframework>=2.0",
]
writer-myframework-sqlite = [
  "shardyfusion[sqlite]",
  "myframework>=2.0",
]

[dependency-groups]
cap-writer-myframework = ["myframework>=2.0"]
```

Import the framework lazily inside the writer module — never at the top of `shardyfusion/__init__.py`'s import graph.

## Tests

| Layer | What to test |
|---|---|
| Unit | Routing decisions on synthetic input; uses `FakeSlateDbAdapter`. |
| Unit | Backpressure / rate-limit knobs work as documented. |
| Integration | End-to-end against moto S3 with real adapters. |
| E2E | Garage S3 via `tests/e2e/writer/`. |
| Property | `tests/unit/writer/core/test_routing_contract.py` must still pass. |

Add `py{311,312,313}-myframework-slatedb-{unit,integration,e2e}` envs to `tox.ini`.

## Documentation

Per [`adding-a-use-case.md`](adding-a-use-case.md), add at minimum:

- `docs/use-cases/build-myframework-slatedb.md`
- `docs/use-cases/build-myframework-sqlite.md`

Update [`architecture/writer-core.md`](../architecture/writer-core.md) if you introduce a new shared primitive.

## Common mistakes

- **Re-implementing `route_key`.** Always reuse `_writer_core.route_key`.
- **Forgetting `managed_run_record`.** Run records won't be written; loser cleanup deferred from a previous run can't progress.
- **Top-level framework import.** Breaks the base install.
- **Accepting `key_col` instead of `key_fn`.** Spark is the exception (DataFrame-native); for everything else, use `key_fn`.
- **Adding new `WriteConfig` fields for one framework.** Framework knobs go on the writer function signature.

## See also

- [`architecture/writer-core.md`](../architecture/writer-core.md).
- [`architecture/retry-and-cleanup.md`](../architecture/retry-and-cleanup.md).
- [`adding-an-adapter.md`](adding-an-adapter.md) — the related "add a backend" workflow.
- [`history/design-decisions/adr-004-consistent-writer-retry.md`](../history/design-decisions/adr-004-consistent-writer-retry.md) (after Phase 6).
