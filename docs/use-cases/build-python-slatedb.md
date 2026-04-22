# Build a SlateDB snapshot with the Python writer

Use the **Python writer** (no Spark, no Java, no cluster) to build a sharded SlateDB snapshot from any Python iterable, and read it back with `ShardedReader`.

## When to use

- You have a single-process or multi-process Python job that produces records (database extract, in-memory dataset, file scan).
- You want SlateDB on the read side (the default, lowest-friction backend).
- You want to ship a self-contained pipeline without a Spark cluster or Java runtime.
- Dataset fits comfortably in one machine's memory *or* you can stream it as an iterator.

## When NOT to use

- You want SQL queries or range-read access at read time — use the SQLite backend ([`build-python-sqlite.md`](build-python-sqlite.md)).
- You need vector search alongside KV — use [`build-python-slatedb-lancedb.md`](build-python-slatedb-lancedb.md) (composite) or [`build-python-sqlite-vec.md`](build-python-sqlite-vec.md) (unified).
- Dataset is many TB and cannot stream from a single host — use a distributed writer.

## Install

```bash
uv add 'shardyfusion[writer-python]'
# or
pip install 'shardyfusion[writer-python]'
```

The `writer-python` extra pulls SlateDB. If you want the SQLite backend instead, use `writer-python-sqlite`.

## Minimal example

```python
from shardyfusion import WriteConfig, ShardedReader
from shardyfusion.writer.python import write_sharded

records = [{"id": i, "payload": f"row-{i}".encode()} for i in range(10_000)]

config = WriteConfig(
    num_dbs=4,
    s3_prefix="s3://my-bucket/snapshots/users",
)

result = write_sharded(
    records,
    config,
    key_fn=lambda r: r["id"],          # int → encoded as u64be by default
    value_fn=lambda r: r["payload"],   # bytes
)

print(result.manifest_ref.ref)
print(result.run_id)

# Read back
with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/tmp/shardyfusion-cache",
) as reader:
    print(reader.get(42))  # → b"row-42"
```

`write_sharded` returns a [`BuildResult`](../architecture/manifest-and-current.md) containing the manifest ref, run id, per-shard metadata, and timing.

## Configuration

The writer signature (`shardyfusion/writer/python/writer.py:77`):

```python
write_sharded(
    records,
    config,
    *,
    key_fn,                                          # required
    value_fn,                                        # required
    columns_fn=None,                                 # for CEL routing or vector auto-extract
    vector_fn=None,                                  # for unified KV+vector mode
    parallel=False,                                  # one subprocess per shard
    max_queue_size=100,
    max_parallel_shared_memory_bytes=256 * 1024 * 1024,         # global cap
    max_parallel_shared_memory_bytes_per_worker=32 * 1024 * 1024,
    max_writes_per_second=None,                      # token-bucket rate limit
    max_write_bytes_per_second=None,
    max_total_batched_items=None,                    # single-process backpressure
    max_total_batched_bytes=None,
)
```

Key `WriteConfig` fields used by this use-case (`shardyfusion/config.py:162`):

| Field | Default | Purpose |
|---|---|---|
| `num_dbs` | `None` | Number of shards. Required (>0) for HASH sharding without `max_keys_per_shard`. |
| `s3_prefix` | `""` | `s3://bucket/prefix` — required, must include a non-empty key prefix. |
| `key_encoding` | `KeyEncoding.U64BE` | How `key_fn`'s return value is serialized. Use `U32BE` for 4-byte ints, `UTF8` for strings, `RAW` for `bytes`. |
| `batch_size` | `50_000` | Pairs per write batch into the adapter. |
| `adapter_factory` | `None` | `None` → `SlateDbFactory()`. |
| `sharding` | `ShardingSpec()` | Default: HASH. Override for CEL or `max_keys_per_shard`. |
| `output.local_root` | `$TMPDIR/shardyfusion` | Where shards are staged before upload. |
| `manifest.store` | `None` | Defaults to S3-backed `ManifestStore`. Override for Postgres. |
| `metrics_collector` | `None` | Pass a `PrometheusCollector` or `OtelCollector` to enable metrics. |
| `shard_retry` | `None` | Required for shard retries in parallel mode (uses file-spool fallback). |

## Functional properties

- **Streaming-safe**: `records` is `Iterable[T]` — generators work; the entire dataset does not need to fit in memory in single-process mode (subject to per-shard buffer caps).
- **Deterministic sharding**: with HASH (default), the same key always routes to the same `db_id` for a given `num_dbs`. See [`architecture/sharding.md`](../architecture/sharding.md).
- **Atomic publish**: the build is invisible to readers until `_CURRENT` is updated. Two-phase publish details in [`architecture/manifest-and-current.md`](../architecture/manifest-and-current.md).

## Non-functional properties

- **Single-process mode** (`parallel=False`, default): all shard adapters open in the same process. Memory ≈ `num_dbs × per-shard-buffer`. Best for ≤ ~32 shards or when adapter open is cheap.
- **Parallel mode** (`parallel=True`): one `multiprocessing.spawn` subprocess per shard. Records are streamed to workers via shared memory in 8 MiB chunks. Caps: 256 MiB global, 32 MiB per worker by default. Best when shard writes are CPU-bound (SQLite checksums, compression).
- **Backpressure**: `max_total_batched_items` / `max_total_batched_bytes` (single-process only) flush the largest shard buffer when buffer caps are exceeded. `max_writes_per_second` / `max_write_bytes_per_second` apply token-bucket rate limiting.

## Guarantees

- A successful return means the manifest **and** `_CURRENT` are published. Readers opened after this call observe the new snapshot.
- `BuildResult.manifest_ref` is the canonical reference; pin it for reproducible reads.
- Shard URLs in the manifest are the durable winners — losers (failed/superseded attempts) are scheduled for cleanup. See [`architecture/retry-and-cleanup.md`](../architecture/retry-and-cleanup.md).

## Weaknesses

- **No distributed scale-out.** A single host produces all shards. For >100 shards or >100 GB of compressed output, a distributed writer is more efficient.
- **Parallel mode + inferred CEL routing is rejected** at config validation (`writer/python/writer.py:188-193`). Materialize records and use single-process mode, or declare `routing_values` explicitly.
- **No checkpoint/resume.** A failure mid-build aborts the run; the next attempt starts from scratch. Loser cleanup (next successful run) eventually removes any orphaned shard objects.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Bad `WriteConfig` (e.g. missing `s3_prefix`, `num_dbs <= 0`) | `ConfigValidationError` at `WriteConfig.__post_init__` | Fix config; nothing was written. |
| `vector_fn` provided without `vector_spec` (or vice versa) | `ConfigValidationError` at `write_sharded` entry | Either remove `vector_fn` or set `config.vector_spec`. |
| Shard write fails (transient backend error) | `ShardWriteError` raised; if `shard_retry` is set, retried per its config | Set `config.shard_retry`. |
| Some shards have zero successful attempts | `ShardCoverageError` at manifest build | Investigate worker logs; rerun. Losers from this run are cleaned up by a future successful run. |
| Manifest object PUT fails | `PublishManifestError` | Transient — rerun. |
| `_CURRENT` PUT fails after manifest published | `PublishCurrentError` | The manifest exists but is invisible. Rerunning publishes a new pointer; the orphaned manifest is cleaned up later. |
| S3 5xx / throttling | wrapped as `S3TransientError`; auto-retried with exponential backoff | None — handled internally. |

See [`architecture/error-model.md`](../architecture/error-model.md) for the full hierarchy.

## Reading the snapshot

The reader is **completely decoupled** from the writer flavor. A snapshot built with the Python writer reads with the same `ShardedReader` / `AsyncShardedReader` as a snapshot built with Spark, Dask, or Ray — only the **adapter backend** (SlateDB vs SQLite) determines which reader factory you need.

```python
from shardyfusion import ShardedReader

with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
) as reader:
    print(reader.get(42))                            # routed lookup
    print(reader.multi_get([1, 2, 3]))               # batch
    print(reader.route_key(42))                      # which shard?
    print(reader.snapshot_info().run_id)             # what's pinned?
    print([s.row_count for s in reader.shard_details()])

    # Borrow the underlying SlateDB handle for the shard a key lives in
    with reader.reader_for_key(42) as handle:
        ...  # handle.reader is the SlateDB shard reader
```

See [`read-sync-slatedb.md`](read-sync-slatedb.md) for the full reader API surface (sync) or [`read-async-slatedb.md`](read-async-slatedb.md) for the async equivalent.

## See also

- [`architecture/writer-core.md`](../architecture/writer-core.md) — what `_writer_core` does on every shard attempt.
- [`architecture/sharding.md`](../architecture/sharding.md) — HASH vs CEL, key encodings, `max_keys_per_shard`.
- [`architecture/adapters.md`](../architecture/adapters.md) — SlateDB adapter internals.
- [`use-cases/build-python-sqlread-sync-slatedb.md`](read-sync-slatedb.md) — read-side details for `ShardedReader`.
