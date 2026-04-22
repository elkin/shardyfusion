# Build a SlateDB snapshot with the Dask writer

Use the **Dask writer** to build a sharded SlateDB snapshot from a Dask `DataFrame` — Spark-free, Java-free.

## When to use

- Records live in a Dask `DataFrame` (Parquet/CSV scan, Dask-SQL output).
- You want distributed scale-out without a JVM.
- You want SlateDB on the read side.

## When NOT to use

- Single-host workload — use [`build-python-slatedb.md`](build-python-slatedb.md).
- You want SQLite shards — use [`build-dask-sqlite.md`](build-dask-sqlite.md).

## Install

```bash
uv add 'shardyfusion[writer-dask]'
```

`dask[dataframe]>=2024.1` comes with the extra.

## Minimal example

```python
import dask.dataframe as dd
from shardyfusion import WriteConfig
from shardyfusion.writer.dask import write_sharded
from shardyfusion.serde import ValueSpec

ddf = dd.read_parquet("s3://lake/users/")

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users",
)

result = write_sharded(
    ddf,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)
print(result.manifest_ref.ref)
```

## Configuration

Dask-specific knobs on `write_sharded` (`shardyfusion/writer/dask/writer.py:151`):

| Param | Default | Purpose |
|---|---|---|
| `key_col` | required | DataFrame column used for routing. |
| `value_spec` | required | How to encode rows to bytes — `binary_col`, `json_cols`, or callable. |
| `sort_within_partitions` | `False` | Sort each partition by `key_col` before writing. |
| `verify_routing` | `True` | Re-verify writer-side routing matches reader-side router. |
| `max_writes_per_second` / `max_write_bytes_per_second` | `None` | Token-bucket rate limits. |
| `vector_fn` / `vector_columns` | `None` | For unified KV+vector mode. |

The writer repartitions internally via `ddf.shuffle(on=DB_ID_COL, npartitions=num_dbs)` so the per-shard task layout matches `num_dbs` regardless of input partitioning.

## Single-DB variant

`write_single_db` (`shardyfusion/writer/dask/single_db_writer.py:67`) — enforces `config.num_dbs == 1`. Knobs include `num_partitions`, `prefetch_partitions`, `cache_input`.

## Functional properties

- Sharding decisions made via `map_partitions` (`writer/dask/sharding.py`).
- Routing identical to other writers — same `route_key` from `_writer_core`.
- Atomic two-phase publish.

## Non-functional properties

- Uses the **ambient Dask scheduler** — distributed, threads, processes, or single-machine. No knob is exposed.
- One Dask task per shard after the shuffle. Memory per task ≈ `partition_size + per-shard buffers`.
- Routing pass and write pass are separate Dask graphs; the shuffle materializes between them.

## Guarantees

- Same as all writers: successful return ⇒ manifest + `_CURRENT` published.
- `verify_routing=True` (default) catches routing drift.

## Weaknesses

- **No Spark-style executor preemption recovery.** A worker loss surfaces as a Dask task failure; rely on `config.shard_retry` for recovery.
- **`write_vector_sharded` not re-exported.** Lives at `shardyfusion/writer/dask/writer.py:647`; import from the module.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Routing mismatch | `ShardAssignmentError` ("Dask/Python routing mismatch") | Bug in routing change. Don't silence by disabling `verify_routing`. |
| Worker death | Dask retries; if exhausted, `ShardCoverageError` | Configure Dask retries + `config.shard_retry`. |
| Manifest publish | `PublishManifestError` / `PublishCurrentError` | Transient; rerun. |

## Reading the snapshot

The reader is **decoupled from the writer flavor** — no Dask dependency at read time. A snapshot built by Dask reads identically to one built by Python/Spark/Ray.

```python
from shardyfusion import ShardedReader

with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
) as reader:
    value = reader.get(42)
    with reader.reader_for_key(42) as handle:
        ...  # native SlateDB shard handle
```

See [`read-sync-slatedb.md`](read-sync-slatedb.md) for the full reader API.

## See also

- [`build-dask-sqlite.md`](build-dask-sqlite.md) — SQLite backend.
- [`build-spark-slatedb.md`](build-spark-slatedb.md) — when you have Spark.
- [`build-ray-slatedb.md`](build-ray-slatedb.md) — when you want Ray actors.
- [`architecture/writer-core.md`](../architecture/writer-core.md), [`architecture/sharding.md`](../architecture/sharding.md).
