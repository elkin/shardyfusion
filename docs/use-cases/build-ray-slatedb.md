# Build a SlateDB snapshot with the Ray writer

Use the **Ray writer** to build a sharded SlateDB snapshot from a `ray.data.Dataset` — Spark-free, Java-free.

## When to use

- Records live in a Ray Dataset (Ray Data pipeline, ML preprocessing).
- You want Ray's actor-based scheduling.
- You want SlateDB on the read side.

## When NOT to use

- No Ray cluster / no Ray pipeline — use [`build-python-slatedb.md`](build-python-slatedb.md).
- You want SQLite shards — use [`build-ray-sqlite.md`](build-ray-sqlite.md).

## Install

```bash
uv add 'shardyfusion[writer-ray]'
```

`ray[data]>=2.20` comes with the extra.

## Minimal example

```python
import ray
from shardyfusion import WriteConfig
from shardyfusion.writer.ray import write_sharded
from shardyfusion.serde import ValueSpec

ray.init()
ds = ray.data.read_parquet("s3://lake/users/")

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/users",
)

result = write_sharded(
    ds,
    config,
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)
print(result.manifest_ref.ref)
```

## Configuration

Ray-specific knobs on `write_sharded` (`shardyfusion/writer/ray/writer.py:154`):

| Param | Default | Purpose |
|---|---|---|
| `key_col` | required | Column used for routing. |
| `value_spec` | required | Row → bytes encoder. |
| `sort_within_partitions` | `False` | Sort each partition by `key_col`. |
| `verify_routing` | `True` | Re-verify writer/reader routing agreement. |
| `max_writes_per_second` / `max_write_bytes_per_second` | `None` | Rate limits. |
| `vector_fn` / `vector_columns` | `None` | Unified KV+vector mode. |

Internally the writer:

- Forces `DataContext.shuffle_strategy = HASH_SHUFFLE` for the duration of the build (process-wide; guarded by `_SHUFFLE_STRATEGY_LOCK`). On older Ray versions where `HASH_SHUFFLE` is unavailable, the strategy swap is skipped silently.
- Repartitions via `ds.repartition(num_dbs, shuffle=True, keys=[DB_ID_COL])`.
- Writes per-partition via `map_batches(..., batch_format="pandas")`.

## Single-DB variant

`write_single_db` (`shardyfusion/writer/ray/single_db_writer.py:68`) enforces `config.num_dbs == 1`. Uses `ds.sort(key_col)` for global sort.

## Functional / Non-functional properties

- Atomic two-phase publish (same as all writers).
- One Ray task per shard after repartition.
- The shuffle-strategy swap is **process-global**: don't run two unrelated Ray Data pipelines in the same process during the build.

## Guarantees

- Successful return ⇒ manifest + `_CURRENT` published.
- `verify_routing=True` catches routing drift.

## Weaknesses

- **Process-global Ray Data context mutation** during the build; documented at `writer/ray/writer.py:79, 295-304`.
- **`write_vector_sharded` not re-exported.** Lives at `shardyfusion/writer/ray/writer.py:652`; import from the module.
- **No exposed scheduler / parallelism knob** beyond `num_dbs`. Controlled by Ray cluster sizing.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Routing mismatch | `ShardAssignmentError` ("Ray/Python routing mismatch") | Bug in routing change. Don't disable `verify_routing`. |
| Task failure | Ray retries per its own policy; then `ShardCoverageError` | Configure Ray retries + `config.shard_retry`. |
| Manifest / `_CURRENT` publish | `PublishManifestError` / `PublishCurrentError` | Transient; rerun. |

## Reading the snapshot

The reader is **decoupled from the writer flavor** — no Ray dependency at read time. A snapshot built with the Ray writer reads identically to one built with Python/Spark/Dask.

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

- [`build-ray-sqlite.md`](build-ray-sqlite.md).
- [`build-spark-slatedb.md`](build-spark-slatedb.md), [`build-dask-slatedb.md`](build-dask-slatedb.md).
- [`architecture/writer-core.md`](../architecture/writer-core.md).
