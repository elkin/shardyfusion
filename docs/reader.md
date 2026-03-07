# Reader Side

## Overview

The reader side provides two classes for looking up keys across sharded SlateDB snapshots:

- **`ShardedReader`** — non-thread-safe, for single-threaded services
- **`ConcurrentShardedReader`** — thread-safe with lock or pool concurrency modes

Both readers follow the same lifecycle:

1. Load the `_CURRENT` pointer from S3
2. Dereference the manifest JSON it points to
3. Open one shard reader per `db_id`
4. Build a `SnapshotRouter` that maps keys to shard IDs (mirroring write-time sharding)

## ShardedReader vs ConcurrentShardedReader

| Feature | ShardedReader | ConcurrentShardedReader |
|---|---|---|
| Thread-safe | No | Yes |
| Locking overhead | None | Per-shard lock or pool checkout |
| Reference counting | No | Yes (safe refresh under load) |
| Best for | Single-threaded services, scripts | Multi-threaded web servers, gRPC services, CLI |

Use `ShardedReader` when your application is single-threaded or you manage concurrency externally. Use `ConcurrentShardedReader` when multiple threads may call `get()` / `multi_get()` concurrently.

## Basic Usage

```python
from shardyfusion import ShardedReader

reader = ShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
)

value = reader.get(123)
values = reader.multi_get([1, 2, 3])

changed = reader.refresh()
reader.close()
```

Both readers support the context manager protocol:

```python
with ShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
) as reader:
    value = reader.get(123)
```

## Thread-Safety Modes

`ConcurrentShardedReader` supports two concurrency modes via the `thread_safety` parameter:

### Lock Mode (default)

One reader instance per shard, protected by a `threading.Lock`. Concurrent threads serialize on the lock for each shard.

```python
from shardyfusion import ConcurrentShardedReader

reader = ConcurrentShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
    thread_safety="lock",
    max_workers=4,  # for multi_get parallelism
)
```

Best for: moderate concurrency where read latency is low relative to lock contention.

### Pool Mode

Multiple reader instances per shard (`max_workers` copies), managed via a checkout pool. Concurrent threads get their own reader — no lock contention.

```python
reader = ConcurrentShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
    thread_safety="pool",
    max_workers=8,  # 8 reader copies per shard
)
```

Best for: high concurrency or when individual reads have non-trivial latency. Trades memory for throughput.

**Guidance on `max_workers`:** In lock mode, `max_workers` controls the `ThreadPoolExecutor` used by `multi_get` to read from multiple shards in parallel. In pool mode, it also controls how many reader copies are created per shard. Start with 4 and tune based on observed contention.

## Refresh Semantics

Both readers support `refresh()` to pick up new snapshots without restarting:

```python
changed = reader.refresh()  # True if a newer snapshot was loaded
```

### How refresh works

1. Load the `_CURRENT` pointer from S3
2. Compare the `manifest_ref` to the current state
3. If unchanged, return `False` (no-op)
4. Load the new manifest and open new shard readers
5. Swap in the new state and close old readers

### Concurrent refresh safety

`ConcurrentShardedReader` uses reference counting to ensure safe refresh under concurrent reads:

```mermaid
sequenceDiagram
    participant T1 as Thread 1 (get)
    participant R as Reader
    participant T2 as Thread 2 (refresh)

    T1->>R: acquire_state (refcount=1)
    T1->>R: read from shard
    T2->>R: refresh() — swap state
    Note over R: old state marked retired, refcount=1
    T1->>R: release_state (refcount=0)
    Note over R: refcount=0 + retired → close old readers
    T2->>R: new state active
```

In-flight reads always complete against the state they acquired. Old readers are only closed when the last reference is released (refcount drops to 0).

## Custom Reader Factory

By default, readers use `SlateDbReaderFactory` which creates `SlateDBReader` instances. You can provide a custom factory for testing or alternative storage:

```python
from pathlib import Path
from shardyfusion import ShardedReader

def my_factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
    return MyCustomReader(db_url, local_dir, checkpoint_id)

reader = ShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    reader_factory=my_factory,
)
```

The factory must conform to the `ShardReaderFactory` protocol — a callable accepting `db_url`, `local_dir`, and `checkpoint_id` keyword arguments, returning an object with `get(key: bytes) -> bytes | None` and `close()` methods.

## Custom Manifest Reader

For manifests not stored in the default S3 JSON format, provide a `ManifestReader`:

```python
from shardyfusion.manifest_readers import FunctionManifestReader

reader = ShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    manifest_reader=FunctionManifestReader(
        load_current_fn=my_load_current,
        load_manifest_fn=my_load_manifest,
    ),
)
```

## Snapshot Info

Both readers expose metadata about the current snapshot:

```python
info = reader.snapshot_info()
print(info.run_id, info.num_dbs, info.sharding, info.manifest_ref)
```

## Metrics

Pass a `MetricsCollector` to observe reader lifecycle events:

```python
reader = ShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    metrics_collector=my_collector,
)
```

Events emitted: `READER_INITIALIZED`, `READER_GET`, `READER_MULTI_GET`, `READER_REFRESHED`, `READER_CLOSED`.
