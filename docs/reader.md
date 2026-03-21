# Reader Side

## Overview

The reader side provides three classes for looking up keys across sharded SlateDB snapshots:

- **`ShardedReader`** — non-thread-safe, for single-threaded services
- **`ConcurrentShardedReader`** — thread-safe with lock or pool concurrency modes
- **`AsyncShardedReader`** — for asyncio services (FastAPI, aiohttp, etc.)

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
    max_workers=8,              # 8 reader copies per shard
    pool_checkout_timeout=30.0, # seconds to wait for an available reader
)
```

Best for: high concurrency or when individual reads have non-trivial latency. Trades memory for throughput.

When all `max_workers` reader copies for a shard are checked out, the next thread blocks for up to `pool_checkout_timeout` seconds (default 30). If the timeout expires, `PoolExhaustedError` is raised. Tune this value based on your expected read latency and concurrency level.

**Guidance on `max_workers`:** In lock mode, `max_workers` controls the `ThreadPoolExecutor` used by `multi_get` to read from multiple shards in parallel. In pool mode, it also controls how many reader copies are created per shard. Start with 4 and tune based on observed contention.

### Parameter constraints

Both reader classes validate their parameters at construction time:

- `max_workers` must be a positive integer (`>= 1`), or `None` (default)
- `pool_checkout_timeout` must be `> 0` (only used in pool mode)

Invalid values raise `ValueError` immediately.

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

## Custom Manifest Store

For manifests not stored in the default S3 JSON format, provide a `ManifestStore`:

```python
reader = ShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    manifest_store=my_custom_store,  # implements ManifestStore protocol
)
```

## Snapshot Info

Both readers expose metadata about the current snapshot:

```python
info = reader.snapshot_info()
print(info.run_id, info.num_dbs, info.sharding, info.manifest_ref)
```

## Shard Metadata and Routing

Both readers provide methods to inspect shard metadata and routing without performing database reads:

```python
# Per-shard summary (returns list[ShardDetail])
for shard in reader.shard_details():
    print(shard.db_id, shard.row_count, shard.min_key, shard.max_key, shard.db_url)

# Which shard does a key route to?
db_id = reader.route_key(42)

# Full shard metadata for a key (returns RequiredShardMeta)
meta = reader.shard_for_key(42)
print(meta.db_id, meta.db_url, meta.row_count)

# Batch: mapping of keys to their shard metadata
mapping = reader.shards_for_keys([1, 42, 100])
```

## Direct Shard Access

For advanced use cases, you can borrow a raw read handle for a specific shard.
The returned `ShardReaderHandle` delegates to the underlying shard database
but does **not** close the database when you call `close()` — it only releases
the borrow (and decrements the refcount / borrow count).

`ShardReaderHandle` supports the context manager protocol for automatic cleanup:

```python
with reader.reader_for_key(42) as handle:
    raw = handle.get(key_bytes)           # single raw bytes lookup
    batch = handle.multi_get([k1, k2])    # batch lookup on one shard

# Batch variant: one handle per unique key
handles = reader.readers_for_keys([1, 42, 100])
try:
    for key, h in handles.items():
        print(key, h.get(encode(key)))
finally:
    for h in handles.values():
        h.close()
```

!!! warning
    Always close borrowed handles (or use `with`). For both reader variants,
    each handle holds a borrow/refcount increment — unclosed handles prevent
    cleanup of old state after `refresh()`.

## Reader Health

All readers expose a `health()` method that returns a diagnostic snapshot:

```python
health = reader.health()
print(health.status)              # "healthy", "degraded", or "unhealthy"
print(health.manifest_ref)        # current manifest reference
print(health.manifest_age_seconds)  # seconds since manifest was published
print(health.num_shards)          # number of active shards
print(health.is_closed)           # whether the reader has been closed
```

`ReaderHealth` fields:

| Field | Type | Description |
|---|---|---|
| `status` | `Literal["healthy", "degraded", "unhealthy"]` | Overall reader health |
| `manifest_ref` | `str` | Reference to the currently loaded manifest |
| `manifest_age_seconds` | `float` | Age of the manifest in seconds |
| `num_shards` | `int` | Number of shards in the current snapshot |
| `is_closed` | `bool` | Whether the reader has been closed |

Use `staleness_threshold_s` to control when a manifest is considered stale (which affects the `status` field):

```python
health = reader.health(staleness_threshold_s=300.0)  # 5 minutes
```

### Health Check Integration

```python
from fastapi import FastAPI
from shardyfusion import ConcurrentShardedReader

app = FastAPI()
reader = ConcurrentShardedReader(...)

@app.get("/health")
def health_check():
    h = reader.health(staleness_threshold_s=600.0)
    status_code = 200 if h.status == "healthy" else 503
    return {"status": h.status, "manifest_age_s": h.manifest_age_seconds}, status_code
```

!!! warning
    Calling `health()` on a closed reader raises `ReaderStateError`.

## Cold-Start Fallback

If the latest manifest is malformed (e.g., partial write, schema migration in progress), the reader can fall back to a previous valid manifest instead of failing outright.

```mermaid
flowchart TD
    A["load_current()"] --> B["load_manifest(ref)"]
    B -->|Success| C["Open shard readers"]
    B -->|ManifestParseError| D{"max_fallback_attempts > 0?"}
    D -->|No| E["Raise ManifestParseError"]
    D -->|Yes| F["list_manifests(limit=N+1)"]
    F --> G["Try each manifest\n(newest first, skip current)"]
    G -->|Found valid| C
    G -->|All invalid| E
```

### On Cold Start

When a reader is first constructed and `load_manifest()` raises `ManifestParseError`, the reader walks backward through `list_manifests()` to find a valid manifest:

```python
reader = ConcurrentShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    max_fallback_attempts=3,  # default: try up to 3 previous manifests
)
```

Set `max_fallback_attempts=0` to disable fallback and fail immediately on a malformed manifest.

### On Refresh

During `refresh()`, if the new manifest is malformed, the reader silently skips it and keeps its current good state. `refresh()` returns `False` in this case — no error is raised.

```python
changed = reader.refresh()  # False if new manifest was malformed → kept old state
```

This applies to all reader variants: `ShardedReader`, `ConcurrentShardedReader`, and `AsyncShardedReader`.

## Reader-Side Rate Limiting

All reader constructors accept a `rate_limiter` parameter to throttle read operations:

```python
from shardyfusion import ConcurrentShardedReader, TokenBucket

limiter = TokenBucket(rate=1000.0)  # 1000 reads/sec

reader = ConcurrentShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    rate_limiter=limiter,
)

# Each get() and multi_get() call acquires tokens before reading
value = reader.get(42)  # blocks if rate limit exceeded
```

### Async Reader Rate Limiting

`AsyncShardedReader` uses `acquire_async()` (based on `asyncio.sleep`) — it never blocks the event loop:

```python
from shardyfusion import AsyncShardedReader, TokenBucket

limiter = TokenBucket(rate=1000.0)

reader = await AsyncShardedReader.open(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    rate_limiter=limiter,
)

value = await reader.get(42)  # uses asyncio.sleep if throttled
```

!!! note
    The rate limiter uses `try_acquire()` (pure arithmetic, guaranteed non-blocking) paired with `asyncio.sleep` for the wait — no `asyncio.to_thread()` delegation required.

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

## AsyncShardedReader

For asyncio-based services (FastAPI, aiohttp, etc.), `AsyncShardedReader` provides a fully async interface that avoids blocking the event loop.

### Installation

```bash
# Minimal (default SlateDB backend, uses asyncio.to_thread for S3 calls)
uv sync --extra read

# With native async S3 (recommended, default SlateDB backend)
uv sync --extra read-async

# SQLite backend
uv sync --extra read-sqlite

# SQLite backend with APSW range reads
uv sync --extra read-sqlite-range
```

When the `read-async` extra is installed, aiobotocore is available and `AsyncShardedReader` automatically uses `AsyncS3ManifestStore` for native async S3 I/O. Without it, S3 calls are delegated to threads via `asyncio.to_thread()`.

### Basic Usage

```python
from shardyfusion import AsyncShardedReader

# Use the async factory (since __init__ can't do async I/O)
reader = await AsyncShardedReader.open(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
)

value = await reader.get(123)
values = await reader.multi_get([1, 2, 3])

changed = await reader.refresh()
await reader.close()
```

### Async Context Manager

```python
async with await AsyncShardedReader.open(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
) as reader:
    value = await reader.get(123)
```

### Borrow Safety During Reads

`get()` and `multi_get()` hold borrow count increments on the underlying reader state for the duration of each read. This prevents `refresh()` or `close()` from closing shard readers while a read is in progress. The borrow is automatically released when the read completes (or raises).

### Concurrency Control

`multi_get` fans out reads across shards using `asyncio.TaskGroup`. To limit the number of concurrent shard reads, pass `max_concurrency`:

```python
reader = await AsyncShardedReader.open(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
    max_concurrency=4,  # at most 4 concurrent shard reads
)
```

### Sync Metadata Methods

Routing and metadata methods are synchronous (no I/O involved):

```python
info = reader.snapshot_info()
db_id = reader.route_key(42)
meta = reader.shard_for_key(42)
details = reader.shard_details()
```

### Direct Shard Access

Borrow async handles for direct shard access:

```python
async with reader.reader_for_key(42) as handle:
    raw = await handle.get(key_bytes)
    batch = await handle.multi_get([k1, k2])
```

### Custom Async Reader Factory

```python
reader = await AsyncShardedReader.open(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    reader_factory=my_async_factory,  # implements AsyncShardReaderFactory
)
```

The factory must be an async callable returning an object with `async def get(key: bytes) -> bytes | None` and `async def close() -> None`.
