# Read a SlateDB snapshot synchronously

Use **`ShardedReader`** for point-key and multi-key lookups against a published SlateDB snapshot from synchronous code. This page is the canonical reference for the synchronous reader API — other reader pages link back here for shared functionality.

## When to use

- Synchronous service or batch process.
- SlateDB-backed snapshot.
- Single-process or thread-pool concurrency. For higher concurrency, use [`ConcurrentShardedReader`](#concurrent-variant).

## When NOT to use

- You're in async code — use [`read-async-slatedb.md`](read-async-slatedb.md).
- SQLite snapshot — use [`read-sync-sqlite.md`](read-sync-sqlite.md).

## Install

```bash
uv add 'shardyfusion[read]'
```

## Minimal example

```python
from shardyfusion import ShardedReader

reader = ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
)

value = reader.get(b"user-123")
many = reader.multi_get([b"user-1", b"user-2"])
```

## Configuration

Constructor (`shardyfusion/reader/reader.py:72`):

| Param | Default | Purpose |
|---|---|---|
| `s3_prefix` | required | Snapshot root (`s3://bucket/prefix`). |
| `local_root` | required | Local cache directory for adapter state. |
| `manifest_store` | auto | Custom store (e.g. Postgres). Defaults to S3-backed store. |
| `current_pointer_key` | `"_CURRENT"` | Pointer key in S3. |
| `reader_factory` | `SlateDbReaderFactory()` | Adapter factory; swap to `SqliteReaderFactory` / `SqliteRangeReaderFactory` for SQLite. |
| `credential_provider` | `None` | S3 credentials. |
| `s3_connection_options` | `None` | Endpoint URL, region, retries. |
| `max_workers` | `None` | Thread-pool size for multi-shard fan-out in `multi_get`. |
| `max_fallback_attempts` | `3` | If the current manifest is malformed, fall back to previous manifests. |
| `metrics_collector` | `None` | Pass a `PrometheusCollector` / `OtelCollector`. |
| `rate_limiter` | `None` | Token-bucket rate limit on `get` / `multi_get`. |

Call `reader.refresh()` to pick up newly published manifests.

## Reader API

The reader exposes more than just `get`. All of these methods are pinned to the manifest currently loaded by the reader (use `refresh()` to advance).

### Lookups

```python
# Single key
value: bytes | None = reader.get(b"user-123")

# Many keys — internally fans out across shards (uses thread pool if max_workers > 0)
results: dict[KeyInput, bytes | None] = reader.multi_get([b"user-1", b"user-2", b"user-3"])

# Routing context for CEL routing (categorical keys)
value = reader.get(b"user-1", routing_context={"region": "us-east-1"})
```

### Routing introspection — "which shard does this key live in?"

```python
# Just the shard id (db_id)
db_id: int = reader.route_key(b"user-123")

# The shard's metadata (db_id, db_url, row count, schema metadata, etc.)
from shardyfusion.manifest import RequiredShardMeta
meta: RequiredShardMeta = reader.shard_for_key(b"user-123")
print(meta.db_id, meta.db_url, meta.row_count)

# Bulk: keys → shard metadata
key_to_shard = reader.shards_for_keys([b"user-1", b"user-2"])
```

### Direct shard access — "give me the underlying DB handle for this key"

When you need to bypass the routed-`get` path and run native operations against a single shard's database (e.g. SlateDB scans, SQLite SQL queries), borrow a shard reader handle:

```python
# Borrow the shard reader for a single key
with reader.reader_for_key(b"user-123") as handle:
    raw = handle.reader.get(reader.encode_key(b"user-123"))
    # `handle.reader` is the underlying ShardReader (SlateDB or SQLite adapter)

# Borrow handles for many keys (deduplicated by shard)
handles = reader.readers_for_keys([b"user-1", b"user-2"])
for key, handle in handles.items():
    with handle:
        ...
```

The handle is **borrowed** — close it (or use `with`) so the reader can release internal counters. Concurrent variants additionally pool/reference-count the underlying readers.

> SlateDB shard handles expose `get(key_bytes)` and the SlateDB scan API. SQLite shard handles expose `get(key_bytes)` plus the underlying `sqlite3.Connection` (download-and-cache) or APSW connection (range-read) — letting you run arbitrary SQL on a single shard.

### Snapshot inspection

```python
info = reader.snapshot_info()
# SnapshotInfo(run_id=..., num_dbs=..., sharding=..., created_at=...,
#              manifest_ref=..., key_encoding=..., row_count=...)

shards = reader.shard_details()
# list[ShardDetail] with per-shard db_url, row_count, byte_size, etc.

health = reader.health(staleness_threshold=timedelta(hours=1))
# ReaderHealth(status="healthy"|"degraded"|"unhealthy", manifest_age, num_shards, ...)
```

### Refresh & lifecycle

```python
changed: bool = reader.refresh()    # picks up a new _CURRENT atomically
reader.close()                      # releases adapters and threads
```

`refresh()` is the **only** way to observe new publishes — `_CURRENT` is read once at open. Use a periodic refresh loop (or react to a notification) for long-lived services.

## Concurrent variant

`ConcurrentShardedReader` adds reference counting and an internal pool of reader copies per shard, enabling safe high-concurrency `multi_get` fan-out. Same constructor surface plus:

| Param | Default | Purpose |
|---|---|---|
| `pool_mode` | `"single"` | `"pool"` opens N reader copies per shard. |
| `pool_size` | `max_workers` | Reader copies per shard in pool mode. |
| `pool_checkout_timeout` | `30s` | Max wait for a free copy. |

The API surface is identical — `get`, `multi_get`, `route_key`, `shard_for_key`, `reader_for_key`, `refresh`, `snapshot_info`, `shard_details`, `health` all behave the same.

## Functional / Non-functional properties

- Routing computed locally from manifest — no roundtrip per lookup.
- SlateDB local cache reused across calls.
- `multi_get` deduplicates keys per shard before issuing reads.
- `refresh()` is the only way to observe new publishes.

## Guarantees

- Reads are pinned to the manifest loaded at open / last `refresh()`. No partial views.
- Routing matches the writer (`SnapshotRouter.from_build_meta`) — same key always lands on the same shard for a given snapshot.
- If `_CURRENT` points at a malformed manifest, the reader falls back to up to `max_fallback_attempts` previous manifests automatically.

## Weaknesses

- Single-process by default; for fan-out use `ConcurrentShardedReader` or [`read-async-slatedb.md`](read-async-slatedb.md).
- `_CURRENT` is read once; manual `refresh()` required to advance.
- Borrowing a shard handle bypasses metric instrumentation on the routed-`get` path.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Missing `_CURRENT` | `ReaderStateError("CURRENT pointer not found")` | Verify writer published; check `s3_prefix`. |
| Malformed manifest | `ManifestParseError`; cold-start fallback tries previous (up to `max_fallback_attempts=3`) | Investigate writer; rollback via [`operate-manifest-history-and-rollback.md`](operate-manifest-history-and-rollback.md). |
| Routing token unknown (CEL) | `UnknownRoutingTokenError` (subclass of `ValueError`) | Routing changed; `refresh()` or rebuild snapshot. |
| S3 transient | wrapped as `S3TransientError` (auto-retried) → `DbAdapterError` if exhausted | Retry; check credentials, throttling, connectivity. |
| Reader closed | `ReaderStateError` | Open a new reader. |

## See also

- [`read-async-slatedb.md`](read-async-slatedb.md) — async equivalent.
- [`read-sync-sqlite.md`](read-sync-sqlite.md) — SQLite reader (download-and-cache and remote range-read).
- [`architecture/manifest-and-current.md`](../architecture/manifest-and-current.md).
- [`architecture/observability.md`](../architecture/observability.md) — metric events emitted by the reader.
