# Read a SQLite snapshot synchronously

Use **`ShardedReader`** with a SQLite reader factory to read sharded SQLite snapshots synchronously. Two sub-flavors selected by the factory:

- **Download-and-cache** (`SqliteReaderFactory`) — the entire shard `.db` is downloaded once, then all reads are local.
- **Range-read VFS** (`SqliteRangeReaderFactory`, requires `apsw`) — SQLite issues HTTP range GETs against the S3 object; **no full download required**.

## When to use

- Snapshot was built with the SQLite adapter (any writer flavor).
- You're in synchronous code.
- Pick **range-read** when shards are large and you only touch a few keys per shard.
- Pick **download-and-cache** when shards are small and you want lowest per-key latency once warm.

## When NOT to use

- SlateDB snapshot — use [sync SlateDB](slatedb.md).
- Async code — use [async SQLite](../async/sqlite.md).

## Install

```bash
# Download-and-cache (uses stdlib sqlite3)
uv add 'shardyfusion[read-sqlite]'

# Range-read VFS (requires apsw)
uv add 'shardyfusion[read-sqlite-range]'
```

## Minimal example

### Download-and-cache

```python
from shardyfusion import ShardedReader
from shardyfusion.sqlite_adapter import SqliteReaderFactory

reader = ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteReaderFactory(),
)

value = reader.get(b"user-123")
many = reader.multi_get([b"user-1", b"user-2"])
```

### Range-read VFS

```python
from shardyfusion.sqlite_adapter import SqliteRangeReaderFactory

reader = ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/users",
    local_root="/var/cache/shardy/users",
    reader_factory=SqliteRangeReaderFactory(page_cache_pages=1024),
)
```

The range-read VFS translates each SQLite page fetch (default 4 KiB) into an S3 `Range` request, with an LRU page cache in memory.

## Configuration

The reader constructor and full API surface (`get`, `multi_get`, `route_key`, `shard_for_key`, `reader_for_key`, `refresh`, `snapshot_info`, `shard_details`, `health`) are identical to the SlateDB variant — see [sync SlateDB](slatedb.md#reader-api). The only thing that changes is the `reader_factory`.

`SqliteRangeReaderFactory` extra parameters:

| Field | Default | Purpose |
|---|---|---|
| `page_cache_pages` | `1024` | Page-cache LRU size (~4 MiB). `0` disables caching. |
| `s3_connection_options` | `None` | Endpoint URL, region, retries. |
| `credential_provider` | `None` | S3 credentials. |

## Single-shard access (SQL & scans)

Borrow a shard handle to run arbitrary SQL on a single shard:

```python
with reader.reader_for_key(b"user-123") as handle:
    cur = handle.reader.connection.cursor()
    cur.execute("SELECT count(*) FROM kv WHERE k LIKE ?", (b"user-%",))
    print(cur.fetchone())
```

Both download-and-cache and range-read expose the underlying `sqlite3.Connection` or APSW connection.

You can also enumerate shards:

```python
for shard in reader.shard_details():
    print(shard.db_id, shard.db_url, shard.row_count)
```

The `db_url` is a stable S3 URL — you can download it independently or open it in any SQLite tool.

## Functional properties

- **Download-and-cache**: first access pulls the whole shard `.db` to `local_root`; subsequent reads are local-disk-fast.
- **Range-read VFS**: per-page S3 GETs with an LRU cache; cold-cache latency higher but no full-shard download.
- **Same routed API** as SlateDB (`get`, `multi_get`, `refresh`, etc.).
- **Direct SQL** on any shard via `reader_for_key().reader.connection`.

## Non-functional properties

| | Download-and-cache | Range-read |
|---|---|---|
| **Disk usage** | `sum(shard_size)` once warm | ~`page_cache_pages × 4 KiB` memory; ~zero disk |
| **Latency (cold)** | Full shard download | One S3 GET per page (~tens of ms) |
| **Latency (warm)** | Local-disk SQLite | Memory hit |
| **S3 GET cost** | One GET per shard | Amplifies on scans; prefer for point lookups |

## Guarantees

- Same routing/snapshot pinning guarantees as SlateDB. See [sync SlateDB](slatedb.md#guarantees).
- Both reader strategies see the same `.db` file — the byte content is identical; only the access path differs.

## Weaknesses

- Download-and-cache: cold-shard latency proportional to shard size; disk pressure proportional to total snapshot size.
- Range-read: higher per-key latency, more S3 GETs, requires `apsw` (LGPL).
- SQLite has no built-in compression — shard size = raw `(key, value)` bytes plus index overhead.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Disk full (download-and-cache) | `DbAdapterError` | Free disk or switch to range-read. |
| Range-read 416 / connection drop | `DbAdapterError` (wrapped `S3VfsError`) | Retry; check S3 connectivity. |
| Missing `_CURRENT` / malformed manifest | Same as SlateDB — see [sync SlateDB](slatedb.md#failure-modes-recovery). |
| `apsw` not installed for range-read | `ImportError` at factory call | `uv add 'shardyfusion[read-sqlite-range]'`. |

## See also

- [KV Storage Overview](../../overview.md)
- [Sync SlateDB](slatedb.md) — full reader API surface (shared with this page)
- [Async SQLite](../async/sqlite.md) — async equivalent
- [`architecture/adapters.md`](../../../../architecture/adapters.md) — SQLite adapter internals
