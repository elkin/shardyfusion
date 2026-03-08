# Design: Database-backed manifests and multi-engine shard fusion

## Executive Summary

Two independent extensions to shardyfusion:

1. **Database-backed manifest storage** — Store manifests in RDBMS tables instead of (or alongside) S3 `_CURRENT` files. Readers poll or subscribe via CDC.
2. **ShardFuser abstraction** — A minimal interface that builds a unified, engine-native view over sharded databases. Returns a DuckDB connection, SQLite connection, or SlateDB-style KV reader — not a fixed `get()`/`multi_get()` API. Plus a standalone Router API for direct shard access.

---

## 1. Current state (as of main)

### Manifest lifecycle

```
Writer → ManifestPublisher.publish_manifest() → S3 manifest file
       → ManifestPublisher.publish_current()  → S3 _CURRENT pointer

Reader → ManifestReader.load_current()        → CurrentPointer
       → ManifestReader.load_manifest(ref)    → ParsedManifest
       → builds SnapshotRouter + opens ShardReaders
```

`ManifestPublisher` and `ManifestReader` are both Protocols. Custom implementations can already point at non-S3 storage. But there's no built-in database-backed implementation.

### Reader architecture

```python
class ShardReader(Protocol):
    def get(self, key: bytes) -> bytes | None: ...
    def close(self) -> None: ...

class ShardReaderFactory(Protocol):
    def __call__(self, *, db_url: str, local_dir: Path,
                 checkpoint_id: str | None) -> ShardReader: ...
```

Both `ShardedReader` and `ConcurrentShardedReader` are locked into `ShardReader` (KV-only). There's no way to get a DuckDB connection, run SQL, or get a unified view.

### Writer architecture

`DbAdapter` / `DbAdapterFactory` protocols in `slatedb_adapter.py` — already engine-agnostic in principle:
```python
class DbAdapter(Protocol):
    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None: ...
    def flush(self) -> None: ...
    def checkpoint(self) -> str | None: ...
    def close(self) -> None: ...
```

All four writers (Spark, Ray, Dask, Python) use this. A SQLite or DuckDB adapter would plug in here.

---

## 2. Database-backed manifest storage

### Problem

S3 `_CURRENT` works but has limitations:
- No history / audit trail (each publish overwrites)
- No subscription mechanism (readers must poll S3)
- No transactional guarantees across manifest + pointer updates
- Organizations with RDBMS infrastructure prefer database-native manifest management

### Design

#### New protocol implementations (not new protocols)

The existing `ManifestPublisher` and `ManifestReader` protocols are sufficient. We add database-backed implementations:

**`shardyfusion/manifest_stores/rdbms.py`**:

```python
@dataclass(slots=True)
class RdbmsManifestPublisher:
    """Publish manifests to an RDBMS table.

    Table schema:
        manifests(
            run_id        TEXT PRIMARY KEY,
            manifest_ref  TEXT NOT NULL,       -- S3 URL or inline JSON
            content_type  TEXT NOT NULL,
            payload       TEXT,                -- inline manifest JSON (optional)
            created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_current    BOOLEAN NOT NULL DEFAULT FALSE
        )
    """

    connection_url: str             # sqlalchemy-style or raw DSN
    table_name: str = "manifests"
    inline_manifest: bool = False   # if True, store payload in table too

    def publish_manifest(self, *, name: str, artifact: ManifestArtifact,
                         run_id: str) -> str:
        # INSERT manifest row
        # If inline_manifest: store payload in `payload` column
        # Otherwise: just store the S3 ref (caller uploads to S3 separately)
        # Return the manifest_ref (S3 URL or "db://{table}/{run_id}")
        ...

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        # UPDATE manifests SET is_current=FALSE WHERE is_current=TRUE
        # UPDATE manifests SET is_current=TRUE WHERE run_id=...
        # (single transaction)
        ...


@dataclass(slots=True)
class RdbmsManifestReader:
    """Read manifests from an RDBMS table."""

    connection_url: str
    table_name: str = "manifests"

    def load_current(self) -> CurrentPointer | None:
        # SELECT manifest_ref, content_type, run_id, created_at
        # FROM manifests WHERE is_current = TRUE
        # Returns CurrentPointer or None
        ...

    def load_manifest(self, ref: str, content_type: str | None = None) -> ParsedManifest:
        # If ref starts with "db://": load from payload column
        # If ref starts with "s3://": delegate to S3 fetch
        # Parse JSON → ParsedManifest
        ...
```

#### CDC / polling support

For readers that want push-based updates:

```python
@dataclass(slots=True)
class PollingManifestWatcher:
    """Periodically polls the manifest reader and calls back on change."""

    reader: ManifestReader
    interval_seconds: float = 30.0
    on_change: Callable[[CurrentPointer], None] | None = None

    def start(self) -> None: ...   # background thread
    def stop(self) -> None: ...
```

CDC integration (Debezium, pg_notify, etc.) is left to users — they implement `ManifestReader` and call `reader.refresh()` when notified.

#### `load_current()` semantics

With RDBMS storage, `load_current()` becomes "query the latest manifest" — no `_CURRENT` file needed. The `_CURRENT` concept becomes an implementation detail of `DefaultS3ManifestReader`, not a protocol requirement.

### Files to create

| File | Purpose |
|------|---------|
| `shardyfusion/manifest_stores/__init__.py` | Package |
| `shardyfusion/manifest_stores/rdbms.py` | `RdbmsManifestPublisher`, `RdbmsManifestReader` |
| `tests/unit/shared/test_rdbms_manifest.py` | Unit tests (SQLite in-memory as RDBMS stand-in) |

### Dependencies

```toml
[project.optional-dependencies]
rdbms = ["sqlalchemy>=2.0"]
```

Or keep it dependency-free by accepting a raw `Connection` object (stdlib `sqlite3.Connection` or `psycopg2.connection`) rather than using SQLAlchemy. Decision: start with raw `sqlite3` for tests and a generic DB-API 2.0 `Connection` protocol, add SQLAlchemy later if needed.

---

## 3. ShardFuser: unified view over sharded databases

### Problem

The current reader returns `bytes | None` per key. This is too narrow:
- DuckDB users want a DuckDB connection that can query across all shards with SQL
- SQLite users want to ATTACH all shards and run cross-shard queries
- SlateDB users are served by the current KV reader, but should also be accessible via the same pattern
- Users shouldn't need to understand routing internals to query a specific shard

### Design: The ShardFuser protocol

A `ShardFuser` has **one method**: it takes a manifest and returns a native engine handle.

```python
# shardyfusion/fuser.py

from typing import Protocol, TypeVar

T = TypeVar("T")


class ShardFuser(Protocol[T]):
    """Build a unified, engine-native view over sharded databases.

    The returned handle is engine-specific:
    - DuckDB: duckdb.DuckDBPyConnection
    - SQLite: sqlite3.Connection (with all shards ATTACHed)
    - SlateDB: ShardedKvView (thin wrapper with get/multi_get)
    """

    def fuse(
        self,
        manifest: ParsedManifest,
        manifest_ref: str,
        *,
        local_root: str | Path,
    ) -> T:
        """Build and return a unified view. Caller owns closing."""
        ...

    def close_view(self, view: T) -> None:
        """Release resources held by a fused view."""
        ...
```

#### Why generic `T` instead of a base class?

Because DuckDB connections, SQLite connections, and KV readers have completely different APIs. Forcing them into a common base destroys the value — the whole point is to get a *native* handle. The `ShardFuser` is generic over its return type.

### Built-in fusers

#### DuckDB fuser

```python
# shardyfusion/fusers/duckdb.py

@dataclass(slots=True)
class DuckDbFuser:
    """Fuse sharded DuckDB databases into a single queryable connection.

    Uses DuckDB's httpfs extension to read directly from S3,
    or downloads locally first. Creates UNION ALL views across shards.
    """

    s3_client_config: S3ClientConfig | None = None
    mode: Literal["remote", "local"] = "remote"  # remote = httpfs, local = download

    def fuse(
        self,
        manifest: ParsedManifest,
        manifest_ref: str,
        *,
        local_root: str | Path,
    ) -> "duckdb.DuckDBPyConnection":
        import duckdb

        conn = duckdb.connect()

        if self.mode == "remote":
            # Configure httpfs for S3 access
            self._configure_httpfs(conn)
            for shard in manifest.shards:
                conn.execute(
                    f"ATTACH '{shard.db_url}' AS shard_{shard.db_id:05d} (READ_ONLY)"
                )
        else:
            # Download each shard locally, then attach
            for shard in manifest.shards:
                local_path = Path(local_root) / f"shard={shard.db_id:05d}" / "shard.db"
                download_file(shard.db_url, local_path, ...)
                conn.execute(
                    f"ATTACH '{local_path}' AS shard_{shard.db_id:05d} (READ_ONLY)"
                )

        # Create unified view across all shards
        shard_selects = [
            f"SELECT *, {s.db_id} AS _shard_id FROM shard_{s.db_id:05d}.main.kv"
            for s in manifest.shards
        ]
        conn.execute(
            f"CREATE VIEW unified AS {' UNION ALL '.join(shard_selects)}"
        )

        return conn

    def close_view(self, view: "duckdb.DuckDBPyConnection") -> None:
        view.close()
```

Usage:
```python
fuser = DuckDbFuser(s3_client_config={...})
manifest_reader = DefaultS3ManifestReader(s3_prefix)
current = manifest_reader.load_current()
manifest = manifest_reader.load_manifest(current.manifest_ref)

conn = fuser.fuse(manifest, current.manifest_ref, local_root="/tmp/shards")

# Now you have a native DuckDB connection — do whatever you want
result = conn.sql("SELECT * FROM unified WHERE key > 1000 LIMIT 10").fetchall()
arrow_table = conn.sql("SELECT * FROM unified").arrow()
df = conn.sql("SELECT * FROM unified").df()

fuser.close_view(conn)
```

#### SQLite fuser

```python
# shardyfusion/fusers/sqlite.py

@dataclass(slots=True)
class SqliteFuser:
    """Fuse sharded SQLite databases into a single connection via ATTACH."""

    s3_client_config: S3ClientConfig | None = None

    def fuse(
        self,
        manifest: ParsedManifest,
        manifest_ref: str,
        *,
        local_root: str | Path,
    ) -> "sqlite3.Connection":
        import sqlite3

        conn = sqlite3.connect(":memory:")

        for shard in manifest.shards:
            local_path = Path(local_root) / f"shard={shard.db_id:05d}" / "shard.db"
            download_file(shard.db_url, local_path, ...)
            conn.execute(
                f"ATTACH DATABASE ? AS shard_{shard.db_id:05d}",
                (str(local_path),)
            )

        # Create unified view
        shard_selects = [
            f"SELECT *, {s.db_id} AS _shard_id FROM shard_{s.db_id:05d}.kv"
            for s in manifest.shards
        ]
        conn.execute(
            f"CREATE VIEW unified AS {' UNION ALL '.join(shard_selects)}"
        )

        return conn

    def close_view(self, view: "sqlite3.Connection") -> None:
        view.close()
```

#### SlateDB KV fuser

The existing reader is essentially a fuser that returns a KV-specific view. We can wrap it:

```python
# shardyfusion/fusers/kv.py

@dataclass(slots=True)
class ShardedKvView:
    """Thin KV wrapper returned by KvFuser. Delegates to router + shard readers."""
    router: SnapshotRouter
    readers: dict[int, ShardReader]

    def get(self, key: KeyInput) -> bytes | None:
        db_id = self.router.route_one(key)
        key_bytes = self.router.encode_lookup_key(key)
        return self.readers[db_id].get(key_bytes)

    def multi_get(self, keys: Sequence[KeyInput]) -> dict[KeyInput, bytes | None]:
        grouped = self.router.group_keys(list(keys))
        results: dict[KeyInput, bytes | None] = {}
        for db_id, shard_keys in grouped.items():
            for key in shard_keys:
                key_bytes = self.router.encode_lookup_key(key)
                results[key] = self.readers[db_id].get(key_bytes)
        return results


@dataclass(slots=True)
class KvFuser:
    """Fuse sharded SlateDB databases into a KV view."""
    reader_factory: ShardReaderFactory | None = None
    env_file: str | None = None

    def fuse(self, manifest: ParsedManifest, manifest_ref: str,
             *, local_root: str | Path) -> ShardedKvView:
        factory = self.reader_factory or SlateDbReaderFactory(env_file=self.env_file)
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        readers: dict[int, ShardReader] = {}
        for shard in manifest.shards:
            local_path = Path(local_root) / f"shard={shard.db_id:05d}"
            local_path.mkdir(parents=True, exist_ok=True)
            readers[shard.db_id] = factory(
                db_url=shard.db_url, local_dir=local_path,
                checkpoint_id=shard.checkpoint_id,
            )
        return ShardedKvView(router=router, readers=readers)

    def close_view(self, view: ShardedKvView) -> None:
        for reader in view.readers.values():
            reader.close()
```

### Can the fused view do writes?

Yes, for file-based engines. DuckDB and SQLite support read-write connections.

For DuckDB:
```python
# After fuse(), the connection can write to individual shards:
conn.execute("INSERT INTO shard_00002.kv VALUES (?, ?)", (key, value))
```

For SQLite:
```python
conn.execute("INSERT INTO shard_00002.kv VALUES (?, ?)", (key, value))
```

The fuser doesn't need write methods — the native connection handles it. Users who want to write need to:
1. Know which shard a key routes to (→ Router API, see below)
2. Use the native connection to write to that shard's table

### Convenience: FusedReader (fuser + manifest lifecycle)

For users who don't want to manage manifests manually:

```python
# shardyfusion/fused_reader.py

class FusedReader(Generic[T]):
    """Combines ManifestReader + ShardFuser + refresh lifecycle."""

    def __init__(
        self,
        *,
        manifest_reader: ManifestReader,
        fuser: ShardFuser[T],
        local_root: str,
    ) -> None:
        self._manifest_reader = manifest_reader
        self._fuser = fuser
        self._local_root = local_root
        self._view: T | None = None
        self._manifest_ref: str | None = None

    @property
    def view(self) -> T:
        """The current fused view. Call open() first."""
        if self._view is None:
            raise ReaderStateError("FusedReader not opened; call open() first")
        return self._view

    def open(self) -> T:
        current = self._manifest_reader.load_current()
        if current is None:
            raise ReaderStateError("No CURRENT pointer found")
        manifest = self._manifest_reader.load_manifest(
            current.manifest_ref, current.manifest_content_type
        )
        self._view = self._fuser.fuse(
            manifest, current.manifest_ref, local_root=self._local_root
        )
        self._manifest_ref = current.manifest_ref
        return self._view

    def refresh(self) -> bool:
        """Reload manifest. Returns True if view changed."""
        current = self._manifest_reader.load_current()
        if current is None:
            raise ReaderStateError("No CURRENT pointer found")
        if current.manifest_ref == self._manifest_ref:
            return False

        manifest = self._manifest_reader.load_manifest(
            current.manifest_ref, current.manifest_content_type
        )
        old_view = self._view
        self._view = self._fuser.fuse(
            manifest, current.manifest_ref, local_root=self._local_root
        )
        self._manifest_ref = current.manifest_ref
        if old_view is not None:
            self._fuser.close_view(old_view)
        return True

    def close(self) -> None:
        if self._view is not None:
            self._fuser.close_view(self._view)
            self._view = None
```

Usage:
```python
reader = FusedReader(
    manifest_reader=DefaultS3ManifestReader("s3://bucket/prefix"),
    fuser=DuckDbFuser(),
    local_root="/tmp/shards",
)
conn = reader.open()  # → duckdb.DuckDBPyConnection

conn.sql("SELECT count(*) FROM unified").show()
reader.refresh()  # swaps to new snapshot
conn = reader.view  # new connection

reader.close()
```

---

## 4. Router API (standalone)

### Problem

Users want to find which shard holds a key — independent of any reader or fuser. Works with any engine.

### Design

The `SnapshotRouter` already exists and does this. We just need to make it easier to construct standalone:

```python
# Addition to shardyfusion/routing.py

@staticmethod
def from_manifest(manifest: ParsedManifest) -> SnapshotRouter:
    """Construct a router from a parsed manifest."""
    return SnapshotRouter(manifest.required_build, manifest.shards)

@staticmethod
def from_manifest_reader(manifest_reader: ManifestReader) -> SnapshotRouter:
    """Load CURRENT, parse manifest, and build a router."""
    current = manifest_reader.load_current()
    if current is None:
        raise ReaderStateError("No CURRENT pointer found")
    manifest = manifest_reader.load_manifest(
        current.manifest_ref, current.manifest_content_type
    )
    return SnapshotRouter(manifest.required_build, manifest.shards)
```

Usage:
```python
router = SnapshotRouter.from_manifest(manifest)

db_id = router.route_one(42)           # → 3
shard = router.shards[db_id]           # → RequiredShardMeta(db_url="s3://...", ...)
grouped = router.group_keys([1, 2, 3]) # → {0: [2], 3: [1, 3]}
```

Combined with a fuser:
```python
router = SnapshotRouter.from_manifest(manifest)
conn = DuckDbFuser().fuse(manifest, ref, local_root="/tmp")

# Route key → shard → query that specific shard
db_id = router.route_one(42)
result = conn.sql(f"SELECT * FROM shard_{db_id:05d}.kv WHERE key = 42").fetchone()

# Or write to the correct shard
conn.execute(f"INSERT INTO shard_{db_id:05d}.kv VALUES (?, ?)", (key_bytes, value_bytes))
```

---

## 5. Engine field in manifest

To support multi-engine scenarios, the manifest needs to declare which engine wrote the shards.

### Changes to existing models

**`shardyfusion/manifest.py`** — Add to `RequiredBuildMeta`:
```python
engine: str = "slatedb"           # "slatedb" | "sqlite" | "duckdb"
engine_meta: dict[str, Any] = Field(default_factory=dict)
```

Backward compat: existing manifests without `engine` default to `"slatedb"`.

**`shardyfusion/config.py`** — Add to `WriteConfig`:
```python
engine: str = "slatedb"
```

This tells the fuser what to expect without requiring the user to specify it at read time.

---

## 6. File-based engine writers (SQLite, DuckDB)

### File sync layer

**`shardyfusion/file_sync.py`** — S3 upload/download for engines that produce local files:

```python
def upload_file(local_path: Path, s3_url: str, *,
                s3_client_config: S3ClientConfig | None = None) -> None:
    """Upload a local file to S3. Uses multipart for files > 8MB."""

def download_file(s3_url: str, local_path: Path, *,
                  s3_client_config: S3ClientConfig | None = None) -> None:
    """Download from S3 to local path. Creates parent dirs."""
```

### SQLite adapter (writer side)

```python
# shardyfusion/engines/sqlite/adapter.py

class SqliteDbAdapter:
    """DbAdapter that writes to a local SQLite file, uploads on close."""
    # Schema: CREATE TABLE kv(key BLOB PRIMARY KEY, value BLOB) WITHOUT ROWID
    # write_batch() → executemany INSERT OR REPLACE
    # close() → WAL checkpoint → upload to S3 via file_sync
```

### DuckDB adapter (writer side)

```python
# shardyfusion/engines/duckdb/adapter.py

class DuckDbAdapter:
    """DbAdapter that writes to a local DuckDB file, uploads on close."""
    # Schema: CREATE TABLE kv(key BLOB, value BLOB)
    # write_batch() → executemany INSERT
    # close() → upload to S3 via file_sync
```

Both implement `DbAdapter` protocol — plug into any writer (Spark, Ray, Dask, Python) unchanged.

---

## 7. Relationship between existing readers and ShardFuser

The existing `ShardedReader` / `ConcurrentShardedReader` **stay as-is**. They are the production-grade KV reader with thread safety, refresh, metrics, etc.

`ShardFuser` is a **lower-level primitive** — it builds a view but doesn't manage lifecycle. `FusedReader` adds lifecycle management on top.

Over time, `ShardedReader` could be refactored to use `KvFuser` internally, but that's not necessary for the initial implementation.

```
┌─────────────────────────────────────────────────┐
│ High-level (lifecycle, thread-safety, metrics)  │
│                                                 │
│  ShardedReader    ConcurrentShardedReader        │
│  (existing KV)   (existing KV, thread-safe)     │
│                                                 │
│  FusedReader[T]   (new, generic, any engine)    │
└─────────────────────┬───────────────────────────┘
                      │ uses
┌─────────────────────▼───────────────────────────┐
│ Low-level (build view from manifest)            │
│                                                 │
│  ShardFuser[T] protocol                         │
│    ├── DuckDbFuser  → duckdb.Connection         │
│    ├── SqliteFuser  → sqlite3.Connection        │
│    └── KvFuser      → ShardedKvView             │
│                                                 │
│  SnapshotRouter (key → shard routing)           │
└─────────────────────────────────────────────────┘
```

---

## 8. Implementation order

| Phase | Scope | Effort | Breaking? |
|-------|-------|--------|-----------|
| **1: Engine field** | Add `engine`/`engine_meta` to manifest + config | Small | No (defaults to "slatedb") |
| **2: Router convenience** | `SnapshotRouter.from_manifest()` etc. | Tiny | No |
| **3: ShardFuser protocol + KvFuser** | Protocol definition + SlateDB fuser | Small | No |
| **4: File sync + SQLite engine** | `file_sync.py` + SQLite adapter + SqliteFuser | Medium | No |
| **5: DuckDB engine + fuser** | DuckDB adapter + DuckDbFuser | Medium | No |
| **6: FusedReader** | Lifecycle wrapper over ShardFuser | Small | No |
| **7: RDBMS manifest store** | `RdbmsManifestPublisher` + `RdbmsManifestReader` | Medium | No |

Phases 1-3 can land in a single PR. Phases 4-5 are independent. Phases 6-7 are independent.

---

## 9. New files summary

| File | Purpose |
|------|---------|
| `shardyfusion/fuser.py` | `ShardFuser[T]` protocol |
| `shardyfusion/fused_reader.py` | `FusedReader[T]` lifecycle wrapper |
| `shardyfusion/fusers/__init__.py` | Package |
| `shardyfusion/fusers/kv.py` | `KvFuser` + `ShardedKvView` |
| `shardyfusion/fusers/sqlite.py` | `SqliteFuser` |
| `shardyfusion/fusers/duckdb.py` | `DuckDbFuser` |
| `shardyfusion/file_sync.py` | S3 upload/download for file-based engines |
| `shardyfusion/engines/__init__.py` | Package |
| `shardyfusion/engines/sqlite/__init__.py` | SQLite engine exports |
| `shardyfusion/engines/sqlite/adapter.py` | `SqliteFactory` (writer-side `DbAdapter`) |
| `shardyfusion/engines/duckdb/__init__.py` | DuckDB engine exports |
| `shardyfusion/engines/duckdb/adapter.py` | `DuckDbFactory` (writer-side `DbAdapter`) |
| `shardyfusion/manifest_stores/__init__.py` | Package |
| `shardyfusion/manifest_stores/rdbms.py` | `RdbmsManifestPublisher` + `RdbmsManifestReader` |

### Files to modify

| File | Change |
|------|--------|
| `shardyfusion/manifest.py` | Add `engine`, `engine_meta` to `RequiredBuildMeta` |
| `shardyfusion/config.py` | Add `engine` to `WriteConfig` |
| `shardyfusion/_writer_core.py` | Pass `engine` to manifest during build |
| `shardyfusion/routing.py` | Add `from_manifest()`, `from_manifest_reader()` classmethods |
| `shardyfusion/schemas/manifest.schema.json` | Add `engine` and `engine_meta` properties |
| `shardyfusion/__init__.py` | Export new public API |
| `pyproject.toml` | Add `duckdb`, `sqlite`, `rdbms` extras |

---

## 10. Open questions

1. **DuckDB ATTACH over S3** — DuckDB's httpfs can read Parquet/CSV from S3 but cannot ATTACH a DuckDB database file over S3 (as of v1.1). The `remote` mode may need to use `download_file` regardless. Verify DuckDB S3 ATTACH support before implementing.

2. **Schema beyond KV** — The current `DbAdapter.write_batch()` takes `Iterable[tuple[bytes, bytes]]`. For DuckDB to store typed columns (not just key/value blobs), we'd need a `ColumnarDbAdapter` variant. Defer this — start with KV-in-DuckDB, which already enables SQL queries over the blob data.

3. **Unified view creation** — The `CREATE VIEW unified AS ... UNION ALL ...` approach assumes all shards have the same schema. This is guaranteed when all shards are written by the same writer, but should we validate?

4. **Write-through via fused view** — When a user inserts into a shard via the native connection, how does the manifest learn about the new row count? Answer: it doesn't — manifest-tracked metadata (row_count, min_key, max_key) becomes stale. This is acceptable for local mutations. If users need to persist writes, they should re-publish the manifest.

5. **Connection for the raw DB-API 2.0** — Should `RdbmsManifestPublisher` accept a connection factory (callable) or a connection object? A factory is more flexible (reconnect on failure) but adds complexity. Start with connection object, document that callers should handle reconnection.
