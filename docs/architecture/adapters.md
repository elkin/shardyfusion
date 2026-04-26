# Adapters

An **adapter** is the per-shard storage driver. It owns one shard's database file (or directory) and implements `DbAdapter` (`shardyfusion/slatedb_adapter.py:20`) on the writer side and per-backend reader protocols on the reader side. Adapters are the only place where backend-specific I/O lives.

## Writer protocol

`DbAdapter` (`slatedb_adapter.py:20`) is the writer-side interface — a `Protocol` requiring:

- `__enter__` / `__exit__` — context-manager lifecycle.
- `write_batch(pairs)` (`:36`) — write a batch of `(key, value)` pairs.
- `flush()` (`:40`) — finalize an in-progress batch.
- `checkpoint() -> str | None` (`:44`) — durably persist the shard and return its canonical URL (used for winner selection tiebreak and manifest building).
- `close()` (`:48`).

Adapters are constructed by a `DbAdapterFactory` (`slatedb_adapter.py:53`) — a `Protocol` whose `__call__(*, db_url, local_dir)` returns a `DbAdapter`. Factories are dependency-injected at the writer entry point so the same `_writer_core` works against any backend.

Of the built-in factories, only `SlateDbFactory` is in the public `__all__` (`shardyfusion/__init__.py`). SQLite, sqlite-vec, and composite factories are imported from their respective modules directly.

## Built-in writer adapters

| Backend | Factory | Adapter | Module |
|---|---|---|---|
| SlateDB | `SlateDbFactory` | `DefaultSlateDbAdapter` | `slatedb_adapter.py:67`, `:84` |
| SQLite | `SqliteFactory` | `SqliteAdapter` | `sqlite_adapter.py:59`, `:78` |
| sqlite-vec (KV+vector) | `SqliteVecFactory` | `SqliteVecAdapter` | `sqlite_vec_adapter.py:106`, `:134` |
| Composite (KV + vector) | `CompositeFactory` | `CompositeAdapter` | `composite_adapter.py:69`, `:111` |

The composite adapter wires a KV adapter (SlateDB or SQLite) together with a vector adapter (LanceDB) into a single logical shard. See [`use-cases/kv-vector/build/composite.md`](../use-cases/kv-vector/build/composite.md).

## Reader protocols

Readers are split into sync and async, and into "download-and-cache" vs "range-read" strategies for SQLite. Each backend exposes a *factory* that yields a per-shard reader.

| Backend | Mode | Sync factory | Async factory |
|---|---|---|---|
| SlateDB | native | (built into `reader/reader.py`) | (`reader/async_reader.py`) |
| SQLite | download-and-cache | `SqliteReaderFactory` (`sqlite_adapter.py:230`) | `AsyncSqliteReaderFactory` (`sqlite_adapter.py:334`) |
| SQLite | range-read VFS (APSW) | `SqliteRangeReaderFactory` (`sqlite_adapter.py:386`) | `AsyncSqliteRangeReaderFactory` (`sqlite_adapter.py:558`) |
| sqlite-vec | download-and-cache | `SqliteVecReaderFactory` (`sqlite_vec_adapter.py:380`) | `AsyncSqliteVecReaderFactory` (`sqlite_vec_adapter.py:525`) |
| Composite | KV + vector | `CompositeReaderFactory` (`composite_adapter.py:196`) | `AsyncCompositeReaderFactory` (`composite_adapter.py:289`) |

### SQLite: download vs range-read

- **Download-and-cache** (`SqliteShardReader`, `sqlite_adapter.py:250`) — fetches the entire shard `.db` file to local disk on first access, then opens it with the standard `sqlite3` driver. Best for small shards or when local disk is plentiful.
- **Range-read VFS** (`SqliteRangeShardReader`, `sqlite_adapter.py:456`) — uses APSW with a custom VFS (`create_apsw_vfs`, `_sqlite_vfs.py:284`) that issues S3 range requests for individual SQLite pages via [obstore](https://developmentseed.org/obstore/) (Rust `object_store` Python bindings). Reads are decomposed into fixed-size pages (`_PAGE_SIZE = 4096`) keyed by index in an LRU cache; missing pages from a single SQLite read are fetched together via `obstore.get_ranges`, which coalesces adjacent ranges into one HTTP request and parallelises disjoint ones. Best for large shards with sparse access patterns. Page cache is sized by `_normalize_page_cache_pages` (`_sqlite_vfs.py:39`).

### sqlite-vec adapter

`SqliteVecAdapter` (`sqlite_vec_adapter.py:134`) is a unified KV+vector adapter using the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension. Both KV pairs and vectors live in the same SQLite file. Distance metrics are translated via `_sqlite_vec_metric` (`sqlite_vec_adapter.py:60`), which accepts only `cosine` and `l2`; any other value (including `dot_product`) raises `ConfigValidationError`.

The extension is loaded via `_load_sqlite_vec` (`sqlite_vec_adapter.py:77`) which wraps `conn.enable_load_extension`.

### Composite adapter

`CompositeAdapter` (`composite_adapter.py:111`) wires two underlying adapters — one KV (SlateDB/SQLite), one vector (LanceDB) — and exposes a unified writer interface. On the reader side, `CompositeShardReader` (`composite_adapter.py:247`) routes KV lookups to the KV reader and vector queries to the vector reader.

LanceDB supports `cosine`, `l2`, and `dot_product` distances; `dot_product` is mapped to LanceDB's native `"dot"` metric in `vector/adapters/lancedb_adapter.py:142`. Validation lives in `vector/config.py`.

## Adapter errors

- `DbAdapterError` (`errors.py:62`) — base class.
- `SqliteAdapterError` (`sqlite_adapter.py:47`) — SQLite-specific. **Not** in public `__all__`; import from `shardyfusion.sqlite_adapter`.
- `SqliteVecAdapterError` (`sqlite_vec_adapter.py:94`) — sqlite-vec-specific. **Not** in public `__all__`; import from `shardyfusion.sqlite_vec_adapter`.
- `CompositeAdapterError` (`composite_adapter.py:57`) — composite-specific (e.g. mismatched key sets between KV and vector). **Not** in public `__all__`; import from `shardyfusion.composite_adapter`.

## Choosing an adapter

| You want | Pick |
|---|---|
| Just KV, default | SlateDB |
| Just KV, want SQL queries on each shard | SQLite (download) |
| Just KV, large shards, sparse access | SQLite (range-read) |
| KV + vector, single file per shard | sqlite-vec |
| KV + vector, vector engine of record (LanceDB) | Composite (KV adapter + LanceDB) |

## See also

- [`writer-core.md`](writer-core.md) — how factories are invoked.
- [`optional-imports.md`](optional-imports.md) — extras pattern that gates adapter availability.
- Use-case index for backend-specific recipes.
