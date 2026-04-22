# Choosing a reader strategy

This section covers reading sharded KV snapshots. Pick based on your code style and backend:

| Reader | Code style | Backends | Best for |
|---|---|---|---|
| **[Sync SlateDB](sync/slatedb.md)** | Synchronous | SlateDB | Default, lowest friction |
| **[Sync SQLite](sync/sqlite.md)** | Synchronous | SQLite | SQL queries, download-and-cache or range-read VFS |
| **[Async SlateDB](async/slatedb.md)** | `asyncio` | SlateDB | FastAPI, async workers |
| **[Async SQLite](async/sqlite.md)** | `asyncio` | SQLite | Async SQLite wrappers |

All readers share the same conceptual model: load `_CURRENT` → dereference manifest → open per-shard adapters → route lookups locally. See [KV Storage Overview](../overview.md) for the conceptual model.

## Key concept: refresh

Readers pin all lookups to the manifest loaded at open time (or last `refresh()`). To observe a newly published snapshot:

```python
changed = reader.refresh()  # True if newer manifest was loaded
```

This is the **only** way to advance — there is no automatic polling.
