# SQLite B-tree metadata sidecar

The writer emits a small auxiliary artifact alongside each finalized SQLite
shard. The sidecar bundles the pages that range-mode readers always traverse
during navigation — page 1, the schema btree, and every interior B-tree page
— so the reader can fetch them once on shard open and pin them for the
lifetime of the shard reader.

## Why it exists

A finalized shard database is read-only once published. Page numbering and
the content of every B-tree internal node are frozen for the life of the
shard. Without a sidecar:

- Each point lookup walks root → interior → leaf — typically 3-4 page
  reads per query for a 1 M-row shard.
- The reader's LRU cache must hold those interior pages alongside hot
  leaves, and on warm-but-pressured workloads it evicts them, regenerating
  the round trips.

With a sidecar:

- One S3 GET on shard open primes the page cache with every interior page.
- Each point lookup needs at most one S3 GET (the leaf) for an uncached
  read; warm reads serve from the pinned cache.
- ~3× lower steady-state latency per point lookup, and far less LRU churn
  at high QPS.

For shardyfusion's typical fan-out (≤ 1 GiB shards, 4 KiB pages, depth-3
B-trees) the sidecar is ~0.5–2% of the main DB.

## Artifact

Each finalized shard publishes two sibling objects:

```
s3://bucket/.../shards/run_id=X/db=00000/attempt=00/shard.db
s3://bucket/.../shards/run_id=X/db=00000/attempt=00/shard.btreemeta
```

The sidecar is a self-describing little-endian binary blob with a small
uncompressed header followed by a zstd-compressed body:

```
offset   size      field
------   -------   ----------------------------------
   0     8         magic            = b"SFBTM\x00\x00\x00"
   8     4         format_version   (u32) = 3
  12     ...       zstd-compressed body
```

The body, once decompressed, contains:

```
offset   size      field
------   -------   ----------------------------------
   0     4         page_size        (u32, e.g. 4096)
   4     4         page_count N     (u32) — number of pages in sidecar
   8     8 * N     index            (u32 pageno, u32 offset) per entry,
                                    sorted ascending by pageno
   8+8N  PS * N    page_data        (page_size bytes each)
```

`offset` is the body-relative byte position of the corresponding page
slab. Equivalently, when a consumer writes the decompressed body to a
file on disk, `offset` is the file offset.

This shape lets a reader pick its memory strategy:

- **Pin everything in memory.** Decompress the body, populate the LRU
  cache by walking the index. Standard path for the eventual range-mode
  reader.
- **On-disk lookups.** Decompress the body to a local file. Hold only
  the `pageno → offset` map in memory (~8 bytes per page) and `pread`
  individual pages on demand; the OS page cache amortizes hot reads.
  For million-page sidecars this trades ~MB-scale resident memory for
  ~KB-scale.

Storing `offset` explicitly is redundant when pages are uniformly
`page_size` bytes (it equals `8 + 8*N + i*page_size`), but keeping it
in the format costs almost nothing under zstd (offsets are an
arithmetic progression and compress to a few bits each) and:

- Removes the offset-arithmetic burden from disk consumers.
- Future-proofs the format for variable-size pages (e.g. per-page
  compression for very large sidecars).

The magic and version stay uncompressed so a reader can validate the
artifact (and reject mismatched versions) before paying the
decompression cost. Btree pages compress ~12× under zstd at level 3 in
practice — interior pages are typically ~50% free space, and the
per-page headers and cell pointer arrays repeat across the bundle. A
representative 100 k-row shard (9.5 MiB DB) emits a ~21 KB sidecar
(~0.2% of the DB).

The byte-exact format is specified in
[`reference/sqlite-btree-sidecar-format.md`](../reference/sqlite-btree-sidecar-format.md);
this page covers the surrounding architecture (page selection,
discovery, failure semantics).

## Page selection

The extractor uses SQLite's own `dbstat` virtual table (via APSW) to
identify pages, then reads page bytes by direct file I/O at the
deterministic `(pageno - 1) * page_size` offsets. The selection query is:

```sql
SELECT pageno
FROM   dbstat
WHERE  pagetype = 'internal'
    OR name IN ('sqlite_master', 'sqlite_schema')
ORDER  BY pageno;
```

This selects:

1. Every interior B-tree page in any btree (table or index, regular or
   shadow tables of the `sqlite-vec` virtual table).
2. Every page belonging to the schema btree, regardless of whether it is
   leaf or interior. Schema leaves matter: they hold each table/index's
   `rootpage`, which SQLite reads on every connection open. Caching them
   eliminates one round trip on shard open.

We deliberately do not parse the SQLite file format ourselves. SQLite's
own introspection automatically handles every page type, transparently
covers virtual-table shadow btrees, and tracks any future format changes.

## Manifest discovery

When the writer emits sidecars, it also records a snapshot-level flag in
the manifest's `custom` field:

```json
{
  "sqlite_btreemeta": {
    "format_version": 1,
    "page_size": 4096,
    "filename": "shard.btreemeta"
  }
}
```

A range-mode reader can detect the flag once per snapshot and fetch
`{shard.db_url}/shard.btreemeta` for every shard it opens, with no
per-shard probe.

## Configuration

Both `SqliteFactory.emit_btree_metadata` and
`SqliteVecFactory.emit_btree_metadata` default to `True`. Opt out
explicitly when desired:

```python
from shardyfusion.sqlite_adapter import SqliteFactory

SqliteFactory(emit_btree_metadata=False)
```

Recommended only when the deployment never uses the range-read VFS — for
download-mode reads, the sidecar costs an extra PUT at write time without
benefit.

## Coverage

| Adapter / Factory     | Sidecar emitted?                                                                |
|-----------------------|---------------------------------------------------------------------------------|
| `SqliteFactory`       | Yes — toggle on the factory.                                                    |
| `SqliteVecFactory`    | Yes — same toggle, same code path. `dbstat` covers `sqlite-vec` shadow btrees. |
| `CompositeAdapter` with `SqliteFactory` for the KV side | KV-side only — the inner SQLite adapter handles its own sidecar; the vector half (e.g. LanceDB) is not SQLite. |
| Other adapters (SlateDB, etc.) | No.                                                                      |

## Failure semantics

Sidecar emission is **best-effort**. The contract:

| Event                                                | Behavior                                                                                                                                            |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| APSW not installed (writer base install only)        | Debug-level `sqlite_btreemeta_unsupported` log with `reason=apsw_not_installed`; skip sidecar; continue to upload `shard.db`.                       |
| `zstandard` not installed                            | Same — `reason=zstandard_not_installed`.                                                                                                             |
| `dbstat` virtual table missing in the SQLite build   | Same — `reason=dbstat_unavailable`.                                                                                                                  |
| Unsafe journal mode on the finalized DB              | Same — `reason=unsupported_journal_mode`. The extractor reads page bytes by direct file I/O at deterministic offsets; this is safe only when committed state lives entirely in the main `.db` file (modes `off`, `delete`, `memory`, `truncate`, `persist`). WAL-family modes can park committed pages in a `-wal` sidecar that raw file I/O would miss, so the guard refuses to run on `wal`/`wal2` files rather than emit a silently incomplete sidecar. |
| Any other extraction error                           | Warning-level `sqlite_btreemeta_failed` log; skip sidecar; continue to upload `shard.db`.                                                            |
| Sidecar PUT fails                                    | Same — sidecar dropped, manifest still lists the shard, reader (when wired) falls back to per-page lazy fetch.                                       |
| Main `shard.db` PUT fails                            | Existing behavior — re-raise; the shard's retry path handles it.                                                                                     |

A missing or corrupt sidecar must never block a write or break the reader's
existing path.

### Why we don't use `sqlite_dbpage`

SQLite has a virtual table, `sqlite_dbpage`, that returns raw page bytes
through SQLite's own pager — which would automatically handle WAL,
encryption, custom VFSes, and concurrent writers. We deliberately don't
use it: APSW does not consistently ship with `SQLITE_ENABLE_DBPAGE_VTAB`
enabled, which would make sidecar emission silently no-op on otherwise
healthy environments.

Instead, the extractor uses `dbstat` (which APSW does ship with) to
identify *which* pages we want, then reads their bytes by direct file
I/O at the deterministic `(pageno - 1) * page_size` offsets. In the
narrow context of a finalized shardyfusion shard — `journal_mode=OFF`,
no encryption, no concurrent writers, file already closed by
`checkpoint()` — this is byte-for-byte equivalent to what
`sqlite_dbpage` would return. The journal-mode guard above turns the
"narrow context" assumption into an explicit invariant: any future
writer change that violates it (e.g. enabling WAL) trips the guard at
write time rather than producing silently-incomplete sidecars.

## Dependencies

Sidecar emission requires:

- `apsw` — for the `dbstat` virtual table that identifies which pages to
  bundle. APSW bundles a SQLite built with `SQLITE_ENABLE_DBSTAT_VTAB`
  enabled.
- `zstandard` — for body compression.

Both ship under the `[sqlite-range]` extra, which is also what readers
install for the range-read VFS:

```sh
pip install 'shardyfusion[sqlite-range]'
```

If either dependency is missing the writer logs at debug level and skips
the sidecar; the main `shard.db` upload proceeds unchanged.
