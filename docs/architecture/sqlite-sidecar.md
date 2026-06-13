# SQLite page-cache sidecar

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
s3://bucket/.../shards/run_id=X/db=00000/attempt=00/shard.sidecar
```

The sidecar is a self-describing little-endian binary blob: a small
uncompressed prefix — vendor-neutral `SQPC` magic, a `u8` version, the
uncompressed body size (`u64`), the SQLite page size (`u32`), and the `.db`
object tag (see [Database binding](#database-binding)) — followed by one zstd
frame (content-checksummed). The frame body is metadata-first: the page count
`n`, the sorted `pagenos` bisect key, pages-relative `offsets`, the
overflow-chain CSR index (sorted chain heads + offsets + a flat page list),
then the gap-stripped pages. The byte-exact layout is
specified in
[`reference/sqlite-sidecar-format.md`](../reference/sqlite-sidecar-format.md);
this page covers the surrounding architecture (page selection, gap stripping,
binding, discovery, failure semantics).

The overflow-chain index records every chain in the kv table as a CSR triple — a
sorted `chain_heads` bisect key, a `chain_offsets` table, and a flat `chain_pages`
list of every chain's pages head-first in traversal order — so a range-mode reader
binary-searches a chain head and slices its page list directly off the decompressed
metadata (no per-chain dict), then prefetches the whole chain in one parallel
multi-range request instead of chasing each `next-pageno` pointer serially. It
compresses well (`chain_heads`/`chain_offsets` are monotonic and a chain's pages
are usually near-sequential) and holds no chains when no value overflows.

## Gap stripping

Stored B-tree pages have their unallocated middle physically removed before
compression. SQLite navigates via each page's cell-pointer array and never
reads the gap between that array and the cell-content area, so the writer drops
those bytes and the reader splices zeros back in to recover a page SQLite
treats as identical. Interior pages are ~50% gap, so this roughly halves the
*decompressed* sidecar — the bytes a reader holds resident per open shard —
not merely the wire size (zstd already squashes the zeroed gap either way). The
reader reconstructs each page directly into SQLite's read buffer, so it retains
only the compact stripped body; reconstruction is deterministic from the stored
page header (math in the format spec). A representative 100 k-row shard
(9.5 MiB DB) emits a sidecar well under 1% of the DB. A DB that reserves
per-page bytes (a checksum/encryption VFS) is left without a sidecar, since
stripping would be unsafe there.

## Database binding

The prefix carries the `.db` object's storage version token — its S3 ETag.
This is a **correctness** mechanism: a reader uses the sidecar only when the
live `.db` ETag matches the embedded tag, so a stale or mismatched sidecar can
never feed SQLite wrong interior pages. The writer therefore uploads
`shard.db` first, reads back its ETag, and stamps it into the sidecar; a reader
may additionally pin its range reads with `If-Match` so the store rejects a
`.db` that changes mid-read. The tag is an opaque identity token (never
recomputed), so multipart / SSE-KMS ETags work unchanged; an unbound sidecar
(`tag_len == 0`) is declined by a correctness-strict reader.

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
  "sqlite_sidecar": {
    "format_version": 8,
    "page_size": 4096,
    "filename": "shard.sidecar",
    "codec": "zstd"
  }
}
```

When the factory is configured with `page_size="auto"` (post-write
VACUUM picks per-shard), the manifest reports the literal `"auto"` in
the `page_size` field. Readers must then inspect each shard's SQLite
file header (bytes 16–17, big-endian u16; 1 means 65536) to learn the
real value — the range-read VFS does this automatically and exposes
it via `S3ReadOnlyFile.page_size`.

A range-mode reader can detect the flag once per snapshot and fetch
`{shard.db_url}/shard.sidecar` for every shard it opens, with no
per-shard probe.

## Configuration

Both `SqliteFactory.emit_sidecar` and
`SqliteVecFactory.emit_sidecar` default to `True`. Opt out
explicitly when desired:

```python
from shardyfusion.sqlite_adapter import SqliteFactory

SqliteFactory(emit_sidecar=False)
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
| APSW not installed (writer base install only)        | Debug-level `sqlite_sidecar_unsupported` log with `reason=apsw_not_installed`; skip sidecar; continue to upload `shard.db`.                       |
| `zstandard` not installed                            | Same — `reason=zstandard_not_installed`.                                                                                                             |
| `dbstat` virtual table missing in the SQLite build   | Same — `reason=dbstat_unavailable`.                                                                                                                  |
| Unsafe journal mode on the finalized DB              | Same — `reason=unsupported_journal_mode`. The extractor reads page bytes by direct file I/O at deterministic offsets; this is safe only when committed state lives entirely in the main `.db` file (modes `off`, `delete`, `memory`, `truncate`, `persist`). WAL-family modes can park committed pages in a `-wal` sidecar that raw file I/O would miss, so the guard refuses to run on `wal`/`wal2` files rather than emit a silently incomplete sidecar. |
| Any other extraction error                           | Warning-level `sqlite_sidecar_failed` log; skip sidecar; continue to upload `shard.db`.                                                            |
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
