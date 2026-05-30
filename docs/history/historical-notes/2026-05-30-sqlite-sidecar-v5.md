# 2026-05-30 SQLite page-cache sidecar v5

- Status: `implemented`
- Date: `2026-05-30`
- Commit: `(pending)` on branch `worktree-sidecar-v5`

## Summary

v4 (the overflow-chain sidecar, `shard.btreemeta`) left two things on the
table: it stored whole pages — including the uncleaned, high-entropy gap that
SQLite leaves in the middle of every B-tree page — and it carried no binding to
the `.db` it described. v5 reworks the format to fix both, de-brands it for
reuse outside shardyfusion, and is a clean break (pre-release).

The shipping format:

1. **Vendor-neutral envelope** — magic `SQPC` (4 bytes) + a `u8` version, then
   one zstd frame (content-checksummed). Renamed `shard.btreemeta` →
   `shard.sidecar`; the `emit_btree_metadata` factory flag →
   `emit_sidecar`; the manifest field `sqlite_btreemeta` → `sqlite_sidecar`.
2. **Gap stripping** — the writer physically removes each B-tree page's
   unallocated gap (`[cell_pointer_array_end, cell_content_area_start)`); the
   reader splices zeros back to recover a byte-identical-to-SQLite page. This
   roughly halves the *decompressed/resident* size (interior pages are ~50%
   gap), which is the cost a multi-shard reader actually pays. Wire size is
   ~unchanged from v4 (zstd already squashed the gap), but v4 stored the gap's
   freed-cell garbage, so v5 also compresses a little better.
3. **Object binding (correctness)** — the writer uploads `shard.db` first,
   reads back its S3 ETag, and stamps it into the sidecar prefix. A reader uses
   the sidecar only when the live `.db` ETag matches, so a stale or mismatched
   sidecar can never feed SQLite wrong interior pages. The ETag is verifiable
   by a range-reader for free (it rides every S3 response header), which a
   content hash is not.
4. **Body reorder** — metadata first (`page_size | n | pagenos | offsets |
   chains`), then the gap-stripped pages as a single trailing blob. `pagenos`
   is the bisect key; pages-relative `offsets` locate each (now
   variable-length) stored page; the overflow-chain map carries over from v4.

## Design path (rejected alternatives)

The format was narrowed through review; each cut has a reason worth keeping:

- **Per-page compression + zstd dictionary** — would give true random-access
  decode, but for an all-hot navigation payload a reader caches nearly every
  page anyway, so the memory win is theoretical while the reader complexity
  (per-page cache, dictionary) is real. Whole-file single-frame compression is
  both smaller and simpler; rejected per-page.
- **Split-frame compression / mmap-addressable fixed-width index** — same
  reasoning: optimizes a sparse-access pattern this payload does not have.
- **Rootpage map** (`table → rootpage`) — redundant. SQLite resolves rootpages
  itself from the `sqlite_schema` pages the sidecar already bundles; a VFS
  reader never consults it. Cut.
- **`flags` field** — the only case that would set it (a reserved-bytes DB,
  where gap stripping is unsafe) is handled by skipping the sidecar entirely,
  so no runtime flag is needed: v5 pages are always gap-stripped.

## Reader contract (the reader does not exist yet)

The format is staged for a range-read VFS. On open it would: validate the
prefix, compare `db_tag` to the live `.db` ETag (mismatch → ignore the sidecar,
serve `.db` directly), decompress the body, bisect `pagenos`, and reconstruct
the located page into SQLite's `xRead` buffer. Overflow heads trigger a
coalesced prefetch of the whole chain from the chain map.

## Verification

Gap-stripping correctness is gated by a reconstruct-and-check test: reconstruct
every stored page, rebuild a DB with them, and assert `PRAGMA integrity_check`
returns `ok` and every row reads back identically — SQLite never reads the gap,
so the rebuilt DB is equivalent. The binding is verified end-to-end against
moto S3 (the sidecar's embedded tag equals the live `.db` ETag).
