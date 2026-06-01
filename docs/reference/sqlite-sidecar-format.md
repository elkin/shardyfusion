# SQLite Page-Cache Sidecar Format — v7

A compact binary file holding a selected set of SQLite database pages, indexed
by page number, so a remote-storage SQLite reader can prefetch and cache the
pages SQLite must traverse on every query — the schema btree and all interior
B-tree nodes — eliminating per-query round trips for B-tree navigation. It also
records every overflow chain so a reader can prefetch a chain in one coalesced
range request, and binds itself to the exact `.db` object it describes.

The format is **vendor-neutral** (magic `SQPC`, no shardyfusion-specific
fields) so it can be produced and consumed outside shardyfusion.

This document specifies the byte layout. It does not mandate *which* pages a
producer includes; any subset is valid as long as the layout holds.

## Wire format

All multi-byte integers are little-endian. Byte offsets are zero-based.

| Offset       | Size     | Field            | Description                                  |
| ------------ | -------- | ---------------- | -------------------------------------------- |
| 0            | 4        | `magic`          | `b"SQPC"`                                     |
| 4            | 1        | `format_version` | `u8` — value `7` for this spec               |
| 5            | 8        | `body_size`      | `u64` — uncompressed size of `body` in bytes |
| 13           | 1        | `tag_len`        | `u8` — length of `db_tag` (`0` = unbound)    |
| 14           | tag_len  | `db_tag`         | the `.db` object-version token (S3 ETag), UTF-8 |
| 14 + tag_len | rest     | `body`           | one zstd frame (with content checksum)       |

The prefix is uncompressed so a reader can validate the magic/version and read
the binding tag before paying the decompression cost. The zstd frame is written
with a content checksum, so a torn or bit-rotted sidecar fails to decode rather
than yielding corrupt pages.

## Body

After zstd-decompressing `body`, the bytes are, in order:

| Field         | Size            | Description                                                        |
| ------------- | --------------- | ------------------------------------------------------------------ |
| `page_size`   | `u32`           | SQLite page size in bytes (power of two in `[512, 65536]`)         |
| `n`           | `u32`           | number of pages in the sidecar                                     |
| `pagenos`     | `n × u32`       | SQLite page numbers (1-based), sorted strictly ascending           |
| `offsets`     | `(n+1) × u32`   | start offset of each stored page **within `pages`**; entry `n` is the `pages` byte length |
| `chain_count` | `u32`           | number of overflow chains `C` (may be 0)                           |
| `chain_heads`   | `C × u32`     | head page numbers, one per chain, **sorted strictly ascending** (the chain bisect key) |
| `chain_offsets` | `(C+1) × u32` | start index of each chain **within `chain_pages`**, in page-number units; entry `C` is the `chain_pages` length `M` |
| `chain_pages`   | `M × u32`     | every chain's pages, **head-first** in traversal order, concatenated in `chain_heads` order |
| `pages`       | variable        | `n` **gap-stripped** pages, concatenated in `pagenos` order        |

All small index/metadata sections precede the bulk `pages` blob, so a reader may
stream-decompress just the metadata prefix without inflating the page bytes.

### Field semantics

- **`pagenos`** is the bisect key: a reader binary-searches it to test whether a
  requested page is in the sidecar and to find its index `i`.
- **`offsets`** locates page `i` as `pages[offsets[i] : offsets[i+1]]`. Stored
  pages are variable-length (see gap stripping), so these offsets are
  load-bearing.
- **`chain_heads`** is the chain bisect key: a reader binary-searches it for a
  cell's overflow head `H`, and on a hit at index `j` the chain's pages are the
  direct slice `chain_pages[chain_offsets[j] : chain_offsets[j+1]]` — no
  per-chain dict is built first. (`chain_offsets` counts page numbers, not bytes
  like `offsets`, since every `chain_pages` entry is a fixed-width `u32`.)
- **`chain_pages`** stores each chain head-first in traversal order, so a reader
  prefetches the whole chain from the main `.db` in one parallel/coalesced range
  request rather than chasing each page's 4-byte big-endian next-pointer
  serially. Overflow page *contents* are not stored — only their page numbers.
- **`pages`** are gap-stripped (below). A reader reconstructs each to a full
  `page_size` page before handing it to SQLite.

## Gap stripping

Each stored B-tree page has its unallocated middle physically removed. A SQLite
B-tree page is an 8/12-byte header, a 2-byte-per-cell *cell pointer array*
growing forward, a contiguous *unallocated gap*, then the *cell content area*
growing backward to the page end. SQLite navigates via the pointer array and
never reads the gap, so dropping it (and refilling with zeros on reconstruction)
is invisible to SQLite while roughly halving the decompressed/resident size.

Per page (header fields are **big-endian**):

- `base` = 100 on page 1 (it carries the 100-byte DB header first), else 0.
- page-type byte at `page[base]`: interior index/table = 2/5, leaf index/table
  = 10/13. Any other value means it is not a B-tree page — store it whole.
- header size = 12 for interior pages (a 4-byte right-child pointer), else 8.
- `n_cells` = `page[base+3 : base+5]`.
- `cca` (cell-content-area start) = `page[base+5 : base+7]`; a stored `0` means
  65536.
- `cpa_end = base + header_size + 2 * n_cells`.

The **stored** page is `page[:cpa_end] + page[cca:]` — the gap `[cpa_end, cca)`
is dropped (a full page with no gap is stored unchanged). The **reconstructed**
page is `stored[:cpa_end] + b"\x00" * (cca - cpa_end) + stored[cpa_end:]`, with
`cca`/`n_cells` read from the stored header. Reconstruction is deterministic and
needs no extra per-page metadata.

A producer **must not** gap-strip a DB that reserves per-page bytes (byte 20 of
the 100-byte DB header is non-zero), since a checksum/encryption VFS may cover
the gap; such a DB should be left without a sidecar.

## Database binding

`db_tag` is the storage object-version token of the `.db` the sidecar describes
— the S3 ETag (GCS generation, etc.). It exists for **correctness**: a reader
uses the sidecar only when the *live* `.db` tag equals `db_tag`, so a stale or
mismatched sidecar can never feed SQLite wrong pages. A producer therefore
uploads the `.db` first, captures its tag, then builds the sidecar around it. A
reader may additionally pin its `.db` range reads with `If-Match: <db_tag>` so
the store rejects a `.db` that changes mid-read.

The tag is an *identity* token, not a content hash (multipart/SSE-KMS ETags are
not the content MD5): it is stored and compared opaquely, never recomputed. An
unbound sidecar (`tag_len == 0`) carries no binding; a correctness-strict reader
declines it.

## Validation

A reader rejects (and falls back to fetching pages on demand) if any of:

- `magic` is not `b"SQPC"`.
- `format_version` is not understood by the reader.
- zstd decode fails (including the frame-checksum check).
- `page_size` is not a power of two in `[512, 65536]`.
- `pagenos` is not strictly ascending.
- `offsets` is not monotonically non-decreasing, its first entry is not `0`, or
  its last entry does not equal the `pages` byte length.
- `chain_heads` is not strictly ascending.
- `chain_offsets` is not strictly increasing, its first entry is not `0`, or its
  last entry does not equal the `chain_pages` length. (Strictly increasing
  because every chain — head included — has at least one page.)
- a stored chain does not begin with its head
  (`chain_pages[chain_offsets[j]] != chain_heads[j]` for some `j`).
- a reconstructed page is not exactly `page_size` bytes.

Separately — a *correctness gate*, not a structural reject — a reader declines
to use the sidecar (serving the `.db` directly) when `db_tag` differs from the
live `.db` object tag.

## Versioning

`format_version` is a monotonic `u8`. A reader that receives a higher version
than it understands treats the sidecar as absent and fetches pages on demand.
Versions 1–4 used an 8-byte `SFBTM` magic under the name `shard.btreemeta` and
stored whole (un-stripped) pages with a `(pageno, offset)` index; v5 is a
breaking change (new magic, 1-byte version, ETag binding, gap-stripped pages).
v6 adds `body_size` (u64, offset 5) to the uncompressed prefix so a reader can
size its decompression buffer before paying the zstd decode cost. v7 restructures
the overflow-chain section from variable-length `(head, L, pageno…)` records into
a CSR triple (sorted `chain_heads` + `chain_offsets` + flat `chain_pages`) so a
reader binary-searches a chain head and slices its page list directly off the
decompressed metadata, with no dict construction.

## Reference parser (Python)

```python
import struct
import zstandard

MAGIC = b"SQPC"
VERSION = 7


def parse_sidecar(blob: bytes):
    """Return ``(db_tag, body_size, page_size, [(pageno, full_page_bytes), ...], chains)``."""
    assert blob[:4] == MAGIC, "bad magic"
    assert blob[4] == VERSION, f"unsupported version {blob[4]}"
    body_size = int.from_bytes(blob[5:13], "little")
    tag_len = blob[13]
    db_tag = blob[14 : 14 + tag_len].decode("utf-8") if tag_len else None
    body = zstandard.ZstdDecompressor().decompress(blob[14 + tag_len :])

    cur = 0
    page_size, n = struct.unpack_from("<II", body, cur)
    cur += 8
    pagenos = list(struct.unpack_from(f"<{n}I", body, cur)) if n else []
    cur += 4 * n
    offsets = list(struct.unpack_from(f"<{n + 1}I", body, cur))
    cur += 4 * (n + 1)
    (chain_count,) = struct.unpack_from("<I", body, cur)
    cur += 4
    chain_heads = (
        list(struct.unpack_from(f"<{chain_count}I", body, cur)) if chain_count else []
    )
    cur += 4 * chain_count
    chain_offsets = list(struct.unpack_from(f"<{chain_count + 1}I", body, cur))
    cur += 4 * (chain_count + 1)
    n_pages = chain_offsets[-1]
    flat = list(struct.unpack_from(f"<{n_pages}I", body, cur)) if n_pages else []
    cur += 4 * n_pages
    chains = [flat[chain_offsets[i] : chain_offsets[i + 1]] for i in range(chain_count)]
    # A real reader bisects ``chain_heads`` for a cell's overflow head and slices
    # ``flat[chain_offsets[j]:chain_offsets[j+1]]``; the rebuilt chains must each
    # start with that head.
    assert chain_heads == [c[0] for c in chains]
    pages = body[cur:]

    out = []
    for i, pageno in enumerate(pagenos):
        stored = pages[offsets[i] : offsets[i + 1]]
        out.append((pageno, _reconstruct_page(stored, pageno, page_size)))
    return db_tag, body_size, page_size, out, chains


def _reconstruct_page(stored: bytes, pageno: int, page_size: int) -> bytes:
    base = 100 if pageno == 1 else 0
    ptype = stored[base]
    if ptype not in (2, 5, 10, 13):
        return stored  # stored whole
    hdr = 12 if ptype in (2, 5) else 8
    n_cells = int.from_bytes(stored[base + 3 : base + 5], "big")
    cca = int.from_bytes(stored[base + 5 : base + 7], "big") or 65536
    cpa_end = base + hdr + 2 * n_cells
    return stored[:cpa_end] + b"\x00" * (cca - cpa_end) + stored[cpa_end:]
```
