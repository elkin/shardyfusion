# SQLite B-Tree Sidecar Format — v4

A compact binary file containing a selected set of SQLite database pages,
indexed by page number, plus a map of every kv overflow chain.
Designed to let a remote-storage SQLite reader prefetch and cache the
pages SQLite must traverse on every query — the schema btree and all
interior B-tree nodes — and to prefetch entire overflow chains in one
parallel multi-range request rather than chasing each next-pointer
sequentially.

This document specifies the byte layout. It does not specify *which*
pages a producer should include; any subset of pages is valid as long as
the layout below is correct.

## Wire format

All multi-byte integers are little-endian. Byte offsets are zero-based.

| Offset | Size | Field            | Description                          |
| ------ | ---- | ---------------- | ------------------------------------ |
| 0      | 8    | `magic`          | `b"SFBTM\x00\x00\x00"`               |
| 8      | 4    | `format_version` | `u32` — value `4` for this spec      |
| 12     | rest | `body`           | zstd-compressed body (see next)      |

The magic and version are deliberately uncompressed so a reader can
validate the artifact and reject unsupported versions before paying the
decompression cost.

## Body format

After zstd-decompressing `body`, the bytes are:

| Offset                  | Size                    | Field         | Description                                                                 |
| ----------------------- | ----------------------- | ------------- | --------------------------------------------------------------------------- |
| 0                       | 4                       | `page_size`   | `u32` — SQLite page size in bytes                                           |
| 4                       | 4                       | `n`           | `u32` — number of pages in the sidecar                                      |
| 8                       | 8 · n                   | `index`       | `n` × `(u32 pageno, u32 offset)`, sorted ascending by pageno                |
| 8 + 8 · n               | page_size · n           | `slabs`       | `n` × `page_size`-byte page contents, in same order                         |
| 8 + 8 · n + page_size·n | 4                       | `chain_count` | `u32` — number of overflow chains `C` (may be 0)                            |
| (next)                  | variable                | `chains`      | `C` chain entries; each is `u32 head_pageno`, `u32 length L`, `L × u32` pagenos in chain order (head first) |

### Field semantics

- **`page_size`** is the SQLite database's page size — a power of two in
  the range `[512, 65536]`.
- **`n`** is the number of `(pageno, offset)` index entries and the
  number of page slabs. Both arrays have the same length and the same
  ordering.
- **`pageno`** is the SQLite page number (1-based, matching SQLite's
  on-disk convention). Index entries are sorted strictly ascending by
  `pageno`.
- **`offset`** is the body-relative byte position of the corresponding
  slab. For this spec (uniform `page_size`), `offset_i = 8 + 8*n + i *
  page_size`. The field is stored explicitly so a consumer can write the
  decompressed body to a file and access individual pages via `pread`
  while holding only the `pageno → offset` map in memory.
- A **slab** is the SQLite page for the corresponding `pageno`,
  byte-for-byte identical to `read(source_db, (pageno - 1) * page_size,
  page_size)` from the source database.
- **`chain_count`** is the number of overflow chains the producer
  enumerated.  May be `0`; the field is always present.
- Each **chain** entry begins with a `head_pageno` (the leaf cell's
  overflow pointer) and a `length` `L`, followed by `L` `u32` pagenos
  in traversal order — the head first, then each successor read from
  the 4-byte big-endian `next-pageno` at offset 0 of every overflow
  page.  A reader can prefetch the entire chain in a single parallel
  multi-range request keyed on these pagenos instead of walking the
  on-disk next-pointers serially.

## Validation

A reader rejects the sidecar if any of:

- `magic` does not match `b"SFBTM\x00\x00\x00"`.
- `format_version` is not understood by the reader.
- zstd decompression fails.
- `page_size` is not a power of two in `[512, 65536]`.
- Index entries are not strictly ascending by `pageno`.
- Any `offset + page_size` exceeds the body length.

A producer is responsible for ensuring its output passes these checks.

## Versioning

`format_version` is a monotonic `u32`. A reader that receives a higher
version than it understands should treat the sidecar as absent and fall
back to fetching pages on demand. Versions 1 and 2 existed during
development and never shipped in a release. Version 3 shipped briefly
and is byte-compatible with v4 for the `page_size`/`index`/`slabs`
section; v3 readers MUST refuse v4 (the trailing `chain_count` +
`chains` block is not present in v3 and will be silently truncated by a
naive parser).

## Reference parser (Python)

```python
import struct
import zstandard

MAGIC = b"SFBTM\x00\x00\x00"
VERSION = 4


def parse_sidecar(
    blob: bytes,
) -> tuple[int, list[tuple[int, bytes]], list[list[int]]]:
    """Return ``(page_size, [(pageno, page_bytes), ...], [[head, p1, ...], ...])``."""
    assert blob[:8] == MAGIC, "bad magic"
    fv = int.from_bytes(blob[8:12], "little")
    assert fv == VERSION, f"unsupported format_version {fv}"
    body = zstandard.ZstdDecompressor().decompress(blob[12:])
    page_size, n = struct.unpack("<II", body[:8])
    pairs = struct.unpack(f"<{2 * n}I", body[8 : 8 + 8 * n])
    pages = [
        (pageno, body[offset : offset + page_size])
        for pageno, offset in zip(pairs[0::2], pairs[1::2], strict=True)
    ]
    cursor = 8 + 8 * n + n * page_size
    (chain_count,) = struct.unpack("<I", body[cursor : cursor + 4])
    cursor += 4
    chains: list[list[int]] = []
    for _ in range(chain_count):
        head, length = struct.unpack("<II", body[cursor : cursor + 8])
        cursor += 8
        chain = list(struct.unpack(f"<{length}I", body[cursor : cursor + 4 * length]))
        assert chain and chain[0] == head
        cursor += 4 * length
        chains.append(chain)
    return page_size, pages, chains
```

## Reference producer (Python)

```python
import struct
import zstandard


def build_sidecar(
    page_size: int,
    pages: list[tuple[int, bytes]],
    chains: list[list[int]] | None = None,
) -> bytes:
    """Build a v4 sidecar.

    ``pages`` must be sorted strictly ascending by ``pageno``; each
    ``page_bytes`` must be exactly ``page_size`` bytes.  ``chains`` is
    a list of overflow chains in traversal order (head first); pass
    ``None`` or ``[]`` for KV shards with no overflow chains.
    """
    n = len(pages)
    data_start = 8 + 8 * n
    index_pairs: list[int] = []
    for i, (pageno, _) in enumerate(pages):
        index_pairs.extend([pageno, data_start + i * page_size])

    chains = chains or []
    chain_parts: list[bytes] = [struct.pack("<I", len(chains))]
    for chain in chains:
        chain_parts.append(struct.pack("<II", chain[0], len(chain)))
        chain_parts.append(struct.pack(f"<{len(chain)}I", *chain))

    body = (
        struct.pack("<II", page_size, n)
        + (struct.pack(f"<{2 * n}I", *index_pairs) if n else b"")
        + b"".join(slab for _, slab in pages)
        + b"".join(chain_parts)
    )
    return (
        MAGIC
        + VERSION.to_bytes(4, "little")
        + zstandard.ZstdCompressor(level=3).compress(body)
    )
```
