# SQLite B-Tree Sidecar Format — v3

A compact binary file containing a selected set of SQLite database pages,
indexed by page number. Designed to let a remote-storage SQLite reader
prefetch and cache the pages SQLite must traverse on every query — the
schema btree and all interior B-tree nodes — eliminating per-query round
trips for B-tree navigation.

This document specifies the byte layout. It does not specify *which*
pages a producer should include; any subset of pages is valid as long as
the layout below is correct.

## Wire format

All multi-byte integers are little-endian. Byte offsets are zero-based.

| Offset | Size | Field            | Description                          |
| ------ | ---- | ---------------- | ------------------------------------ |
| 0      | 8    | `magic`          | `b"SFBTM\x00\x00\x00"`               |
| 8      | 4    | `format_version` | `u32` — value `3` for this spec      |
| 12     | rest | `body`           | zstd-compressed body (see next)      |

The magic and version are deliberately uncompressed so a reader can
validate the artifact and reject unsupported versions before paying the
decompression cost.

## Body format

After zstd-decompressing `body`, the bytes are:

| Offset      | Size              | Field       | Description                                              |
| ----------- | ----------------- | ----------- | -------------------------------------------------------- |
| 0           | 4                 | `page_size` | `u32` — SQLite page size in bytes                        |
| 4           | 4                 | `n`         | `u32` — number of pages in the sidecar                   |
| 8           | 8 · n             | `index`     | `n` × `(u32 pageno, u32 offset)`, sorted ascending       |
| 8 + 8 · n   | page_size · n     | `slabs`     | `n` × `page_size`-byte page contents, in same order      |

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
development and never shipped in a release.

## Reference parser (Python)

```python
import struct
import zstandard

MAGIC = b"SFBTM\x00\x00\x00"
VERSION = 3


def parse_sidecar(blob: bytes) -> tuple[int, list[tuple[int, bytes]]]:
    """Return ``(page_size, [(pageno, page_bytes), ...])``."""
    assert blob[:8] == MAGIC, "bad magic"
    fv = int.from_bytes(blob[8:12], "little")
    assert fv == VERSION, f"unsupported format_version {fv}"
    body = zstandard.ZstdDecompressor().decompress(blob[12:])
    page_size, n = struct.unpack("<II", body[:8])
    pairs = struct.unpack(f"<{2 * n}I", body[8 : 8 + 8 * n])
    return page_size, [
        (pageno, body[offset : offset + page_size])
        for pageno, offset in zip(pairs[0::2], pairs[1::2], strict=True)
    ]
```

## Reference producer (Python)

```python
import struct
import zstandard


def build_sidecar(page_size: int, pages: list[tuple[int, bytes]]) -> bytes:
    """Build a v3 sidecar.

    ``pages`` must be sorted strictly ascending by ``pageno``; each
    ``page_bytes`` must be exactly ``page_size`` bytes.
    """
    n = len(pages)
    data_start = 8 + 8 * n
    index_pairs: list[int] = []
    for i, (pageno, _) in enumerate(pages):
        index_pairs.extend([pageno, data_start + i * page_size])
    body = (
        struct.pack("<II", page_size, n)
        + (struct.pack(f"<{2 * n}I", *index_pairs) if n else b"")
        + b"".join(slab for _, slab in pages)
    )
    return (
        MAGIC
        + VERSION.to_bytes(4, "little")
        + zstandard.ZstdCompressor(level=3).compress(body)
    )
```
