"""Shared parser for the v7 SQLite page-cache sidecar, for tests.

Mirrors the reference parser in ``docs/reference/sqlite-sidecar-format.md`` and
enforces the v7 structural invariants (the spec's "Validation" section) so every
test that round-trips a sidecar exercises them.  Kept in one place so a future
format bump touches a single decoder rather than each test module's own copy.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest

from shardyfusion.sqlite_adapter import _SIDECAR_FORMAT_VERSION, _SIDECAR_MAGIC


@dataclass(frozen=True)
class ParsedSidecar:
    """A decoded sidecar: wire header plus reconstructed page/chain views."""

    version: int
    db_tag: str | None
    page_size: int
    n: int
    pagenos: list[int]
    stored_pages: list[bytes]
    chains: list[list[int]]


def parse_sidecar(blob: bytes) -> ParsedSidecar:
    """Parse a v7 sidecar blob and validate its structural invariants.

    Wire: ``magic(4) + version(u8) + body_size(u64) + tag_len(u8) + tag +
    zstd(body)``; the decompressed body is ``page_size(u32) + n(u32) +
    pagenos(u32*n) + offsets(u32*(n+1)) + chain_count(u32) + chain_heads(u32*C)
    + chain_offsets(u32*(C+1)) + chain_pages(u32*M) + gap-stripped pages``.
    """
    zstandard = pytest.importorskip("zstandard")

    assert len(blob) >= 14, len(blob)
    assert blob[:4] == _SIDECAR_MAGIC, blob[:4]
    version = blob[4]
    assert version == _SIDECAR_FORMAT_VERSION, version
    body_size = int.from_bytes(blob[5:13], "little")
    tag_len = blob[13]
    assert len(blob) >= 14 + tag_len, (len(blob), tag_len)
    db_tag = blob[14 : 14 + tag_len].decode("utf-8") if tag_len else None
    body = zstandard.ZstdDecompressor().decompress(blob[14 + tag_len :])
    assert body_size == len(body), (body_size, len(body))

    cur = 0
    page_size, n = struct.unpack_from("<II", body, cur)
    cur += 8
    assert 512 <= page_size <= 65536 and page_size & (page_size - 1) == 0
    pagenos = list(struct.unpack_from(f"<{n}I", body, cur)) if n else []
    cur += 4 * n
    offsets = list(struct.unpack_from(f"<{n + 1}I", body, cur))
    cur += 4 * (n + 1)
    assert all(p > 0 for p in pagenos), pagenos
    assert all(a < b for a, b in zip(pagenos, pagenos[1:], strict=False)), pagenos
    assert offsets[0] == 0, offsets
    assert all(a <= b for a, b in zip(offsets, offsets[1:], strict=False)), offsets

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

    # v7 chain invariants (the spec's "Validation" section): heads strictly
    # ascending (the bisect key), the offset table well-formed, and every chain
    # non-empty and stored head-first.  The ``lo < hi`` term short-circuits the
    # head-first index, so a malformed empty chain fails the assert (not IndexError).
    assert all(a < b for a, b in zip(chain_heads, chain_heads[1:], strict=False)), (
        chain_heads
    )
    assert chain_offsets[0] == 0 and chain_offsets[-1] == len(flat)
    assert all(
        chain_offsets[i] < chain_offsets[i + 1]
        and flat[chain_offsets[i]] == chain_heads[i]
        for i in range(chain_count)
    )

    pages_blob = body[cur:]
    assert offsets[-1] == len(pages_blob), (offsets[-1], len(pages_blob))
    stored = [pages_blob[offsets[i] : offsets[i + 1]] for i in range(n)]
    for pageno, page in zip(pagenos, stored, strict=True):
        assert len(reconstruct_page(page, pageno, page_size)) == page_size
    return ParsedSidecar(version, db_tag, page_size, n, pagenos, stored, chains)


def reconstruct_page(stored: bytes, pageno: int, page_size: int) -> bytes:
    """Reader-side inverse of the writer's gap stripping: splice the zero gap
    back to recover a full ``page_size`` page."""
    base = 100 if pageno == 1 else 0
    ptype = stored[base]
    if ptype not in (2, 5, 10, 13):
        return stored
    hdr = 12 if ptype in (2, 5) else 8
    n_cells = int.from_bytes(stored[base + 3 : base + 5], "big")
    cca = int.from_bytes(stored[base + 5 : base + 7], "big") or 65536
    cpa_end = base + hdr + 2 * n_cells
    return stored[:cpa_end] + b"\x00" * (cca - cpa_end) + stored[cpa_end:]
