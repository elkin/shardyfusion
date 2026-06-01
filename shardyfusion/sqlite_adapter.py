"""SQLite shard adapter — write locally, upload once, read from S3.

KV adapter (``SqliteFactory`` / ``SqliteAdapter``): satisfies the
``DbAdapter`` / ``ShardReader`` protocols.  Stores data in a
``WITHOUT ROWID`` ``kv(k BLOB PK, v BLOB)`` table.

Lifecycle: build a local SQLite DB → upload the single ``.db`` file to
S3 on close → reader downloads (or range-reads) the file.

Reader tiers:

* **Download-and-cache** (default): one S3 GET, then all lookups are local.
* **Range-read VFS** (optional, requires ``apsw``): translates SQLite page
  reads into S3 ``Range`` requests with an LRU page cache.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import struct
import types
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, runtime_checkable

from ._local_snapshot_cache import ensure_cached_snapshot
from ._sqlite_vfs import S3ReadOnlyFile, S3VfsError, create_apsw_vfs
from .credentials import CredentialProvider, S3Credentials
from .errors import ConfigValidationError, ShardyfusionError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .sqlite_page_size import (
    SUPPORTED_PAGE_SIZES,
    CellShape,
    recommend_page_size_for_cells,
)
from .storage import ObstoreBackend, create_s3_store, parse_s3_url
from .type_defs import Manifest, S3ConnectionOptions

_logger = get_logger(__name__)

_DB_FILENAME = "shard.db"
_DB_IDENTITY_FILENAME = "shard.identity.json"
_SIDECAR_FILENAME = "shard.sidecar"
# Vendor-neutral 4-byte magic so the page-cache sidecar format can be
# consumed outside shardyfusion.  (v1-v4 used b"SFBTM\x00\x00\x00" under the
# old ``shard.btreemeta`` name.)
_SIDECAR_MAGIC = b"SQPC"
# Uncompressed prefix (magic + u8 version + u64 body_size + the .db object
# tag for a correctness binding) then one zstd frame (with content checksum)
# over a metadata-first body: ``page_size | n | pagenos | offsets |
# chain_count | chain_heads | chain_offsets | chain_pages | gap-stripped
# pages``.  Pages are gap-stripped — the unallocated middle of each B-tree
# page is physically dropped and the reader splices zeros back in — which
# roughly halves the decompressed/resident size.  The ``(pageno, offset)``
# index is split into the ``pagenos`` bisect key plus pages-relative
# ``offsets`` (stripped pages are variable-length); overflow chains are a
# parallel CSR triple (sorted ``chain_heads`` + ``chain_offsets`` + flat
# ``chain_pages``) so a reader bisects chain heads and slices the page list
# with no dict.  Format documented in
# ``docs/reference/sqlite-sidecar-format.md``.
_SIDECAR_FORMAT_VERSION = 7
# zstd default level.  Btree pages compress ~12× at level 3 with
# sub-millisecond cost; higher levels squeeze a few extra percent for
# significantly more time and aren't worth it on the writer hot path.
_SIDECAR_ZSTD_LEVEL = 3


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SqliteAdapterError(ShardyfusionError):
    """SQLite adapter error (non-retryable)."""

    retryable = False


class SidecarUnavailableError(SqliteAdapterError):
    """Raised when APSW, ``dbstat``, ``zstandard``, or the on-disk file's
    invariants make raw-file sidecar extraction unsafe.

    The ``args[0]`` reason string distinguishes causes
    (``apsw_not_installed`` | ``dbstat_unavailable`` |
    ``zstandard_not_installed`` | ``unsupported_journal_mode`` |
    ``page_size_not_int`` | ``page_read_short`` |
    ``reserved_bytes_unsupported``) so that ``log_event`` calls can record
    the specific cause for diagnostics.
    """

    retryable = False


# Journal modes whose committed state lives entirely in the main ``.db``
# file once the writer's connection is closed.  Sidecar extraction reads
# pages by direct file I/O at deterministic offsets, which is safe only
# when no committed pages are parked in a sibling artifact (``-wal``).
# WAL modes can park committed-but-not-yet-checkpointed pages in the WAL
# sidecar; raw file I/O would silently miss them.  See
# ``docs/reference/sqlite-sidecar-format.md``.
_SIDECAR_SAFE_JOURNAL_MODES: frozenset[str] = frozenset(
    {"off", "delete", "memory", "truncate", "persist"}
)


# ---------------------------------------------------------------------------
# Adaptive page_size (post-write VACUUM)
# ---------------------------------------------------------------------------


# Fraction of values that should fit inline; the picker chooses the
# smallest supported page size whose inline threshold accommodates this
# percentile.  0.95 is a deliberate trade-off: tight enough to
# eliminate overflow for the vast majority of values, loose enough that
# a long tail of outliers does not blow up to the largest page size.
_AUTO_PAGE_SIZE_PERCENTILE: float = 0.95


def _maybe_repage_to_auto(
    db_path: Path,
    *,
    db_url: str,
    connection_opener: Callable[[Path], sqlite3.Connection] | None = None,
    extra_cells: list[CellShape] | None = None,
) -> None:
    """Rewrite ``db_path`` in-place at the page size recommended for its values.

    The rewrite uses ``PRAGMA page_size = N; VACUUM;`` on a fresh
    connection; ``VACUUM`` honours the pragma even when it changes the
    on-disk page size.  No-op when the recommended size matches the
    current size.

    ``connection_opener`` is called with the DB path and must return an
    open ``sqlite3.Connection`` configured with whatever extensions are
    needed to read every virtual table the file owns.  The default opens
    a plain :func:`sqlite3.connect`; the sqlite-vec adapter overrides
    this to load the vec extension so ``VACUUM`` can resolve
    ``vec_index``.

    ``extra_cells`` is an optional list of additional :class:`CellShape`
    entries to size alongside the kv cell.  Used by
    :class:`SqliteVecAdapter` to fold the ``vec_index`` leaf cell
    (``4*dim`` payload, rowid-varint key) into the recommendation so the
    embedding fits inline on the chosen page.  KV-only adapters pass
    ``None`` (default).

    If the ``kv`` table is empty AND no ``extra_cells`` were supplied,
    the function returns without rewriting (the default 4 KiB size is
    correct for an empty file).  When ``extra_cells`` are present, the
    picker runs against just those cells even when ``kv`` is empty —
    this matters for vec-only workloads where every embedding row would
    otherwise spill to overflow.

    Errors are logged and re-raised — the caller's ``seal()`` already
    treats this as part of the finalize step.
    """

    if connection_opener is None:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
    else:
        conn = connection_opener(db_path)
    try:
        ((current_page_size,),) = conn.execute("PRAGMA page_size").fetchall()
        current_page_size = int(current_page_size)

        ((row_count,),) = conn.execute("SELECT count(*) FROM kv").fetchall()
        row_count = int(row_count)
        cells: list[CellShape] = []
        p95_value_bytes = 0
        max_key_bytes = 0
        if row_count > 0:
            # Nearest-rank percentile via OFFSET: rank = ceil(p * N) (1-indexed)
            # → OFFSET = rank - 1.  Using `int(...) - 1` would round toward zero
            # and pick one rank below the true p95 whenever `p * N` is non-
            # integer (e.g. N=21, 41, 73, ...).
            offset = max(0, math.ceil(row_count * _AUTO_PAGE_SIZE_PERCENTILE) - 1)
            ((p95_value_bytes,),) = conn.execute(
                "SELECT length(v) FROM kv ORDER BY length(v) LIMIT 1 OFFSET ?",
                (offset,),
            ).fetchall()
            ((max_key_bytes,),) = conn.execute(
                "SELECT max(length(k)) FROM kv"
            ).fetchall()
            p95_value_bytes = int(p95_value_bytes)
            max_key_bytes = int(max_key_bytes or 0)
            cells.append(
                CellShape(payload_bytes=p95_value_bytes, max_key_bytes=max_key_bytes)
            )
        if extra_cells:
            cells.extend(extra_cells)
        if not cells:
            # KV-only adapter with an empty file: default page size stands.
            return

        target = recommend_page_size_for_cells(cells)
        if target == current_page_size:
            return

        conn.execute(f"PRAGMA page_size = {target}")
        conn.execute("VACUUM")
        event_fields: dict[str, Any] = {
            "db_url": db_url,
            "from_page_size": current_page_size,
            "to_page_size": int(target),
            "kv_row_count": row_count,
            "extra_cells": [
                (int(c.payload_bytes), int(c.max_key_bytes))
                for c in (extra_cells or [])
            ],
        }
        if row_count > 0:
            # Only meaningful when kv actually had rows.  On a vec-only
            # workload (row_count=0 with extra_cells) the 0 values
            # would falsely suggest "kv had zero-byte rows".
            event_fields["kv_p95_value_bytes"] = p95_value_bytes
            event_fields["kv_max_key_bytes"] = max_key_bytes
        log_event(
            "sqlite_adapter_repaged",
            level=logging.DEBUG,
            logger=_logger,
            **event_fields,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Page-cache sidecar extraction
# ---------------------------------------------------------------------------

# SQLite B-tree page-type byte (the page header's first byte): interior
# index/table are 2/5, leaf index/table are 10/13.  Interior pages carry a
# 4-byte right-child pointer, so their header is 12 bytes vs 8 for leaves.
_INTERIOR_PAGE_TYPES: frozenset[int] = frozenset({2, 5})
_BTREE_PAGE_TYPES: frozenset[int] = _INTERIOR_PAGE_TYPES | {10, 13}

# Byte 20 of the 100-byte DB header is "bytes reserved per page".  Non-zero
# means a checksum/encryption VFS may cover the whole page (including the
# gap), so gap stripping would be unsafe.
_RESERVED_BYTES_OFFSET = 20


def _strip_page_gap(slab: bytes, pageno: int, page_size: int) -> bytes:
    """Return ``slab`` with its unallocated middle gap physically removed.

    A SQLite B-tree page is laid out as an 8/12-byte header, a 2-byte-per-cell
    *cell pointer array* growing forward, a contiguous *unallocated gap*, then
    the *cell content area* growing backward to the end of the page.  SQLite
    navigates via the pointer array and never reads the gap, so the writer
    drops it here; the reader splices zeros back in (see ``_reconstruct_page``
    in the format spec / tests) to recover a page SQLite treats as identical.
    Interior pages are ~50% gap, so this roughly halves the decompressed
    sidecar size.

    Header fields are big-endian.  ``base`` is 100 on page 1 (it carries the
    100-byte DB header first), else 0.  A non-B-tree page (page-type byte not
    in :data:`_BTREE_PAGE_TYPES`) is returned unchanged.  A full page with no
    free gap (``cca == cpa_end``) is also returned unchanged.
    """
    base = 100 if pageno == 1 else 0
    ptype = slab[base]
    if ptype not in _BTREE_PAGE_TYPES:
        return slab
    hdr = 12 if ptype in _INTERIOR_PAGE_TYPES else 8
    n_cells = int.from_bytes(slab[base + 3 : base + 5], "big")
    # A stored cell-content-area start of 0 encodes 65536 (only valid when
    # page_size == 65536).
    cca = int.from_bytes(slab[base + 5 : base + 7], "big") or 65536
    cpa_end = base + hdr + 2 * n_cells
    # Drop the unallocated gap [cpa_end, cca).
    return slab[:cpa_end] + slab[cca:]


def build_sidecar_frame(
    db_path: Path,
    *,
    db_bytes: bytes | None = None,
) -> tuple[bytes, int]:
    """Build the v7 page-cache sidecar *frame* for a finalized SQLite DB.

    Returns ``(frame, decompressed_size)`` where ``frame`` is the compressed
    zstd body (no uncompressed prefix) and ``decompressed_size`` is exactly
    ``len(body)`` — the number of bytes a reader gets after zstd-decompressing
    the frame.  Call :func:`frame_to_sidecar` to prepend the ``SQPC`` prefix
    (magic, version, and the ``.db`` object tag) and obtain a complete sidecar
    object; :func:`extract_sidecar` does both in one call.

    The sidecar bundles every interior B-tree page plus every
    ``sqlite_master`` / ``sqlite_schema`` page — each **gap-stripped** (see
    :func:`_strip_page_gap`) — so a range-read reader can fetch them once on
    shard open and reconstruct them locally, eliminating the per-query round
    trips needed to walk B-tree internals.  It also carries an overflow-chain
    CSR index (sorted ``chain_heads`` + ``chain_offsets`` + flat ``chain_pages``)
    so the reader can bisect a chain head and prefetch the whole chain in one
    coalesced range request.

    Page identification is delegated to SQLite's ``dbstat`` virtual table
    (via APSW for reliable availability — APSW bundles a SQLite built with
    ``SQLITE_ENABLE_DBSTAT_VTAB``).  APSW and ``zstandard`` are lazy-imported
    so the writer base install does not require them; if either (or
    ``dbstat``) is unavailable, raises :class:`SidecarUnavailableError` which
    the adapter treats as a soft skip.

    ``db_bytes`` is the already-read file contents — sharing it avoids reading
    the finalized file twice.  Pass ``None`` to read ``db_path`` here.

    Body (the zstd-decompressed frame), in order:

    * ``u32`` page size
    * ``u32`` page count ``N``
    * ``N * u32`` pagenos, sorted ascending (the bisect key)
    * ``(N+1) * u32`` offsets into the trailing pages blob — page ``i`` is
      ``pages[off[i]:off[i+1]]``; the final entry is the blob length
    * ``u32`` overflow chain count ``C``, then the CSR triple — ``C * u32``
      ``chain_heads`` (sorted), ``(C+1) * u32`` ``chain_offsets`` (page-number
      units), and ``M * u32`` ``chain_pages`` (every chain head-first,
      ``M = chain_offsets[-1]``)
    * the gap-stripped pages, concatenated in pageno order
    """

    try:
        import apsw  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SidecarUnavailableError("apsw_not_installed") from exc

    try:
        import zstandard  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SidecarUnavailableError("zstandard_not_installed") from exc

    conn = apsw.Connection(
        f"file:{db_path}?mode=ro",
        flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
    )
    try:
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT pageno FROM dbstat LIMIT 0").fetchall()
        except apsw.SQLError as exc:
            raise SidecarUnavailableError("dbstat_unavailable") from exc

        # Guard against future writer changes that would silently break the
        # raw-file extraction path (e.g. switching to WAL mode without a
        # forced checkpoint).  PRAGMA journal_mode reflects the file's
        # persistent setting even on a read-only connection.
        journal_rows = cursor.execute("PRAGMA journal_mode").fetchall()
        mode_value = journal_rows[0][0] if journal_rows else None
        if (
            not isinstance(mode_value, str)
            or mode_value.lower() not in _SIDECAR_SAFE_JOURNAL_MODES
        ):
            raise SidecarUnavailableError("unsupported_journal_mode")

        ((page_size_raw,),) = cursor.execute("PRAGMA page_size").fetchall()
        if not isinstance(page_size_raw, int):
            raise SidecarUnavailableError("page_size_not_int")
        page_size = page_size_raw

        # The schema btree is named ``sqlite_schema`` in modern SQLite
        # (renamed from ``sqlite_master`` in 3.33); accept both for portability
        # across SQLite versions.
        rows = cursor.execute(
            "SELECT pageno FROM dbstat "
            " WHERE pagetype = 'internal' "
            "    OR name IN ('sqlite_master', 'sqlite_schema') "
            " ORDER BY pageno"
        ).fetchall()
        page_nums: list[int] = sorted(
            {int(r[0]) for r in rows if isinstance(r[0], int)}
        )

        overflow_rows = cursor.execute(
            "SELECT pageno FROM dbstat WHERE pagetype = 'overflow' ORDER BY pageno"
        ).fetchall()
        overflow_pages: list[int] = sorted(
            {int(r[0]) for r in overflow_rows if isinstance(r[0], int)}
        )
    finally:
        conn.close()

    if db_bytes is None:
        db_bytes = db_path.read_bytes()

    # Gap stripping is only safe when no per-page reserved tail (checksum /
    # encryption VFS) could cover the dropped gap region.  The writer creates
    # DBs with the default 0 reserved, so this is a defensive skip.
    reserved = (
        db_bytes[_RESERVED_BYTES_OFFSET]
        if len(db_bytes) > _RESERVED_BYTES_OFFSET
        else 0
    )
    if reserved:
        raise SidecarUnavailableError("reserved_bytes_unsupported")

    stripped: list[bytes] = []
    for pgno in page_nums:
        offset = (pgno - 1) * page_size
        slab = db_bytes[offset : offset + page_size]
        if len(slab) != page_size:
            raise SidecarUnavailableError("page_read_short")
        stripped.append(_strip_page_gap(slab, pgno, page_size))

    chains = _enumerate_overflow_chains(
        overflow_pages=overflow_pages,
        db_bytes=db_bytes,
        page_size=page_size,
    )

    n = len(page_nums)
    # Offsets into the trailing pages blob (pages-relative), N+1 entries with
    # a sentinel final offset == blob length.  Stripped pages are
    # variable-length, so these offsets are load-bearing (not derivable from
    # a uniform stride).
    offsets: list[int] = [0]
    for s in stripped:
        offsets.append(offsets[-1] + len(s))

    # Overflow chains as a CSR triple — a sorted head key, an offset table, and
    # a flat page list — mirroring pagenos/offsets/pages.  Heads already come
    # out sorted from _enumerate_overflow_chains, so a reader bisects
    # chain_heads and slices chain_pages directly (no dict).  chain_offsets is
    # in page-number units (entries are fixed-width u32), unlike the byte-unit
    # ``offsets`` above.
    chain_offsets: list[int] = [0]
    flat_pages: list[int] = []
    for chain in chains:
        flat_pages.extend(chain)
        chain_offsets.append(len(flat_pages))
    chain_bytes_parts: list[bytes] = [
        struct.pack("<I", len(chains)),
        struct.pack(f"<{len(chains)}I", *(c[0] for c in chains)) if chains else b"",
        struct.pack(f"<{len(chain_offsets)}I", *chain_offsets),
        struct.pack(f"<{len(flat_pages)}I", *flat_pages) if flat_pages else b"",
    ]

    body = b"".join(
        [
            struct.pack("<II", page_size, n),
            struct.pack(f"<{n}I", *page_nums) if n else b"",
            struct.pack(f"<{n + 1}I", *offsets),
            b"".join(chain_bytes_parts),
            b"".join(stripped),
        ]
    )
    compressor = zstandard.ZstdCompressor(
        level=_SIDECAR_ZSTD_LEVEL, write_checksum=True
    )
    return compressor.compress(body), len(body)


def frame_to_sidecar(
    frame: bytes, db_tag: str | None = None, body_size: int | None = None
) -> bytes:
    """Prepend the uncompressed ``SQPC`` prefix to a compressed sidecar frame.

    Wire prefix: ``magic(4) + u8 version + u64 body_size + u8 tag_len + tag``,
    then ``frame``.  ``db_tag`` is the ``.db`` object-version token (S3 ETag)
    bound into the sidecar so a reader only trusts it when the live ``.db``
    tag matches.  A ``None``/non-str (or over-255-byte) tag yields an unbound
    sidecar (``tag_len == 0``), which a correctness-strict reader declines.
    ``body_size`` is the uncompressed size of the sidecar body (the
    decompressed zstd frame); when ``None`` it is written as ``0``.
    """
    tag_bytes = db_tag.encode("utf-8") if isinstance(db_tag, str) else b""
    if len(tag_bytes) > 255:
        tag_bytes = b""
    body_size_val = body_size if body_size is not None else 0
    return b"".join(
        [
            _SIDECAR_MAGIC,
            _SIDECAR_FORMAT_VERSION.to_bytes(1, "little"),
            body_size_val.to_bytes(8, "little"),
            len(tag_bytes).to_bytes(1, "little"),
            tag_bytes,
            frame,
        ]
    )


def extract_sidecar(
    db_path: Path,
    *,
    db_bytes: bytes | None = None,
    db_tag: str | None = None,
) -> bytes:
    """Read a finalized SQLite DB and return the complete v7 sidecar object.

    Thin wrapper over :func:`build_sidecar_frame` + :func:`frame_to_sidecar`,
    kept for callers/tests that want the whole blob (prefix + frame) in one
    call.  See :func:`build_sidecar_frame` for the body layout and the
    lazy-import / ``SidecarUnavailableError`` contract.
    """
    frame, size = build_sidecar_frame(db_path, db_bytes=db_bytes)
    return frame_to_sidecar(frame, db_tag, body_size=size)


def _enumerate_overflow_chains(
    *,
    overflow_pages: Sequence[int],
    db_bytes: bytes,
    page_size: int,
) -> list[list[int]]:
    """Reconstruct every overflow chain from raw page successor pointers.

    Each overflow page starts with a 4-byte big-endian ``next-pageno``
    field (``0`` marks the end of the chain).  A chain head is any
    overflow page that is not referenced as a successor by any other
    overflow page (i.e. it is referenced from a leaf cell instead).
    Returns a list of chains; each chain is ``[head, p1, p2, ...]`` in
    traversal order.

    Cycles are defensively short-circuited: if a successor walk re-enters
    a page already visited in the current chain, the walk stops there.
    This should never happen in a well-formed SQLite file but the cost
    of the guard is negligible.
    """

    if not overflow_pages:
        return []

    successor: dict[int, int] = {}
    overflow_set = set(overflow_pages)
    for pgno in overflow_pages:
        offset = (pgno - 1) * page_size
        next_bytes = db_bytes[offset : offset + 4]
        if len(next_bytes) != 4:
            raise SidecarUnavailableError("page_read_short")
        next_pageno = int.from_bytes(next_bytes, "big")
        # 0 = end of chain; any non-overflow successor is treated as
        # malformed and stops the walk.
        if next_pageno and next_pageno in overflow_set:
            successor[pgno] = next_pageno

    successors_set = set(successor.values())
    heads = sorted(overflow_set - successors_set)

    chains: list[list[int]] = []
    for head in heads:
        chain: list[int] = [head]
        seen = {head}
        cur = head
        while cur in successor:
            nxt = successor[cur]
            if nxt in seen:
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        chains.append(chain)
    return chains


def maybe_upload_sidecar(
    *,
    backend: ObstoreBackend,
    db_url: str,
    db_path: Path,
    db_bytes: bytes,
    db_tag: str | None = None,
    frame: bytes | None = None,
    body_size: int | None = None,
) -> None:
    """Best-effort upload of the page-cache sidecar.

    Never raises: extraction or upload failures are logged and the caller
    proceeds.  Used by both ``SqliteAdapter.close()`` and
    ``SqliteVecAdapter.close()``.

    When ``frame`` (a pre-built compressed frame from :func:`build_sidecar_frame`,
    typically produced at ``seal()``) is supplied, it is wrapped with the
    ``db_tag`` prefix and uploaded directly — no re-extraction.  Otherwise the
    sidecar is extracted here from ``db_path`` / ``db_bytes``.  ``db_tag`` is the
    ``.db`` object's storage version token (S3 ETag); it binds the sidecar to the
    live ``.db``.  The caller must upload the ``.db`` first so this tag is
    available.
    """
    if frame is not None:
        payload = frame_to_sidecar(frame, db_tag, body_size=body_size)
    else:
        try:
            payload = extract_sidecar(db_path, db_bytes=db_bytes, db_tag=db_tag)
        except SidecarUnavailableError as exc:
            log_event(
                "sqlite_sidecar_unsupported",
                level=logging.DEBUG,
                logger=_logger,
                db_url=db_url,
                reason=str(exc.args[0]) if exc.args else "unknown",
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            log_failure(
                "sqlite_sidecar_failed",
                severity=FailureSeverity.TRANSIENT,
                logger=_logger,
                error=exc,
                db_url=db_url,
                stage="extract",
            )
            return

    sidecar_key = f"{db_url.rstrip('/')}/{_SIDECAR_FILENAME}"
    try:
        backend.put(
            sidecar_key,
            payload,
            content_type="application/octet-stream",
        )
    except Exception as exc:
        log_failure(
            "sqlite_sidecar_failed",
            severity=FailureSeverity.TRANSIENT,
            logger=_logger,
            error=exc,
            db_url=db_url,
            stage="upload",
        )
        return

    log_event(
        "sqlite_sidecar_uploaded",
        level=logging.DEBUG,
        logger=_logger,
        db_url=db_url,
        sidecar_bytes=len(payload),
    )


def _read_and_build_sidecar(
    db_path: Path, *, db_url: str
) -> tuple[bytes | None, bytes | None, int | None]:
    """Best-effort: read the finalized DB once and build its page-cache sidecar.

    Returns ``(db_bytes, frame, decompressed_size)``:

    * read failure → ``(None, None, None)``;
    * read ok but sidecar unavailable/failed → ``(db_bytes, None, None)``;
    * success → ``(db_bytes, frame, decompressed_size)``.

    Never raises — a sidecar problem must not fail ``seal()`` / publish, and the
    caller still uploads ``db_bytes`` as the ``.db`` even when ``frame`` is
    ``None``.  Shared by ``SqliteAdapter`` and ``SqliteVecAdapter`` so the
    seal-time build lives in one place.
    """
    try:
        db_bytes = db_path.read_bytes()
    except OSError as exc:  # pragma: no cover - defensive
        log_event(
            "sqlite_sidecar_unsupported",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
            reason=f"db_read_failed:{exc}",
        )
        return None, None, None
    try:
        frame, size = build_sidecar_frame(db_path, db_bytes=db_bytes)
    except SidecarUnavailableError as exc:
        log_event(
            "sqlite_sidecar_unsupported",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
            reason=str(exc.args[0]) if exc.args else "unknown",
        )
        return db_bytes, None, None
    except Exception as exc:  # pragma: no cover - defensive
        log_failure(
            "sqlite_sidecar_failed",
            severity=FailureSeverity.TRANSIENT,
            logger=_logger,
            error=exc,
            db_url=db_url,
            stage="extract",
        )
        return db_bytes, None, None
    return db_bytes, frame, size


# ---------------------------------------------------------------------------
# Writer: KV adapter (Layer 1)
# ---------------------------------------------------------------------------


# ``page_size`` accepts either a concrete SQLite page size or the
# ``"auto"`` sentinel.  Under ``"auto"`` the adapter opens at the default
# 4096-byte size, then after :meth:`SqliteAdapter.seal` analyses the
# value-size distribution in the finalized DB and rewrites it in-place
# (``PRAGMA page_size = N; VACUUM;``) if a larger page would have kept
# the bulk of values inline.  Trade-offs in :mod:`sqlite_page_size` and
# the adapter docstring.  Mutually exclusive with
# :attr:`shardyfusion.config.KeyValueWriteConfig.profile_value_sizes_for_page_size`
# — combining the two raises ``ConfigValidationError`` at config build.
PageSizeMode = int | Literal["auto"]


def _validate_factory_page_size(page_size: PageSizeMode) -> PageSizeMode:
    if isinstance(page_size, str):
        if page_size != "auto":
            raise ConfigValidationError(
                f"page_size string must be 'auto', got {page_size!r}"
            )
        return page_size
    page_size = int(page_size)
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise ConfigValidationError(
            f"page_size {page_size!r} not in supported sizes {SUPPORTED_PAGE_SIZES}"
        )
    return page_size


@dataclass(slots=True)
class SqliteFactory:
    """Picklable factory that builds local SQLite KV shards.

    ``page_size`` is one of the supported SQLite page sizes (4096, 8192,
    16384, 32768, 65536) or the string ``"auto"``.  Larger pages raise
    the inline-payload threshold so large values stay on the leaf page
    instead of spilling into overflow chains (one extra S3 GET per page
    under the range-read reader).  Trade-offs:

    * **Pro** — fewer S3 GETs per large-value lookup; higher B-tree
      fanout shrinks the tree depth; range scans hit fewer leaves.
    * **Con** — every cache miss fetches ``page_size`` bytes, so small-
      value point reads waste bandwidth on the larger payload; the
      reader page cache costs proportionally more memory per slot.

    Under ``page_size="auto"`` the adapter opens at 4 KB, observes the
    on-disk value-size distribution at seal time, and rewrites the file
    in-place at the recommended size via
    :func:`shardyfusion.sqlite_page_size.recommend_page_size_for_cells`
    (which sizes the kv cell and, in the unified vec variant, the
    ``vec_index`` cell together).  This doubles local I/O on rewrite
    but adds no S3 cost; callers that can pre-compute the percentile
    upstream should pass an explicit int on the factory's
    ``page_size`` instead.

    ``emit_sidecar`` (default ``True``) controls whether each
    finalized shard uploads a sibling ``shard.sidecar`` artifact
    bundling all interior B-tree pages plus every ``sqlite_master`` page
    and the overflow-chain CSR index.  Reader-side range-mode consumers can
    fetch this once on shard open and pin those pages for the lifetime
    of the shard reader.

    The sidecar requires APSW (already declared in the ``[sqlite-range]``
    extra) and a SQLite build with ``SQLITE_ENABLE_DBSTAT_VTAB``.  If
    either is unavailable the writer logs a debug-level event and skips
    the sidecar — the main ``shard.db`` upload proceeds unchanged.
    """

    page_size: PageSizeMode = 4096
    cache_size_pages: int = -2000  # negative = KiB, so ~8 MB
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None
    emit_sidecar: bool = True

    def __post_init__(self) -> None:
        self.page_size = _validate_factory_page_size(self.page_size)

    def vec_payload_bytes_in_kv_db(self) -> int:
        """Per-row embedding payload stored alongside kv in the same .db.

        ``SqliteFactory`` is KV-only: any vector data is written to a
        sidecar by a wrapping :class:`CompositeFactory`, not into this
        adapter's SQLite file.  Returns ``0`` so page-size sizing budgets
        only the kv cell.
        """
        return 0

    def __call__(self, *, db_url: str, local_dir: Path) -> SqliteAdapter:
        return SqliteAdapter(
            db_url=db_url,
            local_dir=local_dir,
            page_size=self.page_size,
            cache_size_pages=self.cache_size_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
            emit_sidecar=self.emit_sidecar,
        )


class SqliteAdapter:
    """KV write-path adapter: builds SQLite DB locally, uploads to S3 on close.

    Satisfies the :class:`~shardyfusion.slatedb_adapter.DbAdapter` protocol.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        page_size: PageSizeMode = 4096,
        cache_size_pages: int = -2000,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
        emit_sidecar: bool = True,
    ) -> None:
        page_size = _validate_factory_page_size(page_size)
        cache_size_pages = int(cache_size_pages)

        self._page_size_mode: PageSizeMode = page_size
        # Connection always opens at a concrete int.  Under "auto" we
        # start at the smallest supported size and let seal() rewrite
        # the file in-place if a larger size would fit values inline.
        initial_page_size = 4096 if page_size == "auto" else int(page_size)

        self._db_url = db_url
        self._local_dir = local_dir
        self._db_path = local_dir / _DB_FILENAME
        self._uploaded = False
        self._closed = False
        self._sealed = False
        # Distinct from ``_sealed``: True once ``seal()`` has begun, even
        # if it later raises (e.g. post-write VACUUM fails).  ``close()``
        # uses this to refuse to upload a half-finalised file while
        # still permitting the documented ``write -> close()`` pattern
        # that skips ``seal()`` entirely.
        self._seal_attempted = False
        self._db_bytes = 0
        self._emit_sidecar = bool(emit_sidecar)
        # Page-cache sidecar built best-effort at seal(); cached here so close()
        # can upload it (and reuse the finalized file bytes) without re-reading.
        self._sidecar_frame: bytes | None = None
        self._sidecar_decompressed_bytes: int | None = None
        self._finalized_db_bytes: bytes | None = None
        self._s3_conn_opts = s3_connection_options
        self._s3_creds: S3Credentials | None = (
            credential_provider.resolve() if credential_provider else None
        )

        local_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        conn.execute(f"PRAGMA page_size = {initial_page_size}")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute(f"PRAGMA cache_size = {cache_size_pages}")
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv "
            "(k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID"
        )
        conn.execute("BEGIN")
        self._conn: sqlite3.Connection | None = conn

        log_event(
            "sqlite_adapter_opened",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
            page_size_mode=str(page_size),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        if self._conn is None:
            raise SqliteAdapterError("Adapter already closed")
        if self._sealed:
            raise SqliteAdapterError("Cannot write after seal")
        self._conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", pairs)

    def flush(self) -> None:
        pass  # no-op: local file, no WAL

    def seal(self) -> None:
        """Finalize the local SQLite DB before upload.

        Performs the trailing ``COMMIT`` + ``PRAGMA optimize``, closes
        the local connection, and captures the on-disk size for
        :meth:`db_bytes`. Earlier shardyfusion versions also returned a
        SHA-256 of the file as a content-addressed checkpoint id; that
        is no longer used (writers now stamp shards with an opaque
        UUID — see :func:`shardyfusion._checkpoint_id.generate_checkpoint_id`).

        Under ``page_size="auto"`` the closed file is reopened to scan
        the kv value-size distribution and rewritten in-place at the
        recommended page size when that differs from the initial 4 KB.
        """
        if self._sealed:
            raise SqliteAdapterError("Adapter already sealed")
        if self._seal_attempted:
            # seal() raised previously (e.g. VACUUM failure).  The adapter
            # is in an indeterminate state — refuse retry on the same
            # instance.  Distinct from "Adapter already closed" because
            # `_conn` was nulled inside the failing seal() call.
            raise SqliteAdapterError(
                "Adapter seal previously failed; cannot retry on the same "
                "instance — open a new adapter against the same db_url."
            )
        if self._conn is None:
            raise SqliteAdapterError("Adapter already closed")
        self._seal_attempted = True
        self._conn.execute("COMMIT")
        self._conn.execute("PRAGMA optimize")
        self._conn.close()
        self._conn = None

        # Run any post-write VACUUM BEFORE marking sealed, so that a
        # repage failure leaves ``_sealed=False`` and ``close()`` skips
        # the upload — otherwise the un-repaged DB ships silently with
        # no signal that the chosen strategy was abandoned.
        if self._page_size_mode == "auto":
            _maybe_repage_to_auto(self._db_path, db_url=self._db_url)

        self._sealed = True
        self._db_bytes = self._db_path.stat().st_size
        # Build the page-cache sidecar now (best-effort) so its exact decompressed
        # size can ride the manifest pipeline alongside db_bytes; close() reuses
        # the cached frame + file bytes for upload.
        if self._emit_sidecar:
            self._build_sidecar_for_upload()
        log_event(
            "sqlite_adapter_sealed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=self._db_url,
        )

    def db_bytes(self) -> int:
        return self._db_bytes

    def sidecar_decompressed_bytes(self) -> int | None:
        """Exact decompressed size of this shard's page-cache sidecar, or
        ``None`` when none was produced (``emit_sidecar`` off, extraction
        skipped/unavailable, or ``seal()`` not called)."""
        return self._sidecar_decompressed_bytes

    def _build_sidecar_for_upload(self) -> None:
        """Build the page-cache sidecar at ``seal()`` (best-effort), caching the
        frame and the finalized file bytes for :meth:`close` and recording the
        exact decompressed size.  Delegates to :func:`_read_and_build_sidecar`."""
        (
            self._finalized_db_bytes,
            self._sidecar_frame,
            self._sidecar_decompressed_bytes,
        ) = _read_and_build_sidecar(self._db_path, db_url=self._db_url)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._conn is not None:
                try:
                    self._conn.execute("COMMIT")
                except sqlite3.OperationalError:
                    pass
                self._conn.close()
                self._conn = None

            if self._db_path.exists() and not self._uploaded:
                if self._seal_attempted and not self._sealed:
                    # seal() was attempted but did not complete (e.g.
                    # post-write VACUUM raised).  Refuse to upload AND
                    # raise so the writer learns the shard was not
                    # published.  ``_closed`` is set first so any
                    # subsequent close() call is a no-op rather than
                    # re-raising.  The ``write -> close()`` pattern
                    # (seal never attempted) still uploads — that
                    # branch is below.
                    log_event(
                        "sqlite_adapter_close_unsealed_skip_upload",
                        level=logging.WARNING,
                        logger=_logger,
                        db_url=self._db_url,
                    )
                    self._closed = True
                    raise SqliteAdapterError(
                        "seal() was attempted but did not complete; "
                        "refusing to upload an un-finalised shard.  See "
                        "preceding seal() exception for the root cause."
                    )
                s3_key = f"{self._db_url.rstrip('/')}/{_DB_FILENAME}"
                bucket, _ = parse_s3_url(s3_key)
                store = create_s3_store(
                    bucket=bucket,
                    credentials=self._s3_creds,
                    connection_options=self._s3_conn_opts,
                )
                backend = ObstoreBackend(store)
                # Reuse the bytes read by seal()'s sidecar build when available
                # so the finalized file is read only once.
                db_bytes = (
                    self._finalized_db_bytes
                    if self._finalized_db_bytes is not None
                    else self._db_path.read_bytes()
                )
                # Upload the .db first so the sidecar can bind to its object
                # version (ETag); a reader refuses a sidecar whose tag does
                # not match the live .db.
                backend.put(
                    s3_key,
                    db_bytes,
                    content_type="application/x-sqlite3",
                )
                self._uploaded = True
                log_event(
                    "sqlite_adapter_uploaded",
                    level=logging.DEBUG,
                    logger=_logger,
                    db_url=self._db_url,
                )
                # Upload the sidecar: prefer the frame built at seal() (whose
                # size is already recorded); fall back to building here only for
                # the write->close()-without-seal() path.  A sealed shard whose
                # seal-time build was skipped/failed is not retried.
                if self._emit_sidecar and (
                    self._sidecar_frame is not None or not self._sealed
                ):
                    try:
                        db_tag = backend.head(s3_key)
                    except Exception as exc:
                        # An unbound sidecar is still safe (a reader declines
                        # it), so a HEAD hiccup must not abort close().
                        log_event(
                            "sqlite_sidecar_tag_unavailable",
                            level=logging.DEBUG,
                            logger=_logger,
                            db_url=self._db_url,
                            error=str(exc),
                        )
                        db_tag = None
                    maybe_upload_sidecar(
                        backend=backend,
                        db_url=self._db_url,
                        db_path=self._db_path,
                        db_bytes=db_bytes,
                        db_tag=db_tag,
                        frame=self._sidecar_frame,
                        body_size=self._sidecar_decompressed_bytes,
                    )
                # Release the cached read bytes and compressed frame now that the
                # upload is done; the recorded size lives in
                # ``_sidecar_decompressed_bytes`` and is unaffected.
                self._finalized_db_bytes = None
                self._sidecar_frame = None
        except Exception as exc:
            log_failure(
                "sqlite_adapter_close_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise
        else:
            self._closed = True
        finally:
            log_event(
                "sqlite_adapter_closed",
                level=logging.DEBUG,
                logger=_logger,
                db_url=self._db_url,
            )


# ---------------------------------------------------------------------------
# Reader: download-and-cache (Tier 1)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteReaderFactory:
    """Picklable factory for the download-and-cache SQLite shard reader."""

    mmap_size: int = 268435456  # 256 MB
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> SqliteShardReader:
        del manifest  # unused in concrete factory
        return SqliteShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            mmap_size=self.mmap_size,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )


class SqliteShardReader:
    """Download-and-cache reader: one S3 GET, then pure local lookups.

    Satisfies the :class:`~shardyfusion.type_defs.ShardReader` protocol.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        mmap_size: int = 268435456,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._db_url = db_url
        self._identity_path = local_dir / _DB_IDENTITY_FILENAME
        self._checkpoint_id = checkpoint_id

        def _download() -> bytes:
            s3_key = f"{db_url.rstrip('/')}/{_DB_FILENAME}"
            bucket, _ = parse_s3_url(s3_key)
            creds = credential_provider.resolve() if credential_provider else None
            store = create_s3_store(
                bucket=bucket,
                credentials=creds,
                connection_options=s3_connection_options,
            )
            backend = ObstoreBackend(store)
            return backend.get(s3_key)

        self._db_path = ensure_cached_snapshot(
            local_dir=local_dir,
            db_filename=_DB_FILENAME,
            identity_filename=_DB_IDENTITY_FILENAME,
            expected_identity=self._expected_snapshot_identity(),
            downloader=_download,
        )

        mmap_size = int(mmap_size)
        conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.execute(f"PRAGMA mmap_size = {mmap_size}")
        conn.execute("PRAGMA cache_size = -8000")
        self._conn: sqlite3.Connection | None = conn

    def _expected_snapshot_identity(self) -> dict[str, str | None]:
        return {
            "db_url": self._db_url,
            "checkpoint_id": self._checkpoint_id,
        }

    # -- ShardReader protocol --

    def get(self, key: bytes) -> bytes | None:
        if self._conn is None:
            raise SqliteAdapterError("Reader already closed")
        row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Reader: async wrapper (download-and-cache)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncSqliteReaderFactory:
    """Async factory for the download-and-cache SQLite shard reader."""

    mmap_size: int = 268435456
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncSqliteShardReader:
        del manifest  # unused in concrete factory
        inner = await asyncio.to_thread(
            SqliteShardReader,
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            mmap_size=self.mmap_size,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )
        return AsyncSqliteShardReader(inner)


class AsyncSqliteShardReader:
    """Async wrapper around :class:`SqliteShardReader`.

    Delegates blocking ``sqlite3`` calls to a thread executor via
    :func:`asyncio.to_thread`.

    This wrapper does not guard against concurrent ``close()`` and
    ``get()`` calls.  Callers must ensure that ``close()`` is not
    invoked while ``get()`` operations are still in flight.  When used
    via :class:`~shardyfusion.reader.async_reader.AsyncShardedReader`,
    the reader's borrow-count mechanism provides this guarantee
    automatically.
    """

    def __init__(self, inner: SqliteShardReader) -> None:
        self._inner = inner

    async def get(self, key: bytes) -> bytes | None:
        return await asyncio.to_thread(self._inner.get, key)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)


# ---------------------------------------------------------------------------
# Reader: S3 range-read VFS (Tier 2, requires apsw)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteRangeReaderFactory:
    """Picklable factory for the range-read VFS reader.  Requires ``apsw``."""

    # 1024 slots × page_size; ~4 MB at the 4 KiB default but 16 MB at
    # 16 KiB pages and 64 MB at 64 KiB.  Range-mode readers that pin
    # ``page_size`` on the writer above the default should shrink this
    # to keep the per-shard cache budget in line.  ``0`` disables caching.
    page_cache_pages: int = 1024
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> SqliteRangeShardReader:
        del manifest  # unused in concrete factory
        return SqliteRangeShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            page_cache_pages=self.page_cache_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )


class SqliteRangeShardReader:
    """Range-read reader: fetches only the SQLite pages needed via S3 Range requests.

    Point lookups on a shard with 1M keys require ~3-4 page reads (B-tree
    depth).  An LRU page cache reduces repeated page fetches to zero for
    warm paths.

    Requires ``apsw`` (``pip install apsw``).  Satisfies the
    :class:`~shardyfusion.type_defs.ShardReader` protocol.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        page_cache_pages: int = 1024,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        try:
            import apsw  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise SqliteAdapterError(
                "apsw and obstore are required for the range-read VFS reader. "
                "Install via: pip install 'shardyfusion[sqlite-range]' "
                "or 'shardyfusion[sqlite-adaptive]' for the adaptive reader."
            ) from exc

        from .storage import parse_s3_url

        self._db_url = db_url
        self._conn: Any | None = None
        self._vfs: Any | None = None

        s3_key_full = f"{db_url.rstrip('/')}/{_DB_FILENAME}"
        bucket, key = parse_s3_url(s3_key_full)
        creds = credential_provider.resolve() if credential_provider else None

        try:
            self._s3_file = S3ReadOnlyFile(
                bucket=bucket,
                key=key,
                page_cache_pages=page_cache_pages,
                s3_connection_options=s3_connection_options,
                s3_credentials=creds,
            )
        except S3VfsError as exc:
            raise SqliteAdapterError(str(exc)) from exc

        # Register a unique VFS name for this reader instance
        import uuid

        vfs_name = f"s3range_{uuid.uuid4().hex}"
        self._vfs = create_apsw_vfs(vfs_name, self._s3_file)
        self._conn = apsw.Connection(
            f"file:{_DB_FILENAME}?mode=ro",
            flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
            vfs=vfs_name,
        )

    def get(self, key: bytes) -> bytes | None:
        conn = self._conn
        if conn is None:
            raise SqliteAdapterError("Reader is closed")
        cursor = conn.cursor()
        cursor.execute("SELECT v FROM kv WHERE k = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._vfs is not None:
            self._vfs.unregister()
            self._vfs = None


# ---------------------------------------------------------------------------
# Reader: async range-read wrapper
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncSqliteRangeReaderFactory:
    """Async factory for the range-read VFS reader."""

    page_cache_pages: int = 1024  # 0 disables caching
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncSqliteRangeShardReader:
        del manifest  # unused in concrete factory
        inner = await asyncio.to_thread(
            SqliteRangeShardReader,
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            page_cache_pages=self.page_cache_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )
        return AsyncSqliteRangeShardReader(inner)


class AsyncSqliteRangeShardReader:
    """Async wrapper around :class:`SqliteRangeShardReader`.

    This wrapper does not guard against concurrent ``close()`` and
    ``get()`` calls.  Callers must ensure that ``close()`` is not
    invoked while ``get()`` operations are still in flight.
    """

    def __init__(self, inner: SqliteRangeShardReader) -> None:
        self._inner = inner

    async def get(self, key: bytes) -> bytes | None:
        return await asyncio.to_thread(self._inner.get, key)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)


# ---------------------------------------------------------------------------
# Access-mode policy + factory selectors
# ---------------------------------------------------------------------------


SqliteAccessMode = Literal["download", "range", "auto"]
"""User-facing SQLite shard access mode.

"download" downloads each shard file once and serves all lookups locally
(cheap RAM/disk, expensive cold start).  "range" opens shards via the
APSW range-read VFS, fetching only the SQLite pages needed (cheap cold
start, more S3 GETs).  "auto" picks one of the two per snapshot based on
the published ``db_bytes`` distribution; see :func:`decide_access_mode`.
"""


# Default thresholds for ``auto`` resolution (used by readers when no
# explicit value is provided).
DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES = 16 * 1024 * 1024  # 16 MiB
DEFAULT_AUTO_TOTAL_BUDGET_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def decide_access_mode(
    *,
    db_bytes_per_shard: Sequence[int],
    per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
    total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
) -> Literal["download", "range"]:
    """Decide between ``download`` and ``range`` for a snapshot.

    Returns ``"range"`` when *any* shard's size is at or above
    ``per_shard_threshold``, or when the *cumulative* size is at or above
    ``total_budget``; otherwise ``"download"``. Both comparisons use ``>=``
    (a shard exactly equal to the threshold tips the decision toward
    ``range``).

    The two thresholds protect distinct cost dimensions:

    * ``per_shard_threshold`` — guards against a single oversized shard
      blowing local disk on cold start.
    * ``total_budget`` — guards against many small shards summing to a huge
      working set when the reader holds them all simultaneously.

    Args:
        db_bytes_per_shard: Sequence of shard sizes in bytes (typically from
            ``RequiredShardMeta.db_bytes`` for every shard in the manifest).
            Empty input → ``"download"`` (no data to motivate ranged reads).
        per_shard_threshold: Per-shard size at or above which we switch to
            range-read. Defaults to 16 MiB.
        total_budget: Cumulative size at or above which we switch to
            range-read. Defaults to 2 GiB.
    """

    if not db_bytes_per_shard:
        return "download"
    if max(db_bytes_per_shard) >= per_shard_threshold:
        return "range"
    if sum(db_bytes_per_shard) >= total_budget:
        return "range"
    return "download"


@runtime_checkable
class SqliteAccessPolicy(Protocol):
    """Library-only hook for advanced ``auto`` overrides.

    Implementations receive the snapshot's per-shard byte sizes and return
    ``"download"`` or ``"range"``.  Used by reader state builders when the
    user supplies a custom policy instead of the default thresholds.
    """

    def decide(
        self, db_bytes_per_shard: Sequence[int]
    ) -> Literal["download", "range"]: ...


@dataclass(slots=True)
class _ThresholdPolicy:
    """Default :class:`SqliteAccessPolicy` backed by :func:`decide_access_mode`."""

    per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES
    total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES

    def decide(self, db_bytes_per_shard: Sequence[int]) -> Literal["download", "range"]:
        return decide_access_mode(
            db_bytes_per_shard=db_bytes_per_shard,
            per_shard_threshold=self.per_shard_threshold,
            total_budget=self.total_budget,
        )


def make_threshold_policy(
    *,
    per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
    total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
) -> SqliteAccessPolicy:
    """Construct a default threshold-based :class:`SqliteAccessPolicy`."""

    return _ThresholdPolicy(
        per_shard_threshold=per_shard_threshold,
        total_budget=total_budget,
    )


def make_sqlite_reader_factory(
    *,
    mode: Literal["download", "range"],
    mmap_size: int = 268435456,
    page_cache_pages: int = 1024,
    s3_connection_options: S3ConnectionOptions | None = None,
    credential_provider: CredentialProvider | None = None,
) -> SqliteReaderFactory | SqliteRangeReaderFactory:
    """Build a sync SQLite reader factory for a *concrete* access mode.

    ``mode`` must be ``"download"`` or ``"range"``; ``"auto"`` is resolved
    by the reader state builder before it calls this helper.
    """

    if mode == "download":
        return SqliteReaderFactory(
            mmap_size=mmap_size,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    if mode == "range":
        return SqliteRangeReaderFactory(
            page_cache_pages=page_cache_pages,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    raise SqliteAdapterError(
        f"make_sqlite_reader_factory: unsupported mode {mode!r}; "
        "expected 'download' or 'range' (resolve 'auto' before calling)"
    )


def make_async_sqlite_reader_factory(
    *,
    mode: Literal["download", "range"],
    mmap_size: int = 268435456,
    page_cache_pages: int = 1024,
    s3_connection_options: S3ConnectionOptions | None = None,
    credential_provider: CredentialProvider | None = None,
) -> AsyncSqliteReaderFactory | AsyncSqliteRangeReaderFactory:
    """Build an async SQLite reader factory for a *concrete* access mode."""

    if mode == "download":
        return AsyncSqliteReaderFactory(
            mmap_size=mmap_size,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    if mode == "range":
        return AsyncSqliteRangeReaderFactory(
            page_cache_pages=page_cache_pages,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )
    raise SqliteAdapterError(
        f"make_async_sqlite_reader_factory: unsupported mode {mode!r}; "
        "expected 'download' or 'range' (resolve 'auto' before calling)"
    )


# ---------------------------------------------------------------------------
# Adaptive (auto) factories
# ---------------------------------------------------------------------------


class AdaptiveSqliteReaderFactory:
    """Sync SQLite reader factory that auto-selects ``download`` vs ``range``.

    On first invocation for a given snapshot (keyed by
    ``manifest.required_build.run_id``) the factory inspects every shard's
    ``db_bytes`` and asks the configured :class:`SqliteAccessPolicy` for a
    decision.  The resulting concrete factory
    (:class:`SqliteReaderFactory` or :class:`SqliteRangeReaderFactory`) is
    cached and reused for all subsequent shards in the same snapshot.

    The cache is a single slot: when a new ``run_id`` arrives the previous
    factory is replaced.  This matches the reader lifecycle —
    ``ShardedReader.refresh()`` rebuilds all shard readers, so any retired
    sub-factory is dropped naturally.

    Thread-safety
    -------------
    The single-slot cache (``_cached_run_id`` / ``_cached_factory``) is **not**
    internally synchronised.  Concurrent calls into ``_resolve_factory`` from
    multiple threads against different snapshots could race and leave the
    cache in a momentarily inconsistent state (no corruption, but a redundant
    factory build).

    In practice this is safe because every supported caller serialises
    refresh-time state construction:

    * :class:`ConcurrentShardedReader` holds ``_refresh_lock`` while building
      a new ``_ReaderState`` (which is when ``__call__`` invocations happen).
    * :class:`AsyncShardedReader` serialises ``_build_state`` via an
      ``asyncio.Lock``.

    Callers wiring ``AdaptiveSqliteReaderFactory`` into custom reader plumbing
    must preserve this single-writer-during-refresh invariant.
    """

    __slots__ = (
        "_policy",
        "_mmap_size",
        "_page_cache_pages",
        "_s3_connection_options",
        "_credential_provider",
        "_cached_run_id",
        "_cached_factory",
    )

    def __init__(
        self,
        *,
        policy: SqliteAccessPolicy | None = None,
        per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
        total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
        mmap_size: int = 268435456,
        page_cache_pages: int = 1024,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._policy: SqliteAccessPolicy = policy or _ThresholdPolicy(
            per_shard_threshold=per_shard_threshold,
            total_budget=total_budget,
        )
        self._mmap_size = mmap_size
        self._page_cache_pages = page_cache_pages
        self._s3_connection_options = s3_connection_options
        self._credential_provider = credential_provider
        self._cached_run_id: str | None = None
        self._cached_factory: SqliteReaderFactory | SqliteRangeReaderFactory | None = (
            None
        )

    def _resolve_factory(
        self, manifest: Manifest
    ) -> SqliteReaderFactory | SqliteRangeReaderFactory:
        run_id = manifest.required_build.run_id
        if run_id == self._cached_run_id and self._cached_factory is not None:
            return self._cached_factory
        sizes = [shard.db_bytes for shard in manifest.shards]
        mode = self._policy.decide(sizes)
        factory = make_sqlite_reader_factory(
            mode=mode,
            mmap_size=self._mmap_size,
            page_cache_pages=self._page_cache_pages,
            s3_connection_options=self._s3_connection_options,
            credential_provider=self._credential_provider,
        )
        self._cached_run_id = run_id
        self._cached_factory = factory
        return factory

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> SqliteShardReader | SqliteRangeShardReader:
        factory = self._resolve_factory(manifest)
        return factory(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
        )


class AsyncAdaptiveSqliteReaderFactory:
    """Async counterpart of :class:`AdaptiveSqliteReaderFactory`.

    Shares the same single-slot cache contract: callers must serialise
    snapshot-state construction (``AsyncShardedReader`` does so via its
    refresh ``asyncio.Lock``).
    """

    __slots__ = (
        "_policy",
        "_mmap_size",
        "_page_cache_pages",
        "_s3_connection_options",
        "_credential_provider",
        "_cached_run_id",
        "_cached_factory",
    )

    def __init__(
        self,
        *,
        policy: SqliteAccessPolicy | None = None,
        per_shard_threshold: int = DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
        total_budget: int = DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
        mmap_size: int = 268435456,
        page_cache_pages: int = 1024,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._policy: SqliteAccessPolicy = policy or _ThresholdPolicy(
            per_shard_threshold=per_shard_threshold,
            total_budget=total_budget,
        )
        self._mmap_size = mmap_size
        self._page_cache_pages = page_cache_pages
        self._s3_connection_options = s3_connection_options
        self._credential_provider = credential_provider
        self._cached_run_id: str | None = None
        self._cached_factory: (
            AsyncSqliteReaderFactory | AsyncSqliteRangeReaderFactory | None
        ) = None

    def _resolve_factory(
        self, manifest: Manifest
    ) -> AsyncSqliteReaderFactory | AsyncSqliteRangeReaderFactory:
        run_id = manifest.required_build.run_id
        if run_id == self._cached_run_id and self._cached_factory is not None:
            return self._cached_factory
        sizes = [shard.db_bytes for shard in manifest.shards]
        mode = self._policy.decide(sizes)
        factory = make_async_sqlite_reader_factory(
            mode=mode,
            mmap_size=self._mmap_size,
            page_cache_pages=self._page_cache_pages,
            s3_connection_options=self._s3_connection_options,
            credential_provider=self._credential_provider,
        )
        self._cached_run_id = run_id
        self._cached_factory = factory
        return factory

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncSqliteShardReader | AsyncSqliteRangeShardReader:
        factory = self._resolve_factory(manifest)
        return await factory(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
        )
