"""Shared S3 range-read VFS layer for SQLite databases on S3.

Provides :class:`S3ReadOnlyFile` (offset-based S3 reads with a
page-aligned LRU cache) and :func:`create_apsw_vfs` (registers an APSW
VFS backed by an ``S3ReadOnlyFile``).

I/O is implemented on top of `obstore <https://developmentseed.org/obstore/>`_
(Rust ``object_store`` Python bindings).  Compared to the previous
``boto3`` implementation this avoids per-page Python overhead, uses
zero-copy ``Bytes`` buffers, and lets ``obstore.get_ranges`` coalesce
adjacent page misses into a single underlying request with parallel
fetches.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from .credentials import S3Credentials
from .errors import ShardyfusionError
from .storage import create_s3_store
from .type_defs import S3ConnectionOptions

# Fallback page size used for LRU cache keying when the file header
# is unreachable.  Matches the SQLite default; real shards advertise
# their page size in the file header (big-endian u16 at bytes 16-17,
# with a special-case value of 1 meaning 65536) which the VFS parses on
# open and exposes as ``S3ReadOnlyFile.page_size``.
_DEFAULT_PAGE_SIZE = 4096
# Bytes 16-17 of the SQLite file header encode the page size.  Reading
# the first 100 bytes is enough to recover the page size and validate
# the magic; this is one short S3 GET amortised over every subsequent
# page-aligned read.
_SQLITE_HEADER_SIZE = 100
_SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"


def _parse_page_size_from_header(header: bytes) -> int | None:
    """Return the page size encoded in a SQLite file header, or ``None``.

    The header is the first 100 bytes of every SQLite database file.
    Bytes 0-15 hold the magic ``"SQLite format 3\\x00"``; bytes 16-17
    encode page size as big-endian ``u16`` with the special value ``1``
    meaning 65536 (the upper bound that doesn't fit in a u16).  Any
    value outside ``[512, 65536]`` or not a power of two is rejected.
    """
    if len(header) < 18 or header[:16] != _SQLITE_HEADER_MAGIC:
        return None
    raw = int.from_bytes(header[16:18], "big")
    page_size = 65536 if raw == 1 else raw
    if page_size < 512 or page_size > 65536:
        return None
    if page_size & (page_size - 1):
        return None
    return page_size


class S3VfsError(ShardyfusionError):
    """S3 VFS configuration error (non-retryable)."""

    retryable = False


def _normalize_page_cache_pages(page_cache_pages: int) -> int:
    pages = int(page_cache_pages)
    if pages < 0:
        raise S3VfsError("page_cache_pages must be >= 0")
    return pages


def _build_obstore_client(
    *,
    bucket: str,
    s3_connection_options: S3ConnectionOptions | None,
    s3_credentials: S3Credentials | None,
) -> Any:
    """Construct an :class:`obstore.store.S3Store` from shardyfusion config.

    Thin wrapper around :func:`shardyfusion.storage.create_s3_store` that
    passes the explicit bucket required by the VFS layer.
    """
    return create_s3_store(
        bucket=bucket,
        credentials=s3_credentials,
        connection_options=s3_connection_options,
    )


class S3ReadOnlyFile:
    """Read-only virtual file backed by S3 range requests with page LRU cache.

    Reads are decomposed into a contiguous range of fixed-size pages
    (``_PAGE_SIZE`` bytes each).  Cached pages are returned directly;
    missing pages are fetched in a single :func:`obstore.get_ranges`
    call that coalesces adjacent ranges and parallelises non-adjacent
    ones inside Rust.
    """

    def __init__(
        self,
        *,
        bucket: str,
        key: str,
        page_cache_pages: int = 1024,
        s3_connection_options: S3ConnectionOptions | None = None,
        s3_credentials: S3Credentials | None = None,
    ) -> None:
        try:
            import obstore  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise S3VfsError(
                "obstore is required for the SQLite range-read VFS. "
                "Install via: pip install 'shardyfusion[sqlite-range]' "
                "or 'shardyfusion[sqlite-adaptive]' for the adaptive reader."
            ) from exc

        page_cache_pages = _normalize_page_cache_pages(page_cache_pages)

        self._store = _build_obstore_client(
            bucket=bucket,
            s3_connection_options=s3_connection_options,
            s3_credentials=s3_credentials,
        )
        self._key = key
        self._page_cache: OrderedDict[int, bytes] = OrderedDict()
        self._page_cache_pages = page_cache_pages
        self._lock = threading.Lock()

        meta = obstore.head(self._store, key)
        # ``meta`` is an ObjectMeta-like mapping; both attribute and
        # mapping access have appeared across obstore versions.
        size = getattr(meta, "size", None)
        if size is None:
            try:
                size = meta["size"]  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive
                raise S3VfsError(
                    "obstore.head response did not expose a 'size' field"
                ) from None
        self._size: int = int(size)

        self._page_size = self._discover_page_size(obstore)

    @property
    def page_size(self) -> int:
        return self._page_size

    def _discover_page_size(self, obstore: Any) -> int:
        """Read the first 100 bytes of the file and parse the header page size.

        Falls back to :data:`_DEFAULT_PAGE_SIZE` (4096) when the file is
        empty, too small, or the header parse fails.  A 100-byte GET is
        small enough that the cost is negligible compared to the per-shard
        latency benefit of caching the right-sized pages.
        """
        end = min(_SQLITE_HEADER_SIZE, self._size)
        if end <= 0:
            return _DEFAULT_PAGE_SIZE
        results = obstore.get_ranges(self._store, self._key, starts=[0], ends=[end])
        header = bytes(results[0])
        parsed = _parse_page_size_from_header(header)
        return parsed if parsed is not None else _DEFAULT_PAGE_SIZE

    @property
    def size(self) -> int:
        return self._size

    def _fetch_pages(self, page_indices: list[int]) -> dict[int, bytes]:
        """Fetch the given pages from S3 in one batched obstore call."""
        import obstore  # pyright: ignore[reportMissingImports]

        page_size = self._page_size
        starts: list[int] = []
        ends: list[int] = []
        for idx in page_indices:
            start = idx * page_size
            end = min(start + page_size, self._size)
            starts.append(start)
            ends.append(end)

        # ``get_ranges`` coalesces ranges within ``coalesce`` bytes of
        # each other into a single underlying request, and parallelises
        # disjoint ranges.  The default coalesce is 1 MiB which is more
        # than enough for adjacent page misses.
        results = obstore.get_ranges(self._store, self._key, starts=starts, ends=ends)

        out: dict[int, bytes] = {}
        for idx, buf in zip(page_indices, results, strict=True):
            # ``buf`` is an obstore.Bytes (buffer protocol).  Materialise
            # to a plain ``bytes`` for stable cache storage; the buffer
            # backing the Rust object is freed when the Bytes is dropped.
            out[idx] = bytes(buf)
        return out

    def read(self, offset: int, amount: int) -> bytes:
        if amount <= 0:
            return b""
        if offset >= self._size:
            return b""

        # Clamp request to file end.
        end_offset = min(offset + amount, self._size)
        amount = end_offset - offset

        page_size = self._page_size
        first_page = offset // page_size
        last_page = (end_offset - 1) // page_size  # inclusive

        page_indices = list(range(first_page, last_page + 1))
        cached: dict[int, bytes] = {}
        missing: list[int] = []

        if self._page_cache_pages > 0:
            with self._lock:
                for idx in page_indices:
                    page = self._page_cache.get(idx)
                    if page is not None:
                        self._page_cache.move_to_end(idx)
                        cached[idx] = page
                    else:
                        missing.append(idx)
        else:
            missing = list(page_indices)

        fetched: dict[int, bytes] = {}
        if missing:
            fetched = self._fetch_pages(missing)

            if self._page_cache_pages > 0:
                with self._lock:
                    for idx, data in fetched.items():
                        # Skip if another thread populated meanwhile.
                        if idx in self._page_cache:
                            self._page_cache.move_to_end(idx)
                            continue
                        while len(self._page_cache) >= self._page_cache_pages:
                            self._page_cache.popitem(last=False)
                        self._page_cache[idx] = data

        # Assemble result.
        if len(page_indices) == 1:
            idx = page_indices[0]
            page = cached.get(idx) or fetched.get(idx, b"")
            page_start = idx * page_size
            local_off = offset - page_start
            return page[local_off : local_off + amount]

        chunks: list[bytes] = []
        for i, idx in enumerate(page_indices):
            page = cached.get(idx) or fetched.get(idx, b"")
            page_start = idx * page_size
            if i == 0:
                local_off = offset - page_start
                chunks.append(page[local_off:])
            elif i == len(page_indices) - 1:
                # Trim tail to honour the requested amount.
                want = end_offset - page_start
                chunks.append(page[:want])
            else:
                chunks.append(page)
        return b"".join(chunks)


def create_apsw_vfs(vfs_name: str, s3_file: S3ReadOnlyFile) -> Any:
    """Create and register an APSW VFS backed by S3 range reads.

    Defines VFS and VFSFile subclasses at call time because ``apsw`` is
    an optional dependency that may not be available at module load.
    Returns the registered VFS instance (caller must keep it alive while
    the connection is open, and call ``unregister()`` on close).
    """
    import apsw  # pyright: ignore[reportMissingImports]

    class S3RangeVFS(apsw.VFS):
        def __init__(self) -> None:
            # Empty string = no parent VFS; all I/O handled by our methods.
            super().__init__(vfs_name, "")

        def xOpen(
            self, name: str | apsw.URIFilename | None, flags: list[int]
        ) -> S3RangeVFSFile:
            return S3RangeVFSFile("", flags)

        def xAccess(self, pathname: str, flags: int) -> bool:
            # No journal/WAL/SHM files exist on S3.
            return False

        def xFullPathname(self, name: str) -> str:
            return name

    class S3RangeVFSFile(apsw.VFSFile):
        def __init__(self, _name: str, _flags: list[int]) -> None:
            # Do NOT call super().__init__() — there is no underlying
            # base file to open; all I/O goes through S3 range reads.
            pass

        def xRead(self, amount: int, offset: int) -> bytes:
            data = s3_file.read(offset, amount)
            # SQLite expects exactly ``amount`` bytes; pad with zeros
            # if the read is short (e.g. at end of file).
            if len(data) < amount:
                data += b"\x00" * (amount - len(data))
            return data

        def xFileSize(self) -> int:
            return s3_file.size

        def xClose(self) -> None:
            pass

        def xLock(self, level: int) -> None:
            pass

        def xUnlock(self, level: int) -> None:
            pass

        def xCheckReservedLock(self) -> bool:
            return False

        def xFileControl(self, op: int, ptr: int) -> bool:
            return False

        def xSectorSize(self) -> int:
            return s3_file.page_size

        def xDeviceCharacteristics(self) -> int:
            return 0

    return S3RangeVFS()
