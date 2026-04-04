"""Shared S3 range-read VFS layer for SQLite databases on S3.

Provides :class:`S3ReadOnlyFile` (offset-based S3 reads with LRU page
cache) and :func:`create_apsw_vfs` (registers an APSW VFS backed by an
``S3ReadOnlyFile``).  Used by both the KV ``sqlite_adapter`` and the
vector search reader.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from .credentials import S3Credentials
from .errors import ShardyfusionError
from .type_defs import S3ConnectionOptions


class S3VfsError(ShardyfusionError):
    """S3 VFS configuration error (non-retryable)."""

    retryable = False


def _normalize_page_cache_pages(page_cache_pages: int) -> int:
    pages = int(page_cache_pages)
    if pages < 0:
        raise S3VfsError("page_cache_pages must be >= 0")
    return pages


class S3ReadOnlyFile:
    """Read-only virtual file backed by S3 range requests with LRU cache.

    This is the core I/O layer used by the APSW VFS implementation.
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
        from .storage import create_s3_client

        page_cache_pages = _normalize_page_cache_pages(page_cache_pages)

        self._client = create_s3_client(s3_credentials, s3_connection_options)
        self._bucket = bucket
        self._key = key
        self._page_cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self._page_cache_pages = page_cache_pages
        self._lock = threading.Lock()

        head = self._client.head_object(Bucket=bucket, Key=key)
        self._size: int = head["ContentLength"]

    @property
    def size(self) -> int:
        return self._size

    def read(self, offset: int, amount: int) -> bytes:
        if offset >= self._size:
            return b""

        cache_key = (offset, amount)
        if self._page_cache_pages > 0:
            with self._lock:
                cached = self._page_cache.get(cache_key)
                if cached is not None:
                    self._page_cache.move_to_end(cache_key)
                    return cached

        end = min(offset + amount - 1, self._size - 1)
        resp = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=f"bytes={offset}-{end}",
        )
        data = resp["Body"].read()

        if self._page_cache_pages > 0:
            with self._lock:
                # LRU eviction
                if len(self._page_cache) >= self._page_cache_pages:
                    self._page_cache.popitem(last=False)
                self._page_cache[cache_key] = data

        return data


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
            return 4096

        def xDeviceCharacteristics(self) -> int:
            return 0

    return S3RangeVFS()
