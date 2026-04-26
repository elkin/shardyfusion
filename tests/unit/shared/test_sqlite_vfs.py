"""Tests for _sqlite_vfs.S3ReadOnlyFile with mocked obstore."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from shardyfusion._sqlite_vfs import (
    S3ReadOnlyFile,
    S3VfsError,
    _normalize_page_cache_pages,
)

# ---------------------------------------------------------------------------
# Helpers — fake obstore module
# ---------------------------------------------------------------------------


_PAGE = 4096


class _FakeBytes:
    """Stand-in for obstore.Bytes; supports the buffer protocol via bytes()."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __bytes__(self) -> bytes:
        return self._data

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._data)


class _FakeObstoreModule:
    """In-memory fake of the ``obstore`` top-level module.

    Backs every ``S3Store`` instance with a dict of ``key -> bytes`` so
    tests can drive ``head``/``get_ranges`` realistically while still
    asserting on call counts.
    """

    def __init__(self) -> None:
        self._objects: dict[tuple[int, str], bytes] = {}
        self.head_calls: list[tuple[Any, str]] = []
        self.get_ranges_calls: list[tuple[Any, str, list[int], list[int]]] = []

    # store-construction helper
    def make_store_module(self) -> types.ModuleType:
        outer = self

        class _S3Store:
            def __init__(self, bucket: str, **kwargs: Any) -> None:
                self.bucket = bucket
                self.kwargs = kwargs

            def __repr__(self) -> str:  # pragma: no cover - debug
                return f"_S3Store(bucket={self.bucket!r})"

        store_mod = types.ModuleType("obstore.store")
        store_mod.S3Store = _S3Store  # type: ignore[attr-defined]
        outer._S3Store = _S3Store  # type: ignore[attr-defined]
        return store_mod

    # public API used by _sqlite_vfs
    def head(self, store: Any, path: str) -> dict[str, Any]:
        self.head_calls.append((store, path))
        data = self._objects.get((id(store), path), b"")
        return {"size": len(data), "path": path}

    def get_ranges(
        self,
        store: Any,
        path: str,
        *,
        starts: list[int],
        ends: list[int],
        coalesce: int = 1024 * 1024,
    ) -> list[_FakeBytes]:
        self.get_ranges_calls.append((store, path, list(starts), list(ends)))
        data = self._objects.get((id(store), path), b"")
        out: list[_FakeBytes] = []
        for s, e in zip(starts, ends, strict=True):
            out.append(_FakeBytes(data[s:e]))
        return out

    # test helper
    def put(self, store: Any, path: str, data: bytes) -> None:
        self._objects[(id(store), path)] = data


@pytest.fixture()
def fake_obstore() -> Iterator[_FakeObstoreModule]:
    """Install a fake ``obstore`` package in ``sys.modules`` for the test."""
    fake = _FakeObstoreModule()
    obstore_mod = types.ModuleType("obstore")
    # Dispatch through lambdas so reassigning ``fake.head`` (e.g. in
    # ``_build``) takes effect for in-flight callers.
    obstore_mod.head = lambda *a, **kw: fake.head(*a, **kw)  # type: ignore[attr-defined]
    obstore_mod.get_ranges = lambda *a, **kw: fake.get_ranges(*a, **kw)  # type: ignore[attr-defined]
    obstore_mod.store = fake.make_store_module()  # type: ignore[attr-defined]

    saved = {name: sys.modules.get(name) for name in ("obstore", "obstore.store")}
    sys.modules["obstore"] = obstore_mod
    sys.modules["obstore.store"] = obstore_mod.store  # type: ignore[attr-defined]
    try:
        yield fake
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _build(
    fake: _FakeObstoreModule,
    *,
    data: bytes,
    page_cache_pages: int = 1024,
    bucket: str = "b",
    key: str = "k",
) -> S3ReadOnlyFile:
    """Construct an S3ReadOnlyFile with ``data`` as the backing object.

    We patch ``head`` to return the size first, then run the constructor,
    then register the data on the actual store instance for ``get_ranges``.
    """
    # Patch head to return correct size for any store/key during init.
    real_head = fake.head

    def head_with_size(store: Any, path: str) -> dict[str, Any]:
        fake.head_calls.append((store, path))
        return {"size": len(data), "path": path}

    fake.head = head_with_size  # type: ignore[assignment]
    try:
        f = S3ReadOnlyFile(bucket=bucket, key=key, page_cache_pages=page_cache_pages)
    finally:
        fake.head = real_head  # type: ignore[assignment]

    # Now register the data against the actual store object so subsequent
    # get_ranges calls return the correct slices.
    fake.put(f._store, key, data)
    return f


# ---------------------------------------------------------------------------
# _normalize_page_cache_pages
# ---------------------------------------------------------------------------


class TestNormalizePageCachePages:
    def test_valid_value(self) -> None:
        assert _normalize_page_cache_pages(10) == 10

    def test_zero_allowed(self) -> None:
        assert _normalize_page_cache_pages(0) == 0

    def test_negative_raises(self) -> None:
        with pytest.raises(S3VfsError, match="page_cache_pages must be >= 0"):
            _normalize_page_cache_pages(-1)


# ---------------------------------------------------------------------------
# S3ReadOnlyFile
# ---------------------------------------------------------------------------


class TestS3ReadOnlyFile:
    def test_init_and_size(self, fake_obstore: _FakeObstoreModule) -> None:
        f = _build(fake_obstore, data=b"x" * 500)
        assert f.size == 500
        assert len(fake_obstore.head_calls) == 1

    def test_read_basic(self, fake_obstore: _FakeObstoreModule) -> None:
        f = _build(fake_obstore, data=b"abcdefghij" * 10)  # 100 bytes
        data = f.read(0, 10)
        assert data == b"abcdefghij"

    def test_read_past_eof_returns_empty(
        self, fake_obstore: _FakeObstoreModule
    ) -> None:
        f = _build(fake_obstore, data=b"x" * 100)
        assert f.read(200, 10) == b""
        assert fake_obstore.get_ranges_calls == []

    def test_zero_length_read_returns_empty(
        self, fake_obstore: _FakeObstoreModule
    ) -> None:
        f = _build(fake_obstore, data=b"x" * 100)
        assert f.read(0, 0) == b""
        assert fake_obstore.get_ranges_calls == []

    def test_negative_length_read_returns_empty(
        self, fake_obstore: _FakeObstoreModule
    ) -> None:
        f = _build(fake_obstore, data=b"x" * 100)
        assert f.read(0, -1) == b""
        assert fake_obstore.get_ranges_calls == []

    def test_read_clamps_to_file_end(self, fake_obstore: _FakeObstoreModule) -> None:
        f = _build(fake_obstore, data=b"x" * 100)
        # Request past EOF: should return remaining bytes only.
        data = f.read(95, 50)
        assert data == b"x" * 5

    def test_read_cache_hit(self, fake_obstore: _FakeObstoreModule) -> None:
        f = _build(fake_obstore, data=b"x" * 100, page_cache_pages=10)

        # First read — cache miss, one get_ranges call
        f.read(0, 10)
        assert len(fake_obstore.get_ranges_calls) == 1

        # Second read (same page) — cache hit, no new fetch
        f.read(0, 10)
        assert len(fake_obstore.get_ranges_calls) == 1

    def test_read_cache_disabled(self, fake_obstore: _FakeObstoreModule) -> None:
        f = _build(fake_obstore, data=b"x" * 100, page_cache_pages=0)

        f.read(0, 10)
        f.read(0, 10)
        assert len(fake_obstore.get_ranges_calls) == 2

    def test_lru_eviction(self, fake_obstore: _FakeObstoreModule) -> None:
        # Need data spanning at least 3 pages to exercise eviction with
        # page-aligned cache.
        size = _PAGE * 3
        f = _build(fake_obstore, data=b"a" * size, page_cache_pages=1)

        # Fill the cache with page 0
        f.read(0, 10)
        assert len(fake_obstore.get_ranges_calls) == 1

        # Read from page 1 — evicts page 0
        f.read(_PAGE, 10)
        assert len(fake_obstore.get_ranges_calls) == 2

        # Page 0 was evicted — re-reading it requires a new fetch
        f.read(0, 10)
        assert len(fake_obstore.get_ranges_calls) == 3

    def test_multi_page_read_uses_single_get_ranges_call(
        self, fake_obstore: _FakeObstoreModule
    ) -> None:
        # Distinct content per page so we can verify assembly.
        data = b"".join(bytes([i % 256]) * _PAGE for i in range(3))
        f = _build(fake_obstore, data=data, page_cache_pages=10)

        # Read straddling pages 0 and 1.
        out = f.read(_PAGE - 5, 10)
        assert out == data[_PAGE - 5 : _PAGE + 5]
        # Single underlying fetch covering both pages.
        assert len(fake_obstore.get_ranges_calls) == 1
        _, _, starts, ends = fake_obstore.get_ranges_calls[0]
        assert starts == [0, _PAGE]
        assert ends == [_PAGE, _PAGE * 2]

    def test_partial_cache_hit_only_fetches_missing(
        self, fake_obstore: _FakeObstoreModule
    ) -> None:
        data = b"".join(bytes([i % 256]) * _PAGE for i in range(3))
        f = _build(fake_obstore, data=data, page_cache_pages=10)

        # Prime page 0
        f.read(0, 10)
        assert len(fake_obstore.get_ranges_calls) == 1

        # Read straddling pages 0 (cached) and 1 (missing).
        f.read(_PAGE - 5, 10)
        # Only page 1 should be fetched.
        assert len(fake_obstore.get_ranges_calls) == 2
        _, _, starts, ends = fake_obstore.get_ranges_calls[1]
        assert starts == [_PAGE]
        assert ends == [_PAGE * 2]


# ---------------------------------------------------------------------------
# Page-aligned cache assembly — hypothesis property test
# ---------------------------------------------------------------------------


from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


class TestReadCorrectness:
    @given(
        size=st.integers(min_value=1, max_value=20_000),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        cache_pages=st.integers(min_value=0, max_value=8),
        ops=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=25_000),
                st.integers(min_value=0, max_value=10_000),
            ),
            min_size=1,
            max_size=20,
        ),
    )
    def test_random_reads_match_ground_truth(
        self,
        size: int,
        seed: int,
        cache_pages: int,
        ops: list[tuple[int, int]],
    ) -> None:
        import random

        rng = random.Random(seed)
        data = bytes(rng.randrange(256) for _ in range(size))

        # Hypothesis runs many examples; build a fresh fake obstore each time.
        fake = _FakeObstoreModule()
        obstore_mod = types.ModuleType("obstore")
        obstore_mod.head = lambda *a, **kw: fake.head(*a, **kw)  # type: ignore[attr-defined]
        obstore_mod.get_ranges = lambda *a, **kw: fake.get_ranges(*a, **kw)  # type: ignore[attr-defined]
        obstore_mod.store = fake.make_store_module()  # type: ignore[attr-defined]

        saved = {name: sys.modules.get(name) for name in ("obstore", "obstore.store")}
        sys.modules["obstore"] = obstore_mod
        sys.modules["obstore.store"] = obstore_mod.store  # type: ignore[attr-defined]
        try:
            f = _build(fake, data=data, page_cache_pages=cache_pages)
            for offset, amount in ops:
                got = f.read(offset, amount)
                # Ground truth: same clamping rules as S3ReadOnlyFile.
                if amount <= 0 or offset >= size:
                    expected = b""
                else:
                    expected = data[offset : offset + amount]
                assert got == expected, (
                    f"Mismatch at offset={offset} amount={amount}: "
                    f"got {got!r}, expected {expected!r}"
                )
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod


# ---------------------------------------------------------------------------
# S3VfsError
# ---------------------------------------------------------------------------


class TestS3VfsError:
    def test_not_retryable(self) -> None:
        err = S3VfsError("test")
        assert err.retryable is False


# ---------------------------------------------------------------------------
# create_apsw_vfs — mocked apsw
# ---------------------------------------------------------------------------


class TestCreateApswVfs:
    """Test create_apsw_vfs by mocking the apsw module."""

    def _make_mock_apsw(self) -> MagicMock:
        """Create a mock apsw module with VFS and VFSFile base classes."""
        mock_apsw = MagicMock()

        class _MockVFS:
            def __init__(self, name: str, base: str) -> None:
                self.name = name
                self.base = base

        class _MockVFSFile:
            pass

        mock_apsw.VFS = _MockVFS
        mock_apsw.VFSFile = _MockVFSFile
        mock_apsw.URIFilename = str
        return mock_apsw

    def test_creates_vfs_instance(self) -> None:
        mock_apsw = self._make_mock_apsw()
        original = sys.modules.get("apsw")
        sys.modules["apsw"] = mock_apsw
        try:
            from shardyfusion._sqlite_vfs import create_apsw_vfs

            mock_s3_file = MagicMock()
            mock_s3_file.size = 1000
            mock_s3_file.read.return_value = b"x" * 100

            vfs = create_apsw_vfs("test-vfs", mock_s3_file)
            assert vfs is not None
            assert vfs.name == "test-vfs"
        finally:
            if original is None:
                sys.modules.pop("apsw", None)
            else:
                sys.modules["apsw"] = original

    def test_vfs_xopen_returns_file(self) -> None:
        mock_apsw = self._make_mock_apsw()
        original = sys.modules.get("apsw")
        sys.modules["apsw"] = mock_apsw
        try:
            from shardyfusion._sqlite_vfs import create_apsw_vfs

            mock_s3_file = MagicMock()
            mock_s3_file.size = 500
            mock_s3_file.read.return_value = b"d" * 100

            vfs = create_apsw_vfs("test-vfs-open", mock_s3_file)
            vfs_file = vfs.xOpen("test.db", [0])
            assert vfs_file is not None
        finally:
            if original is None:
                sys.modules.pop("apsw", None)
            else:
                sys.modules["apsw"] = original

    def test_vfs_file_operations(self) -> None:
        mock_apsw = self._make_mock_apsw()
        original = sys.modules.get("apsw")
        sys.modules["apsw"] = mock_apsw
        try:
            from shardyfusion._sqlite_vfs import create_apsw_vfs

            mock_s3_file = MagicMock()
            mock_s3_file.size = 500
            mock_s3_file.read.return_value = b"x" * 100

            vfs = create_apsw_vfs("test-vfs-ops", mock_s3_file)
            vfs_file = vfs.xOpen("test.db", [0])

            # Test xFileSize
            assert vfs_file.xFileSize() == 500

            # Test xRead
            data = vfs_file.xRead(100, 0)
            assert len(data) == 100
            mock_s3_file.read.assert_called()

            # Test xRead with short read (padding)
            mock_s3_file.read.return_value = b"x" * 50
            data = vfs_file.xRead(100, 0)
            assert len(data) == 100
            assert data[50:] == b"\x00" * 50

            # Test xClose (no-op)
            vfs_file.xClose()

            # Test xLock / xUnlock (no-op)
            vfs_file.xLock(1)
            vfs_file.xUnlock(1)

            # Test xCheckReservedLock
            assert vfs_file.xCheckReservedLock() is False

            # Test xFileControl
            assert vfs_file.xFileControl(0, 0) is False

            # Test xSectorSize
            assert vfs_file.xSectorSize() == 4096

            # Test xDeviceCharacteristics
            assert vfs_file.xDeviceCharacteristics() == 0
        finally:
            if original is None:
                sys.modules.pop("apsw", None)
            else:
                sys.modules["apsw"] = original

    def test_vfs_xaccess_returns_false(self) -> None:
        mock_apsw = self._make_mock_apsw()
        original = sys.modules.get("apsw")
        sys.modules["apsw"] = mock_apsw
        try:
            from shardyfusion._sqlite_vfs import create_apsw_vfs

            mock_s3_file = MagicMock()
            vfs = create_apsw_vfs("test-vfs-access", mock_s3_file)
            assert vfs.xAccess("/path", 0) is False
        finally:
            if original is None:
                sys.modules.pop("apsw", None)
            else:
                sys.modules["apsw"] = original

    def test_vfs_xfullpathname(self) -> None:
        mock_apsw = self._make_mock_apsw()
        original = sys.modules.get("apsw")
        sys.modules["apsw"] = mock_apsw
        try:
            from shardyfusion._sqlite_vfs import create_apsw_vfs

            mock_s3_file = MagicMock()
            vfs = create_apsw_vfs("test-vfs-path", mock_s3_file)
            assert vfs.xFullPathname("test.db") == "test.db"
        finally:
            if original is None:
                sys.modules.pop("apsw", None)
            else:
                sys.modules["apsw"] = original
