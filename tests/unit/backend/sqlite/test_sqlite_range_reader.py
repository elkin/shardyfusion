"""Unit tests for the SQLite range-read backend.

These tests exercise the full APSW VFS pipeline against an in-memory
"S3" object exposed via a fake ``obstore`` module.  Real APSW + real
SQLite handle the page reads; only the storage layer is mocked.
"""

from __future__ import annotations

import sqlite3
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

apsw = pytest.importorskip("apsw")

from shardyfusion._sqlite_vfs import S3ReadOnlyFile  # noqa: E402
from shardyfusion.sqlite_adapter import SqliteRangeShardReader  # noqa: E402

_DB_FILENAME = "shard.db"


# ---------------------------------------------------------------------------
# Fake obstore — backed by an in-memory dict, keyed by id(store)
# ---------------------------------------------------------------------------


class _FakeBytes:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __bytes__(self) -> bytes:
        return self._data

    def __len__(self) -> int:
        return len(self._data)


class _FakeObstore:
    def __init__(self) -> None:
        self._objects: dict[tuple[int, str], bytes] = {}
        self.head_calls = 0
        self.get_ranges_calls = 0

    def make_store_module(self) -> types.ModuleType:
        class _S3Store:
            def __init__(self, bucket: str, **kwargs: Any) -> None:
                self.bucket = bucket
                self.kwargs = kwargs

        store_mod = types.ModuleType("obstore.store")
        store_mod.S3Store = _S3Store  # type: ignore[attr-defined]
        return store_mod

    def head(self, store: Any, path: str) -> dict[str, Any]:
        self.head_calls += 1
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
        self.get_ranges_calls += 1
        data = self._objects.get((id(store), path), b"")
        return [_FakeBytes(data[s:e]) for s, e in zip(starts, ends, strict=True)]

    def put(self, store: Any, path: str, data: bytes) -> None:
        self._objects[(id(store), path)] = data


@pytest.fixture()
def fake_obstore() -> Iterator[_FakeObstore]:
    fake = _FakeObstore()
    obstore_mod = types.ModuleType("obstore")
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sqlite_db(
    tmp_path: Path,
    rows: list[tuple[bytes, bytes]],
    *,
    page_size: int = 4096,
) -> bytes:
    """Build a SQLite DB containing the standard ``kv`` table."""
    db_path = tmp_path / "source.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(f"PRAGMA page_size = {page_size}")
    conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
    conn.executemany("INSERT INTO kv (k, v) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return db_path.read_bytes()


def _open_reader(
    fake: _FakeObstore,
    *,
    db_bytes: bytes,
    local_dir: Path,
    db_url: str = "s3://bucket/shard",
) -> SqliteRangeShardReader:
    """Construct a SqliteRangeShardReader whose backing object is ``db_bytes``.

    The constructor needs ``head()`` to report the right size and
    ``get_ranges()`` to serve the header bytes (so VFS page-size
    discovery picks up the real value), but the fake's dict is keyed
    on ``id(store)`` which we don't know until the store is built
    inside the constructor.  Patch both methods to serve from a
    closure over ``db_bytes`` during init, then register the real data
    against the actual store instance afterwards.
    """
    real_cls = S3ReadOnlyFile

    def make(*args: Any, **kwargs: Any) -> S3ReadOnlyFile:
        original_head = fake.head
        original_get_ranges = fake.get_ranges

        def head_with_size(store: Any, path: str) -> dict[str, Any]:
            fake.head_calls += 1
            return {"size": len(db_bytes), "path": path}

        def get_ranges_with_data(
            store: Any,
            path: str,
            *,
            starts: list[int],
            ends: list[int],
            coalesce: int = 1024 * 1024,
        ) -> list[_FakeBytes]:
            fake.get_ranges_calls += 1
            return [
                _FakeBytes(db_bytes[s:e]) for s, e in zip(starts, ends, strict=True)
            ]

        fake.head = head_with_size  # type: ignore[assignment]
        fake.get_ranges = get_ranges_with_data  # type: ignore[assignment]
        try:
            inst = real_cls(*args, **kwargs)
        finally:
            fake.head = original_head  # type: ignore[assignment]
            fake.get_ranges = original_get_ranges  # type: ignore[assignment]
        # Register the real data against the now-known store instance.
        fake.put(inst._store, inst._key, db_bytes)
        return inst

    with patch("shardyfusion.sqlite_adapter.S3ReadOnlyFile", side_effect=make):
        reader = SqliteRangeShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=None,
        )
    return reader


# ---------------------------------------------------------------------------
# End-to-end tests through APSW VFS
# ---------------------------------------------------------------------------


class TestSqliteRangeShardReaderVFS:
    @pytest.fixture()
    def db_bytes(self, tmp_path: Path) -> bytes:
        return _build_sqlite_db(
            tmp_path, [(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3")]
        )

    def test_get_through_vfs(
        self, tmp_path: Path, db_bytes: bytes, fake_obstore: _FakeObstore
    ) -> None:
        reader = _open_reader(
            fake_obstore, db_bytes=db_bytes, local_dir=tmp_path / "read"
        )
        try:
            assert reader.get(b"key1") == b"val1"
            assert reader.get(b"key2") == b"val2"
            assert reader.get(b"key3") == b"val3"
            assert reader.get(b"missing") is None
        finally:
            reader.close()

    def test_close_unregisters_vfs(
        self, tmp_path: Path, db_bytes: bytes, fake_obstore: _FakeObstore
    ) -> None:
        reader = _open_reader(
            fake_obstore, db_bytes=db_bytes, local_dir=tmp_path / "read"
        )
        vfs_names_before = set(apsw.vfsnames())
        reader.close()
        vfs_names_after = set(apsw.vfsnames())
        # Each reader registers a unique UUID-suffixed VFS; close must
        # unregister it.
        assert len(vfs_names_after) < len(vfs_names_before)

    def test_repeated_lookups_reuse_page_cache(
        self, tmp_path: Path, db_bytes: bytes, fake_obstore: _FakeObstore
    ) -> None:
        """Repeated reads of the same key should not refetch the same pages."""
        reader = _open_reader(
            fake_obstore, db_bytes=db_bytes, local_dir=tmp_path / "read"
        )
        try:
            reader.get(b"key1")
            calls_after_first = fake_obstore.get_ranges_calls
            for _ in range(5):
                assert reader.get(b"key1") == b"val1"
            # The same lookup hit the page cache for B-tree pages; we
            # tolerate at most a small handful of additional fetches
            # (e.g. for any pages SQLite released between calls).
            assert fake_obstore.get_ranges_calls <= calls_after_first, (
                "Cache hit should prevent additional page fetches"
            )
        finally:
            reader.close()

    def test_get_after_close_raises(
        self, tmp_path: Path, db_bytes: bytes, fake_obstore: _FakeObstore
    ) -> None:
        from shardyfusion.sqlite_adapter import SqliteAdapterError

        reader = _open_reader(
            fake_obstore, db_bytes=db_bytes, local_dir=tmp_path / "read"
        )
        reader.close()
        with pytest.raises(SqliteAdapterError, match="closed"):
            reader.get(b"key1")


# ---------------------------------------------------------------------------
# Page-size coverage: writer can produce any supported page_size; the
# range-read VFS must discover it from the file header and round-trip
# every value byte-for-byte.
# ---------------------------------------------------------------------------


class TestRangeReaderHandlesEverySupportedPageSize:
    """End-to-end: build a shard at each supported page_size, read back
    via the range VFS, assert the VFS discovered the right size and the
    values round-trip.  This is the test that catches a regression in
    the hardcoded ``_PAGE_SIZE = 4096`` we removed in this PR."""

    @pytest.mark.parametrize("page_size", [4096, 8192, 16384, 32768, 65536])
    def test_round_trip(
        self,
        tmp_path: Path,
        fake_obstore: _FakeObstore,
        page_size: int,
    ) -> None:
        # Use values that span a full SQLite leaf at the chosen page
        # size, so the read genuinely exercises multi-page navigation
        # (root + interior + leaf), not just a single-page lookup.
        rows = [(i.to_bytes(8, "big"), b"v" * (page_size // 4)) for i in range(64)]
        db_bytes = _build_sqlite_db(tmp_path, rows, page_size=page_size)

        reader = _open_reader(
            fake_obstore, db_bytes=db_bytes, local_dir=tmp_path / "read"
        )
        try:
            # The VFS discovers the page_size from the SQLite header at
            # open time; this is the load-bearing assertion behind the
            # whole feature.
            assert reader._s3_file.page_size == page_size
            # Every row round-trips byte-for-byte.
            for k, v in rows:
                assert reader.get(k) == v
            # Missing keys still return None.
            assert reader.get(b"missing-key-bytes") is None
        finally:
            reader.close()

    def test_default_factory_default_page_size_still_works(
        self, tmp_path: Path, fake_obstore: _FakeObstore
    ) -> None:
        """Regression: a shard produced at the default page_size=4096
        must still be readable through the (now-per-file-aware) VFS."""
        rows = [(b"k1", b"v1"), (b"k2", b"v2"), (b"k3", b"v3")]
        db_bytes = _build_sqlite_db(tmp_path, rows)  # default page_size=4096

        reader = _open_reader(
            fake_obstore, db_bytes=db_bytes, local_dir=tmp_path / "read"
        )
        try:
            assert reader._s3_file.page_size == 4096
            for k, v in rows:
                assert reader.get(k) == v
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# Error path: missing apsw
# ---------------------------------------------------------------------------


class TestSqliteRangeShardReaderImportErrors:
    def test_missing_apsw_raises_with_install_hint(
        self, tmp_path: Path, fake_obstore: _FakeObstore
    ) -> None:
        from shardyfusion.sqlite_adapter import SqliteAdapterError

        # Hide apsw via a sys.modules sentinel that raises on import.
        original = sys.modules.get("apsw")
        sys.modules["apsw"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(SqliteAdapterError) as exc_info:
                SqliteRangeShardReader(
                    db_url="s3://bucket/shard",
                    local_dir=tmp_path / "read",
                    checkpoint_id=None,
                )
            assert "shardyfusion[sqlite-range]" in str(exc_info.value)
            assert "obstore" in str(exc_info.value)
        finally:
            if original is None:
                sys.modules.pop("apsw", None)
            else:
                sys.modules["apsw"] = original


# ---------------------------------------------------------------------------
# S3ReadOnlyFile error propagation through SqliteRangeShardReader
# ---------------------------------------------------------------------------


class TestSqliteRangeShardReaderS3VfsErrorPropagation:
    def test_s3vfs_error_wrapped_as_adapter_error(self, tmp_path: Path) -> None:
        """If S3ReadOnlyFile raises S3VfsError, the reader wraps it."""
        from shardyfusion._sqlite_vfs import S3VfsError
        from shardyfusion.sqlite_adapter import SqliteAdapterError

        with patch(
            "shardyfusion.sqlite_adapter.S3ReadOnlyFile",
            side_effect=S3VfsError("bad config"),
        ):
            with pytest.raises(SqliteAdapterError, match="bad config"):
                SqliteRangeShardReader(
                    db_url="s3://bucket/shard",
                    local_dir=tmp_path / "read",
                    checkpoint_id=None,
                )


# ---------------------------------------------------------------------------
# Smoke test: per-instance VFS names don't collide across multiple readers
# ---------------------------------------------------------------------------


class TestMultipleReaders:
    @pytest.fixture()
    def db_bytes(self, tmp_path: Path) -> bytes:
        return _build_sqlite_db(tmp_path, [(b"a", b"1"), (b"b", b"2")])

    def test_two_readers_open_simultaneously(
        self, tmp_path: Path, db_bytes: bytes, fake_obstore: _FakeObstore
    ) -> None:
        r1 = _open_reader(
            fake_obstore,
            db_bytes=db_bytes,
            local_dir=tmp_path / "r1",
            db_url="s3://bucket/shard1",
        )
        r2 = _open_reader(
            fake_obstore,
            db_bytes=db_bytes,
            local_dir=tmp_path / "r2",
            db_url="s3://bucket/shard2",
        )
        try:
            assert r1.get(b"a") == b"1"
            assert r2.get(b"b") == b"2"
        finally:
            r1.close()
            r2.close()


# Suppress unused-import warning for MagicMock — kept for future tests.
_ = MagicMock
