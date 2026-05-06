"""Unit tests for the SQLite btree-metadata sidecar extraction.

The sidecar bundles all interior B-tree pages plus every ``sqlite_master``
page so a range-read reader can fetch them once and pin them in cache.
These tests verify the extraction (driven by SQLite's own ``dbstat`` /
``sqlite_dbpage`` virtual tables, via APSW), the binary blob layout, and
the graceful-degradation paths when APSW or its virtual tables are absent.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

apsw = pytest.importorskip("apsw")
zstandard = pytest.importorskip("zstandard")

from shardyfusion.sqlite_adapter import (  # noqa: E402
    _BTREEMETA_FORMAT_VERSION,
    _BTREEMETA_MAGIC,
    BtreeMetaUnavailableError,
    extract_btree_metadata,
)


def _build_kv_db(path: Path, *, rows: int, page_size: int = 4096) -> None:
    """Build a small SQLite KV database matching the writer's PRAGMAs."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute(f"PRAGMA page_size = {page_size}")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
    conn.execute("BEGIN")
    pairs = [(i.to_bytes(8, "big"), b"v" * 32) for i in range(rows)]
    conn.executemany("INSERT INTO kv (k, v) VALUES (?, ?)", pairs)
    conn.execute("COMMIT")
    conn.close()


def _parse_sidecar(blob: bytes) -> tuple[int, int, int, list[int], list[bytes]]:
    """Parse the sidecar blob and return (format_version, page_size, n,
    page_numbers, page_blobs).

    Format v3: ``magic(8) + version(4=3) + zstd(body)`` where the body is
    ``page_size(4) + n(4) + (pageno, offset)*n + page_data(page_size*n)``.
    """
    import struct

    assert blob[:8] == _BTREEMETA_MAGIC
    fv = int.from_bytes(blob[8:12], "little")
    body = zstandard.ZstdDecompressor().decompress(blob[12:])
    page_size, n = struct.unpack("<II", body[:8])
    pairs = struct.unpack(f"<{2 * n}I", body[8 : 8 + 8 * n]) if n else ()
    page_nums: list[int] = list(pairs[0::2])
    offsets: list[int] = list(pairs[1::2])
    blobs = [body[off : off + page_size] for off in offsets]
    # Sanity: index offsets must point at page-aligned slabs in body.
    expected_data_start = 8 + 8 * n
    for i, off in enumerate(offsets):
        assert off == expected_data_start + i * page_size, (
            f"offset[{i}]={off} expected={expected_data_start + i * page_size}"
        )
    return fv, page_size, n, page_nums, blobs


def _all_page_types(db_path: Path) -> dict[int, tuple[str | None, str | None]]:
    """Return ``{pageno: (pagetype, name)}`` for every page in the DB."""
    conn = apsw.Connection(
        f"file:{db_path}?mode=ro",
        flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
    )
    try:
        rows = (
            conn.cursor()
            .execute("SELECT pageno, pagetype, name FROM dbstat ORDER BY pageno")
            .fetchall()
        )
    finally:
        conn.close()
    return {int(p): (pt, nm) for p, pt, nm in rows}


# ---------------------------------------------------------------------------
# Basic format
# ---------------------------------------------------------------------------


class TestExtractBtreeMetadata:
    def test_single_row_minimal_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "minimal.db"
        _build_kv_db(db_path, rows=1)

        blob = extract_btree_metadata(db_path)
        fv, page_size, n, page_nums, blobs = _parse_sidecar(blob)

        assert fv == _BTREEMETA_FORMAT_VERSION
        assert page_size == 4096
        # One-row schema: only sqlite_master root (page 1) is needed.
        assert n >= 1
        assert page_nums[0] == 1
        # Slab matches source.
        raw = db_path.read_bytes()
        for pgno, slab in zip(page_nums, blobs, strict=True):
            assert slab == raw[(pgno - 1) * page_size : pgno * page_size]

    def test_format_compressed_smaller_than_uncompressed(self, tmp_path: Path) -> None:
        # The sidecar body is gzip-compressed (format v2). Btree pages
        # compress ~10× in practice — a sanity check that the on-the-wire
        # blob is meaningfully smaller than what an uncompressed body
        # would be, and that the magic + version header is still
        # readable cheaply without decompressing.
        db_path = tmp_path / "size.db"
        _build_kv_db(db_path, rows=10_000)
        blob = extract_btree_metadata(db_path)
        _, page_size, n, _, _ = _parse_sidecar(blob)
        uncompressed_size = 20 + n * (4 + page_size)
        assert len(blob) < uncompressed_size, (
            f"compressed sidecar should be smaller than uncompressed "
            f"({len(blob)} vs {uncompressed_size})"
        )
        # Magic and version are not compressed; reader can sniff them
        # without invoking gzip.
        assert blob[:8] == _BTREEMETA_MAGIC


# ---------------------------------------------------------------------------
# Page selection: KV-only multi-page
# ---------------------------------------------------------------------------


class TestPageSelectionKv:
    def test_includes_interior_btree_pages(self, tmp_path: Path) -> None:
        db_path = tmp_path / "kv.db"
        # Enough rows that the kv btree definitely has at least one interior
        # level (typical kv leaf holds tens of rows; ~10k forces ≥1 interior).
        _build_kv_db(db_path, rows=10_000)

        blob = extract_btree_metadata(db_path)
        _, page_size, _, page_nums, blobs = _parse_sidecar(blob)
        types = _all_page_types(db_path)

        # Sorted ascending.
        assert page_nums == sorted(page_nums)
        # Every selected page is either an interior btree page or part of
        # the schema btree (named "sqlite_master" or "sqlite_schema"
        # depending on SQLite version).
        schema_names = {"sqlite_master", "sqlite_schema"}
        for pgno in page_nums:
            pagetype, name = types[pgno]
            assert pagetype == "internal" or name in schema_names, (
                f"unexpected included page {pgno}: pagetype={pagetype!r} name={name!r}"
            )

        # No leaf pages from non-schema btrees should appear.
        included = set(page_nums)
        for pgno, (pagetype, name) in types.items():
            if pagetype == "leaf" and name not in schema_names:
                assert pgno not in included

        # At least one interior page beyond the schema (proves we actually
        # got the kv interior).
        kv_interior = [
            p
            for p, (pt, nm) in types.items()
            if pt == "internal" and nm not in schema_names
        ]
        assert kv_interior, "expected at least one interior kv page"
        for p in kv_interior:
            assert p in included

        # Slabs match source bytes byte-for-byte.
        raw = db_path.read_bytes()
        for pgno, slab in zip(page_nums, blobs, strict=True):
            assert slab == raw[(pgno - 1) * page_size : pgno * page_size]

    def test_multi_page_sqlite_master(self, tmp_path: Path) -> None:
        # Force sqlite_master to span multiple pages by adding many tables.
        # Each CREATE TABLE row in sqlite_master holds the schema text, so a
        # few dozen tables (with reasonably long names) will exceed one page.
        db_path = tmp_path / "many_tables.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA page_size = 4096")
        for i in range(120):
            # Long-ish table name + multi-column schema bloats sqlite_master.
            cols = ", ".join(f"col_{j} INTEGER" for j in range(20))
            conn.execute(
                f"CREATE TABLE table_with_long_name_for_bloat_{i:04d} ({cols})"
            )
        conn.close()

        blob = extract_btree_metadata(db_path)
        _, _, _, page_nums, _ = _parse_sidecar(blob)
        types = _all_page_types(db_path)

        schema_names = {"sqlite_master", "sqlite_schema"}
        master_pages = [p for p, (_, nm) in types.items() if nm in schema_names]
        assert len(master_pages) > 1, (
            "expected the schema btree to span multiple pages for this test"
        )
        # Every schema-btree page is in the sidecar regardless of its
        # pagetype (interior OR leaf).
        for p in master_pages:
            assert p in set(page_nums)


# ---------------------------------------------------------------------------
# Unified KV+vector
# ---------------------------------------------------------------------------


class TestPageSelectionUnified:
    def test_unified_kv_vector_extracts_cleanly(self, tmp_path: Path) -> None:
        # Exercise extraction against an SqliteVecAdapter-shaped DB.
        # Skip if sqlite-vec isn't installed: the fixture cannot be built.
        pytest.importorskip("sqlite_vec")

        from unittest.mock import MagicMock, patch

        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecAdapter

        local_dir = tmp_path / "vec_shard"
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            with SqliteVecAdapter(
                db_url="s3://bucket/vec",
                local_dir=local_dir,
                vector_spec=VectorSpec(dim=8, metric="cosine", index_type="flat"),
            ) as adapter:
                adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])
                import numpy as np

                ids = np.array([1, 2, 3], dtype=np.int64)
                vecs = np.random.rand(3, 8).astype(np.float32)
                adapter.write_vector_batch(ids, vecs)
                adapter.checkpoint()

        db_path = local_dir / "shard.db"
        assert db_path.exists()

        blob = extract_btree_metadata(db_path)
        _, page_size, _, page_nums, blobs = _parse_sidecar(blob)
        types = _all_page_types(db_path)

        # All schema-btree pages are present (necessary for SQLite to open).
        schema_names = {"sqlite_master", "sqlite_schema"}
        for p, (_, nm) in types.items():
            if nm in schema_names:
                assert p in set(page_nums)

        # Slabs match source bytes.
        raw = db_path.read_bytes()
        for pgno, slab in zip(page_nums, blobs, strict=True):
            assert slab == raw[(pgno - 1) * page_size : pgno * page_size]


# ---------------------------------------------------------------------------
# Graceful-degradation paths
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_apsw_unavailable_raises_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "x.db"
        _build_kv_db(db_path, rows=1)

        # Force ``import apsw`` inside the function to fail.
        monkeypatch.setitem(sys.modules, "apsw", None)
        with pytest.raises(BtreeMetaUnavailableError) as excinfo:
            extract_btree_metadata(db_path)
        assert excinfo.value.args[0] == "apsw_not_installed"

    def test_wal_journal_mode_raises_unsupported(self, tmp_path: Path) -> None:
        """A future writer that flips to ``journal_mode=WAL`` must not
        silently break sidecar emission. The extractor refuses to run
        on a WAL-mode file because its raw-file I/O could miss pages
        parked in the ``-wal`` sidecar."""
        import sqlite3

        db_path = tmp_path / "wal.db"
        c = sqlite3.connect(str(db_path))
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("CREATE TABLE kv (k INTEGER PRIMARY KEY, v TEXT)")
        c.execute("INSERT INTO kv VALUES (1, ?)", ("hello",))
        c.commit()
        c.close()

        with pytest.raises(BtreeMetaUnavailableError) as excinfo:
            extract_btree_metadata(db_path)
        assert excinfo.value.args[0] == "unsupported_journal_mode"

    def test_off_journal_mode_succeeds(self, tmp_path: Path) -> None:
        """Sanity check: the writer's actual mode (``OFF``) is on the
        allow-list and extraction proceeds."""
        import sqlite3

        db_path = tmp_path / "off.db"
        c = sqlite3.connect(str(db_path), isolation_level=None)
        c.execute("PRAGMA journal_mode = OFF")
        c.execute("CREATE TABLE kv (k INTEGER PRIMARY KEY, v TEXT)")
        c.execute("BEGIN")
        c.execute("INSERT INTO kv VALUES (1, ?)", ("hello",))
        c.execute("COMMIT")
        c.close()

        # Should not raise.
        blob = extract_btree_metadata(db_path)
        assert blob[:8] == _BTREEMETA_MAGIC

    def test_dbstat_unavailable_raises_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "x.db"
        _build_kv_db(db_path, rows=1)

        # Build a stub apsw module that mimics the real apsw surface but
        # makes any query touching dbstat raise apsw.SQLError. We let
        # sqlite_dbpage probes succeed so the failure is specifically
        # attributed to dbstat.
        import types

        real_apsw = sys.modules["apsw"]
        stub = types.ModuleType("apsw")
        stub.SQLError = real_apsw.SQLError  # type: ignore[attr-defined]
        stub.SQLITE_OPEN_READONLY = real_apsw.SQLITE_OPEN_READONLY  # type: ignore[attr-defined]
        stub.SQLITE_OPEN_URI = real_apsw.SQLITE_OPEN_URI  # type: ignore[attr-defined]

        class _StubCursor:
            def __init__(self, real_cursor: object) -> None:
                self._real = real_cursor

            def execute(self, sql: str, params: tuple = ()):  # type: ignore[no-untyped-def]
                if "dbstat" in sql:
                    raise real_apsw.SQLError("no such table: dbstat")  # type: ignore[attr-defined]
                return self._real.execute(sql, params)  # type: ignore[attr-defined]

            def fetchall(self):  # type: ignore[no-untyped-def]
                return self._real.fetchall()  # type: ignore[attr-defined]

            def fetchone(self):  # type: ignore[no-untyped-def]
                return self._real.fetchone()  # type: ignore[attr-defined]

        class _StubConnection:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self._inner = real_apsw.Connection(*args, **kwargs)  # type: ignore[attr-defined]

            def cursor(self):  # type: ignore[no-untyped-def]
                return _StubCursor(self._inner.cursor())

            def close(self) -> None:
                self._inner.close()

        stub.Connection = _StubConnection  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "apsw", stub)
        with pytest.raises(BtreeMetaUnavailableError) as excinfo:
            extract_btree_metadata(db_path)
        assert excinfo.value.args[0] == "dbstat_unavailable"
