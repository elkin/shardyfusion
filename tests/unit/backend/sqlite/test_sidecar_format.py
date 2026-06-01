"""Unit tests for the v7 SQLite page-cache sidecar.

The sidecar bundles every interior B-tree page plus every schema-btree page,
each **gap-stripped** (the unallocated middle removed), behind a vendor-neutral
``SQPC`` prefix that carries a ``u8`` version and the ``.db`` object tag (S3
ETag) for a correctness binding.  A reader reconstructs the full pages on read.

These tests verify the wire format, the gap-strip / reconstruct round-trip
(via ``PRAGMA integrity_check`` on a rebuilt DB — the correctness contract),
the ETag binding, the overflow-chain CSR index, and the graceful-degradation paths.
"""

from __future__ import annotations

import sqlite3
import struct
import sys
from pathlib import Path

import pytest

apsw = pytest.importorskip("apsw")
zstandard = pytest.importorskip("zstandard")

from shardyfusion.sqlite_adapter import (  # noqa: E402
    _SIDECAR_FORMAT_VERSION,
    _SIDECAR_MAGIC,
    SidecarUnavailableError,
    extract_sidecar,
)
from tests.helpers.sidecar import (  # noqa: E402
    parse_sidecar,
    reconstruct_page,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_kv_db(path: Path, *, rows: int, page_size: int = 4096) -> None:
    """Build a small SQLite KV database matching the writer's PRAGMAs."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute(f"PRAGMA page_size = {page_size}")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO kv (k, v) VALUES (?, ?)",
        [(i.to_bytes(8, "big"), b"v" * 32) for i in range(rows)],
    )
    conn.execute("COMMIT")
    conn.close()


def _build_kv_db_with_large_values(
    path: Path, *, rows: int, value_bytes: int, page_size: int = 4096
) -> None:
    """Build a small DB whose kv values are guaranteed to overflow."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute(f"PRAGMA page_size = {page_size}")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO kv (k, v) VALUES (?, ?)",
        [(i.to_bytes(8, "big"), b"x" * value_bytes) for i in range(rows)],
    )
    conn.execute("COMMIT")
    conn.close()


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
# v7 parser + reference reconstruction (the reader side)
# ---------------------------------------------------------------------------


def _parse_sidecar(
    blob: bytes,
) -> tuple[int, str | None, int, int, list[int], list[bytes], list[list[int]]]:
    """Parse a v7 sidecar via the shared helper, as a positional tuple."""
    p = parse_sidecar(blob)
    return (
        p.version,
        p.db_tag,
        p.page_size,
        p.n,
        p.pagenos,
        p.stored_pages,
        p.chains,
    )


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def _wrap_test_body(body: bytes, *, body_size: int | None = None) -> bytes:
    compressor = zstandard.ZstdCompressor(level=3, write_checksum=True)
    return b"".join(
        [
            _SIDECAR_MAGIC,
            _SIDECAR_FORMAT_VERSION.to_bytes(1, "little"),
            (len(body) if body_size is None else body_size).to_bytes(8, "little"),
            b"\x00",  # unbound tag
            compressor.compress(body),
        ]
    )


def _synthetic_body(
    *,
    page_size: int = 512,
    pagenos: list[int] | None = None,
    offsets: list[int] | None = None,
    chain_heads: list[int] | None = None,
    chain_offsets: list[int] | None = None,
    chain_pages: list[int] | None = None,
) -> bytes:
    pagenos = pagenos if pagenos is not None else [1]
    pages = b"\x00" * (page_size * len(pagenos))
    offsets = (
        offsets
        if offsets is not None
        else [i * page_size for i in range(len(pagenos) + 1)]
    )
    chain_heads = chain_heads or []
    chain_offsets = chain_offsets if chain_offsets is not None else [0]
    chain_pages = chain_pages or []
    return b"".join(
        [
            struct.pack("<II", page_size, len(pagenos)),
            struct.pack(f"<{len(pagenos)}I", *pagenos) if pagenos else b"",
            struct.pack(f"<{len(offsets)}I", *offsets),
            struct.pack("<I", len(chain_heads)),
            struct.pack(f"<{len(chain_heads)}I", *chain_heads) if chain_heads else b"",
            struct.pack(f"<{len(chain_offsets)}I", *chain_offsets),
            struct.pack(f"<{len(chain_pages)}I", *chain_pages) if chain_pages else b"",
            pages,
        ]
    )


class TestFormat:
    def test_prefix_magic_version_and_unbound_tag(self, tmp_path: Path) -> None:
        db_path = tmp_path / "minimal.db"
        _build_kv_db(db_path, rows=1)
        blob = extract_sidecar(db_path)

        # Vendor-neutral 4-byte magic + 1-byte version, readable without zstd.
        assert blob[:4] == _SIDECAR_MAGIC == b"SQPC"
        assert blob[4] == _SIDECAR_FORMAT_VERSION == 7
        assert blob[13] == 0  # tag_len: unbound when no db_tag is passed

        version, tag, page_size, n, pagenos, _, _ = _parse_sidecar(blob)
        assert version == 7
        assert tag is None
        assert page_size == 4096
        assert n >= 1
        assert pagenos[0] == 1  # schema root is always present

    def test_db_tag_is_embedded_and_round_trips(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tagged.db"
        _build_kv_db(db_path, rows=1)
        etag = '"d41d8cd98f00b204e9800998ecf8427e-3"'  # multipart-shaped ETag
        blob = extract_sidecar(db_path, db_tag=etag)

        assert blob[13] == len(etag.encode("utf-8"))
        _, tag, *_ = _parse_sidecar(blob)
        assert tag == etag

    def test_wire_smaller_than_raw_pages(self, tmp_path: Path) -> None:
        db_path = tmp_path / "size.db"
        _build_kv_db(db_path, rows=10_000)
        blob = extract_sidecar(db_path)
        _, _, page_size, n, _, _, _ = _parse_sidecar(blob)
        # Compressed, gap-stripped sidecar is far smaller than the raw pages
        # it represents.
        assert len(blob) < n * page_size

    def test_shared_parser_rejects_mismatched_body_size(self) -> None:
        body = _synthetic_body()
        with pytest.raises(AssertionError):
            parse_sidecar(_wrap_test_body(body, body_size=len(body) + 1))

    def test_shared_parser_rejects_unsorted_page_index(self) -> None:
        body = _synthetic_body(pagenos=[2, 1])
        with pytest.raises(AssertionError):
            parse_sidecar(_wrap_test_body(body))

    def test_shared_parser_rejects_bad_page_offsets(self) -> None:
        body = _synthetic_body(offsets=[1, 512])
        with pytest.raises(AssertionError):
            parse_sidecar(_wrap_test_body(body))

    def test_shared_parser_rejects_bad_chain_csr(self) -> None:
        body = _synthetic_body(
            chain_heads=[7],
            chain_offsets=[0, 1],
            chain_pages=[8],
        )
        with pytest.raises(AssertionError):
            parse_sidecar(_wrap_test_body(body))


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------


class TestPageSelection:
    def test_includes_interior_and_schema_only(self, tmp_path: Path) -> None:
        db_path = tmp_path / "kv.db"
        _build_kv_db(db_path, rows=10_000)  # forces >=1 interior level
        _, _, _, _, pagenos, _, _ = _parse_sidecar(extract_sidecar(db_path))
        types = _all_page_types(db_path)
        schema = {"sqlite_master", "sqlite_schema"}

        assert pagenos == sorted(pagenos)
        for pgno in pagenos:
            pagetype, name = types[pgno]
            assert pagetype == "internal" or name in schema, (pgno, pagetype, name)

        included = set(pagenos)
        for pgno, (pagetype, name) in types.items():
            if pagetype == "leaf" and name not in schema:
                assert pgno not in included

        kv_interior = [
            p for p, (pt, nm) in types.items() if pt == "internal" and nm not in schema
        ]
        assert kv_interior, "expected at least one interior kv page"
        assert set(kv_interior) <= included

    def test_multi_page_schema_btree_fully_included(self, tmp_path: Path) -> None:
        db_path = tmp_path / "many_tables.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA page_size = 4096")
        for i in range(120):
            cols = ", ".join(f"col_{j} INTEGER" for j in range(20))
            conn.execute(
                f"CREATE TABLE table_with_long_name_for_bloat_{i:04d} ({cols})"
            )
        conn.close()

        _, _, _, _, pagenos, _, _ = _parse_sidecar(extract_sidecar(db_path))
        types = _all_page_types(db_path)
        schema = {"sqlite_master", "sqlite_schema"}
        master_pages = [p for p, (_, nm) in types.items() if nm in schema]
        assert len(master_pages) > 1
        assert set(master_pages) <= set(pagenos)


# ---------------------------------------------------------------------------
# Gap stripping (correctness-critical)
# ---------------------------------------------------------------------------


class TestGapStripping:
    def test_reconstruct_preserves_integrity_and_rows(self, tmp_path: Path) -> None:
        """The contract: rebuilding the DB from reconstructed sidecar pages
        leaves SQLite's view unchanged — ``integrity_check`` stays ``ok`` and
        every row reads back identically — because SQLite never reads the gap."""
        db_path = tmp_path / "kv.db"
        _build_kv_db(db_path, rows=10_000)
        ps = 4096
        _, _, page_size, n, pagenos, stored, _ = _parse_sidecar(
            extract_sidecar(db_path)
        )
        assert page_size == ps
        assert pagenos[0] == 1  # exercises the page-1 (base=100) path

        raw = bytearray(db_path.read_bytes())
        stored_total = 0
        for pgno, s in zip(pagenos, stored, strict=True):
            stored_total += len(s)
            recon = reconstruct_page(s, pgno, ps)
            assert len(recon) == ps
            original = bytes(raw[(pgno - 1) * ps : pgno * ps])
            # Reconstruction only zeroes bytes — it never alters a real byte.
            for a, b in zip(original, recon, strict=True):
                assert b == a or b == 0
            raw[(pgno - 1) * ps : pgno * ps] = recon

        # Stripping actually removed bytes (interior pages carry real gaps).
        assert stored_total < n * ps

        rebuilt = tmp_path / "rebuilt.db"
        rebuilt.write_bytes(bytes(raw))
        conn = sqlite3.connect(str(rebuilt))
        try:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            got = conn.execute("SELECT k, v FROM kv ORDER BY k").fetchall()
        finally:
            conn.close()
        src = sqlite3.connect(str(db_path))
        try:
            want = src.execute("SELECT k, v FROM kv ORDER BY k").fetchall()
        finally:
            src.close()
        assert got == want

    def test_reserved_bytes_db_is_skipped(self, tmp_path: Path) -> None:
        """A DB whose header reserves per-page bytes (checksum/encryption VFS)
        is unsafe to gap-strip → the sidecar is skipped, not silently wrong."""
        db_path = tmp_path / "reserved.db"
        _build_kv_db(db_path, rows=10)
        tampered = bytearray(db_path.read_bytes())
        tampered[20] = 32  # bytes-reserved-per-page
        with pytest.raises(SidecarUnavailableError) as exc:
            extract_sidecar(db_path, db_bytes=bytes(tampered))
        assert exc.value.args[0] == "reserved_bytes_unsupported"


# ---------------------------------------------------------------------------
# Overflow chain CSR index
# ---------------------------------------------------------------------------


class TestOverflowChains:
    def test_no_chains_when_values_fit_inline(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tiny.db"
        _build_kv_db(db_path, rows=10)
        _, _, _, _, _, _, chains = _parse_sidecar(extract_sidecar(db_path))
        assert chains == []

    def test_single_overflow_page_chain(self, tmp_path: Path) -> None:
        db_path = tmp_path / "short.db"
        _build_kv_db_with_large_values(db_path, rows=20, value_bytes=3000)
        _, _, _, _, _, _, chains = _parse_sidecar(extract_sidecar(db_path))
        assert len(chains) == 20
        assert all(len(c) == 1 for c in chains)

    def test_multi_page_chain_in_order(self, tmp_path: Path) -> None:
        db_path = tmp_path / "long.db"
        _build_kv_db_with_large_values(db_path, rows=20, value_bytes=20_000)
        _, _, _, _, _, _, chains = _parse_sidecar(extract_sidecar(db_path))
        assert len(chains) == 20
        assert {len(c) for c in chains} == {5}

        seen: set[int] = set()
        for chain in chains:
            for p in chain:
                assert p not in seen
                seen.add(p)

        raw = db_path.read_bytes()
        for chain in chains:
            for src, dst in zip(chain[:-1], chain[1:], strict=True):
                off = (src - 1) * 4096
                assert int.from_bytes(raw[off : off + 4], "big") == dst
            tail = (chain[-1] - 1) * 4096
            assert int.from_bytes(raw[tail : tail + 4], "big") == 0


# ---------------------------------------------------------------------------
# Unified KV + vector
# ---------------------------------------------------------------------------


class TestPageSelectionUnified:
    def test_unified_kv_vector_round_trips(self, tmp_path: Path) -> None:
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
                adapter.seal()

        db_path = local_dir / "shard.db"
        assert db_path.exists()
        _, _, page_size, _, pagenos, stored, _ = _parse_sidecar(
            extract_sidecar(db_path)
        )

        types = _all_page_types(db_path)
        schema = {"sqlite_master", "sqlite_schema"}
        for p, (_, nm) in types.items():
            if nm in schema:
                assert p in set(pagenos)

        # Every stored page reconstructs to a full, byte-consistent page.
        raw = db_path.read_bytes()
        for pgno, s in zip(pagenos, stored, strict=True):
            recon = reconstruct_page(s, pgno, page_size)
            assert len(recon) == page_size
            original = raw[(pgno - 1) * page_size : pgno * page_size]
            for a, b in zip(original, recon, strict=True):
                assert b == a or b == 0


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_apsw_unavailable_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "x.db"
        _build_kv_db(db_path, rows=1)
        monkeypatch.setitem(sys.modules, "apsw", None)
        with pytest.raises(SidecarUnavailableError) as exc:
            extract_sidecar(db_path)
        assert exc.value.args[0] == "apsw_not_installed"

    def test_wal_journal_mode_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "wal.db"
        c = sqlite3.connect(str(db_path))
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("CREATE TABLE kv (k INTEGER PRIMARY KEY, v TEXT)")
        c.execute("INSERT INTO kv VALUES (1, ?)", ("hello",))
        c.commit()
        c.close()
        with pytest.raises(SidecarUnavailableError) as exc:
            extract_sidecar(db_path)
        assert exc.value.args[0] == "unsupported_journal_mode"

    def test_off_journal_mode_succeeds(self, tmp_path: Path) -> None:
        db_path = tmp_path / "off.db"
        c = sqlite3.connect(str(db_path), isolation_level=None)
        c.execute("PRAGMA journal_mode = OFF")
        c.execute("CREATE TABLE kv (k INTEGER PRIMARY KEY, v TEXT)")
        c.execute("BEGIN")
        c.execute("INSERT INTO kv VALUES (1, ?)", ("hello",))
        c.execute("COMMIT")
        c.close()
        assert extract_sidecar(db_path)[:4] == _SIDECAR_MAGIC

    def test_dbstat_unavailable_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types as _types

        db_path = tmp_path / "x.db"
        _build_kv_db(db_path, rows=1)

        real_apsw = sys.modules["apsw"]
        stub = _types.ModuleType("apsw")
        stub.SQLError = real_apsw.SQLError  # type: ignore[attr-defined]
        stub.SQLITE_OPEN_READONLY = real_apsw.SQLITE_OPEN_READONLY  # type: ignore[attr-defined]
        stub.SQLITE_OPEN_URI = real_apsw.SQLITE_OPEN_URI  # type: ignore[attr-defined]

        class _StubCursor:
            def __init__(self, real: object) -> None:
                self._real = real

            def execute(self, sql: str, params: tuple = ()):  # type: ignore[no-untyped-def]
                if "dbstat" in sql:
                    raise real_apsw.SQLError("no such table: dbstat")  # type: ignore[attr-defined]
                return self._real.execute(sql, params)  # type: ignore[attr-defined]

            def fetchall(self):  # type: ignore[no-untyped-def]
                return self._real.fetchall()  # type: ignore[attr-defined]

        class _StubConnection:
            def __init__(self, *a: object, **k: object) -> None:
                self._inner = real_apsw.Connection(*a, **k)  # type: ignore[attr-defined]

            def cursor(self):  # type: ignore[no-untyped-def]
                return _StubCursor(self._inner.cursor())

            def close(self) -> None:
                self._inner.close()

        stub.Connection = _StubConnection  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "apsw", stub)
        with pytest.raises(SidecarUnavailableError) as exc:
            extract_sidecar(db_path)
        assert exc.value.args[0] == "dbstat_unavailable"
