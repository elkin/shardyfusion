"""Unit tests for the SQLite range-read backend."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion._sqlite_vfs import S3ReadOnlyFile, S3VfsError


class TestS3ReadOnlyFile:
    """Test the core S3 range-read I/O layer."""

    @pytest.fixture()
    def s3_client(self) -> MagicMock:
        client = MagicMock()
        client.head_object.return_value = {"ContentLength": 8192}
        return client

    def test_size_from_head(self, s3_client: MagicMock) -> None:
        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = S3ReadOnlyFile(bucket="b", key="k")
        assert f.size == 8192

    def test_read_sends_range_header(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x00" * 4096
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = S3ReadOnlyFile(bucket="b", key="k")

        data = f.read(0, 4096)
        s3_client.get_object.assert_called_once_with(
            Bucket="b", Key="k", Range="bytes=0-4095"
        )
        assert len(data) == 4096

    def test_page_cache_hit(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x01" * 100
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=10)

        # First read — cache miss
        data1 = f.read(0, 100)
        assert s3_client.get_object.call_count == 1

        # Second read — cache hit
        data2 = f.read(0, 100)
        assert s3_client.get_object.call_count == 1  # no new S3 call
        assert data1 == data2

    def test_page_cache_eviction(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x00" * 10
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=2)

        # Fill cache: 2 pages
        f.read(0, 10)
        f.read(10, 10)
        assert s3_client.get_object.call_count == 2

        # Third unique read evicts first page
        f.read(20, 10)
        assert s3_client.get_object.call_count == 3

        # Re-read first page — cache miss
        f.read(0, 10)
        assert s3_client.get_object.call_count == 4

    def test_read_past_eof_returns_empty(self, s3_client: MagicMock) -> None:
        """Reads at or past the file size return empty bytes (no S3 call)."""
        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = S3ReadOnlyFile(bucket="b", key="k")  # size=8192

        assert f.read(8192, 100) == b""
        assert f.read(9999, 50) == b""
        s3_client.get_object.assert_not_called()

    def test_page_cache_hit_refreshes_recency(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x00" * 10
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=2)

        f.read(0, 10)
        f.read(10, 10)
        assert s3_client.get_object.call_count == 2

        # Touch the first page so it becomes most recently used.
        f.read(0, 10)
        assert s3_client.get_object.call_count == 2

        # A third unique page should evict the second page, not the first one.
        f.read(20, 10)
        assert s3_client.get_object.call_count == 3

        f.read(0, 10)
        assert s3_client.get_object.call_count == 3

        f.read(10, 10)
        assert s3_client.get_object.call_count == 4

    def test_page_cache_zero_disables_caching(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x00" * 10
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=0)

        f.read(0, 10)
        f.read(0, 10)

        assert s3_client.get_object.call_count == 2

    def test_negative_page_cache_size_is_rejected(self, s3_client: MagicMock) -> None:
        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            with pytest.raises(S3VfsError, match="page_cache_pages must be >= 0"):
                S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=-1)


# ---------------------------------------------------------------------------
# APSW VFS integration (requires apsw)
# ---------------------------------------------------------------------------

apsw = pytest.importorskip("apsw")

from shardyfusion.sqlite_adapter import SqliteRangeShardReader  # noqa: E402


def _build_sqlite_db(tmp_path: Path, rows: list[tuple[bytes, bytes]]) -> bytes:
    """Build a SQLite DB with kv table and return its raw bytes."""
    db_path = tmp_path / "source.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA page_size = 4096")
    conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
    conn.executemany("INSERT INTO kv (k, v) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return db_path.read_bytes()


class TestSqliteRangeShardReaderVFS:
    """End-to-end test: SqliteRangeShardReader through the APSW VFS pipeline."""

    @pytest.fixture()
    def db_bytes(self, tmp_path: Path) -> bytes:
        return _build_sqlite_db(tmp_path, [(b"key1", b"val1"), (b"key2", b"val2")])

    def _mock_s3_file(self, db_bytes: bytes) -> MagicMock:
        mock = MagicMock(spec=S3ReadOnlyFile)
        mock.size = len(db_bytes)
        mock.read.side_effect = lambda offset, amount: db_bytes[
            offset : offset + amount
        ]
        return mock

    def test_get_through_vfs(self, tmp_path: Path, db_bytes: bytes) -> None:
        mock = self._mock_s3_file(db_bytes)
        with patch("shardyfusion.sqlite_adapter.S3ReadOnlyFile", return_value=mock):
            reader = SqliteRangeShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
                checkpoint_id=None,
            )

        assert reader.get(b"key1") == b"val1"
        assert reader.get(b"key2") == b"val2"
        assert reader.get(b"missing") is None
        reader.close()

    def test_close_unregisters_vfs(self, tmp_path: Path, db_bytes: bytes) -> None:
        mock = self._mock_s3_file(db_bytes)
        with patch("shardyfusion.sqlite_adapter.S3ReadOnlyFile", return_value=mock):
            reader = SqliteRangeShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
                checkpoint_id=None,
            )

        vfs_names_before = set(apsw.vfsnames())
        reader.close()
        vfs_names_after = set(apsw.vfsnames())
        assert len(vfs_names_after) < len(vfs_names_before)
