"""Tests for SqliteVecAdapter — unified KV + vector SQLite shard lifecycle."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# sqlite-vec is optional; skip entire module if unavailable.
sqlite_vec = pytest.importorskip("sqlite_vec")

from shardyfusion.sqlite_vec_adapter import (  # noqa: E402
    SqliteVecAdapter,
    SqliteVecAdapterError,
    SqliteVecFactory,
    SqliteVecReaderFactory,
    SqliteVecShardReader,
    _is_cached_snapshot_current,
)
from shardyfusion.vector.types import SearchResult  # noqa: E402

# ---------------------------------------------------------------------------
# Writer lifecycle
# ---------------------------------------------------------------------------


class TestSqliteVecAdapterLifecycle:
    """Write → checkpoint → close → upload flow."""

    def test_write_kv_and_checkpoint(self, tmp_path: Path) -> None:
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=tmp_path / "shard",
            vector_spec=MagicMock(dim=4),
        )
        adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])
        ckpt = adapter.checkpoint()

        assert ckpt is not None
        assert len(ckpt) == 64  # SHA-256 hex digest

    def test_write_after_checkpoint_raises(self, tmp_path: Path) -> None:
        """After checkpoint, conn is closed so write_batch raises 'already closed'."""
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=tmp_path / "shard",
            vector_spec=MagicMock(dim=4),
        )
        adapter.checkpoint()

        with pytest.raises(SqliteVecAdapterError, match="already closed"):
            adapter.write_batch([(b"k", b"v")])

    def test_write_vector_after_checkpoint_raises(self, tmp_path: Path) -> None:
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=tmp_path / "shard",
            vector_spec=MagicMock(dim=4),
        )
        adapter.checkpoint()

        with pytest.raises(SqliteVecAdapterError, match="already closed"):
            adapter.write_vector_batch(
                np.array([1]), np.random.randn(1, 4).astype(np.float32)
            )

    def test_double_checkpoint_raises(self, tmp_path: Path) -> None:
        """After checkpoint, conn is None so second checkpoint raises 'already closed'."""
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=tmp_path / "shard",
            vector_spec=MagicMock(dim=4),
        )
        adapter.checkpoint()

        with pytest.raises(SqliteVecAdapterError, match="already closed"):
            adapter.checkpoint()

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        with patch("shardyfusion.sqlite_vec_adapter.put_bytes"):
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            )
            adapter.close()

        with pytest.raises(SqliteVecAdapterError, match="already closed"):
            adapter.write_batch([(b"k", b"v")])

    def test_close_uploads_to_s3(self, tmp_path: Path) -> None:
        with patch("shardyfusion.sqlite_vec_adapter.put_bytes") as mock_put:
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            )
            adapter.write_batch([(b"k1", b"v1")])
            adapter.close()

        mock_put.assert_called_once()
        call_args = mock_put.call_args
        assert call_args[0][0] == "s3://bucket/shard/shard.db"

    def test_close_idempotent(self, tmp_path: Path) -> None:
        with patch("shardyfusion.sqlite_vec_adapter.put_bytes") as mock_put:
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            )
            adapter.close()
            adapter.close()

        # Only uploaded once
        assert mock_put.call_count == 1

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with patch("shardyfusion.sqlite_vec_adapter.put_bytes"):
            with SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            ) as adapter:
                adapter.write_batch([(b"k", b"v")])
            assert adapter._closed


# ---------------------------------------------------------------------------
# Vector ID mapping
# ---------------------------------------------------------------------------


class TestSqliteVecIdMapping:
    """Integer vs string ID handling and id_map table population."""

    def _build_db(
        self,
        tmp_path: Path,
        ids: np.ndarray,
        dim: int = 4,
        payloads: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Build a SQLite-vec DB and return the DB file path."""
        shard_dir = tmp_path / "shard"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=dim),
        )
        vecs = np.random.randn(len(ids), dim).astype(np.float32)
        adapter.write_vector_batch(ids, vecs, payloads)
        adapter.checkpoint()
        return shard_dir / "shard.db"

    def test_integer_ids_no_id_map(self, tmp_path: Path) -> None:
        db_path = self._build_db(tmp_path, np.array([10, 20, 30]))

        conn = sqlite3.connect(str(db_path))
        sqlite_vec.load(conn)
        rows = conn.execute("SELECT * FROM vec_id_map").fetchall()
        conn.close()
        assert rows == []

    def test_string_ids_populate_id_map(self, tmp_path: Path) -> None:
        db_path = self._build_db(tmp_path, np.array(["a", "b", "c"], dtype=object))

        conn = sqlite3.connect(str(db_path))
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT internal_id, original_id FROM vec_id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()
        assert len(rows) == 3
        assert rows[0] == (0, "a")
        assert rows[1] == (1, "b")
        assert rows[2] == (2, "c")

    def test_int_ids_advance_next_vec_id(self, tmp_path: Path) -> None:
        """After writing int IDs [10, 20], _next_vec_id should be 21."""
        shard_dir = tmp_path / "shard"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=4),
        )
        vecs = np.random.randn(2, 4).astype(np.float32)
        adapter.write_vector_batch(np.array([10, 20]), vecs)
        assert adapter._next_vec_id == 21

        # Now write strings — should start at 21, no collision
        adapter.write_vector_batch(
            np.array(["x", "y"], dtype=object),
            np.random.randn(2, 4).astype(np.float32),
        )
        adapter.checkpoint()

        conn = sqlite3.connect(str(shard_dir / "shard.db"))
        sqlite_vec.load(conn)
        id_map = dict(
            conn.execute("SELECT internal_id, original_id FROM vec_id_map").fetchall()
        )
        conn.close()
        assert id_map == {21: "x", 22: "y"}

    def test_payloads_stored(self, tmp_path: Path) -> None:
        db_path = self._build_db(
            tmp_path,
            np.array([1, 2]),
            payloads=[{"color": "red"}, {"color": "blue"}],
        )

        conn = sqlite3.connect(str(db_path))
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT rowid, payload FROM vec_payloads ORDER BY rowid"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert json.loads(rows[0][1]) == {"color": "red"}

    def test_none_payloads_not_stored(self, tmp_path: Path) -> None:
        """When payloads list has None entries, those rows are skipped."""
        db_path = self._build_db(
            tmp_path,
            np.array([1, 2]),
            payloads=[{"color": "red"}, None],  # type: ignore[list-item]
        )

        conn = sqlite3.connect(str(db_path))
        sqlite_vec.load(conn)
        rows = conn.execute("SELECT rowid FROM vec_payloads").fetchall()
        conn.close()
        # Only rowid=1 should have a payload
        assert len(rows) == 1
        assert rows[0][0] == 1


# ---------------------------------------------------------------------------
# Reader (search + KV)
# ---------------------------------------------------------------------------


class TestSqliteVecShardReader:
    """End-to-end: write → read via SqliteVecShardReader."""

    def _build_shard(self, tmp_path: Path, dim: int = 4) -> bytes:
        """Build a shard DB with KV + vector data and return its raw bytes."""
        shard_dir = tmp_path / "build"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=dim),
        )
        adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])

        vecs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        adapter.write_vector_batch(
            np.array([100, 200]),
            vecs,
            [{"label": "first"}, {"label": "second"}],
        )
        adapter.checkpoint()
        return (shard_dir / "shard.db").read_bytes()

    def test_kv_get(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.get_bytes", return_value=db_bytes):
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        assert reader.get(b"k1") == b"v1"
        assert reader.get(b"missing") is None
        reader.close()

    def test_vector_search(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.get_bytes", return_value=db_bytes):
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        # Query close to [1, 0, 0, 0] → should return id=100 first
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = reader.search(query, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be closest to query
        assert results[0].id == 100
        assert results[0].payload == {"label": "first"}
        reader.close()

    def test_search_with_string_ids(self, tmp_path: Path) -> None:
        """Reader resolves string IDs from vec_id_map."""
        shard_dir = tmp_path / "build"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=4),
        )
        vecs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        adapter.write_vector_batch(np.array(["doc_a", "doc_b"], dtype=object), vecs)
        adapter.checkpoint()
        db_bytes = (shard_dir / "shard.db").read_bytes()

        with patch("shardyfusion.sqlite_vec_adapter.get_bytes", return_value=db_bytes):
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = reader.search(query, top_k=2)

        assert results[0].id == "doc_a"
        assert results[1].id == "doc_b"
        reader.close()

    def test_search_after_close_raises(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.get_bytes", return_value=db_bytes):
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        reader.close()
        with pytest.raises(SqliteVecAdapterError, match="closed"):
            reader.search(np.zeros(4, dtype=np.float32), top_k=1)

    def test_get_after_close_raises(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.get_bytes", return_value=db_bytes):
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        reader.close()
        with pytest.raises(SqliteVecAdapterError, match="closed"):
            reader.get(b"k1")

    def test_context_manager(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.get_bytes", return_value=db_bytes):
            with SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            ) as reader:
                assert reader.get(b"k1") == b"v1"
        assert reader._conn is None


# ---------------------------------------------------------------------------
# Cache validation
# ---------------------------------------------------------------------------


class TestCacheValidation:
    def test_no_identity_file(self, tmp_path: Path) -> None:
        assert not _is_cached_snapshot_current(
            tmp_path / "missing.json", "s3://b/k", "ckpt1"
        )

    def test_valid_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "identity.json"
        path.write_text(json.dumps({"db_url": "s3://b/k", "checkpoint_id": "ckpt1"}))
        assert _is_cached_snapshot_current(path, "s3://b/k", "ckpt1")

    def test_stale_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "identity.json"
        path.write_text(json.dumps({"db_url": "s3://b/k", "checkpoint_id": "old"}))
        assert not _is_cached_snapshot_current(path, "s3://b/k", "new")

    def test_stale_url(self, tmp_path: Path) -> None:
        path = tmp_path / "identity.json"
        path.write_text(json.dumps({"db_url": "s3://b/old", "checkpoint_id": "ckpt1"}))
        assert not _is_cached_snapshot_current(path, "s3://b/new", "ckpt1")

    def test_corrupted_json(self, tmp_path: Path) -> None:
        path = tmp_path / "identity.json"
        path.write_text("not json")
        assert not _is_cached_snapshot_current(path, "s3://b/k", "ckpt1")

    def test_cached_reader_skips_download(self, tmp_path: Path) -> None:
        """When cache is valid, reader should NOT call get_bytes."""
        read_dir = tmp_path / "read"
        read_dir.mkdir()
        identity = read_dir / "shard.identity.json"
        identity.write_text(
            json.dumps({"db_url": "s3://bucket/shard", "checkpoint_id": "abc"})
        )

        # Write a valid DB to the cache location
        db_path = read_dir / "shard.db"
        conn = sqlite3.connect(str(db_path))
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID"
        )
        conn.execute("CREATE VIRTUAL TABLE vec_index USING vec0(embedding float[4])")
        conn.execute(
            "CREATE TABLE vec_payloads (rowid INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE vec_id_map (internal_id INTEGER PRIMARY KEY, original_id TEXT NOT NULL)"
        )
        conn.close()

        with patch("shardyfusion.sqlite_vec_adapter.get_bytes") as mock_get:
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=read_dir,
                checkpoint_id="abc",
            )
            mock_get.assert_not_called()
        reader.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestSqliteVecFactory:
    def test_factory_creates_adapter(self, tmp_path: Path) -> None:
        factory = SqliteVecFactory(vector_spec=MagicMock(dim=8))
        adapter = factory(db_url="s3://bucket/shard", local_dir=tmp_path / "shard")
        assert isinstance(adapter, SqliteVecAdapter)
        with patch("shardyfusion.sqlite_vec_adapter.put_bytes"):
            adapter.close()

    def test_reader_factory_creates_reader(self, tmp_path: Path) -> None:
        # Build a minimal DB
        shard_dir = tmp_path / "build"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=4),
        )
        adapter.checkpoint()
        db_bytes = (shard_dir / "shard.db").read_bytes()

        factory = SqliteVecReaderFactory()
        with patch("shardyfusion.sqlite_vec_adapter.get_bytes", return_value=db_bytes):
            reader = factory(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        assert isinstance(reader, SqliteVecShardReader)
        reader.close()
