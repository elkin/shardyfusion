"""Tests for SqliteVecAdapter — unified KV + vector SQLite shard lifecycle."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None  # type: ignore[assignment]

from shardyfusion.errors import ConfigValidationError  # noqa: E402
from shardyfusion.sqlite_vec_adapter import (  # noqa: E402
    SqliteVecAdapter,
    SqliteVecAdapterError,
    SqliteVecFactory,
    SqliteVecReaderFactory,
    SqliteVecShardReader,
    _is_cached_snapshot_current,
)
from shardyfusion.vector.types import SearchResult  # noqa: E402


def _is_sqlite_vec_available() -> bool:
    """Check if sqlite-vec is available and loadable."""
    if sqlite_vec is None:
        return False
    try:
        conn = sqlite3.connect(":memory:")
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (sqlite3.OperationalError, AttributeError):
        return False


pytestmark = pytest.mark.vector_sqlite


def _create_sqlite3_conn(*args: Any, **kwargs: Any) -> sqlite3.Connection:
    """Create a sqlite3 connection with extension loading enabled."""
    conn = sqlite3.connect(*args, **kwargs)
    if hasattr(conn, "enable_load_extension"):
        conn.enable_load_extension(True)
    return conn


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
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend"):
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            )
            adapter.close()

        with pytest.raises(SqliteVecAdapterError, match="already closed"):
            adapter.write_batch([(b"k", b"v")])

    def test_close_uploads_to_s3(self, tmp_path: Path) -> None:
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.put = MagicMock()
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            )
            adapter.write_batch([(b"k1", b"v1")])
            adapter.close()

        instance.put.assert_called_once()
        call_args = instance.put.call_args
        assert call_args[0][0] == "s3://bucket/shard/shard.db"

    def test_close_idempotent(self, tmp_path: Path) -> None:
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.put = MagicMock()
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            )
            adapter.close()
            adapter.close()

        # Only uploaded once
        assert instance.put.call_count == 1

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend"):
            with SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=MagicMock(dim=4),
            ) as adapter:
                adapter.write_batch([(b"k", b"v")])
            assert adapter._closed

    def test_dot_product_metric_rejected(self, tmp_path: Path) -> None:
        vector_spec = MagicMock(dim=4)
        vector_spec.metric = "dot_product"

        with pytest.raises(ConfigValidationError, match="does not support dot_product"):
            SqliteVecAdapter(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "shard",
                vector_spec=vector_spec,
            )


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

    def test_integer_ids_have_typed_id_map_rows(self, tmp_path: Path) -> None:
        db_path = self._build_db(tmp_path, np.array([10, 20, 30]))

        conn = _create_sqlite3_conn(str(db_path))
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT internal_id, original_id FROM vec_id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()
        assert len(rows) == 3
        assert json.loads(rows[0][1]) == {"v": 10, "t": "int"}
        assert json.loads(rows[1][1]) == {"v": 20, "t": "int"}
        assert json.loads(rows[2][1]) == {"v": 30, "t": "int"}

    def test_string_ids_populate_id_map(self, tmp_path: Path) -> None:
        db_path = self._build_db(tmp_path, np.array(["a", "b", "c"], dtype=object))

        conn = _create_sqlite3_conn(str(db_path))
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT internal_id, original_id FROM vec_id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()
        assert len(rows) == 3
        assert rows[0][0] == 0
        assert json.loads(rows[0][1]) == {"v": "a", "t": "str"}
        assert rows[1][0] == 1
        assert json.loads(rows[1][1]) == {"v": "b", "t": "str"}
        assert rows[2][0] == 2
        assert json.loads(rows[2][1]) == {"v": "c", "t": "str"}

    def test_synthetic_ids_advance_next_vec_id(self, tmp_path: Path) -> None:
        """After writing 2 IDs, _next_vec_id should be 2 (synthetic monotonic)."""
        shard_dir = tmp_path / "shard"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=4),
        )
        vecs = np.random.randn(2, 4).astype(np.float32)
        adapter.write_vector_batch(np.array([10, 20]), vecs)
        assert adapter._next_vec_id == 2

        # Now write strings — should start at 2, no collision
        adapter.write_vector_batch(
            np.array(["x", "y"], dtype=object),
            np.random.randn(2, 4).astype(np.float32),
        )
        adapter.checkpoint()

        conn = _create_sqlite3_conn(str(shard_dir / "shard.db"))
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT internal_id, original_id FROM vec_id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()
        # All 4 entries with synthetic rowids 0-3
        assert len(rows) == 4
        assert json.loads(rows[0][1]) == {"v": 10, "t": "int"}
        assert json.loads(rows[1][1]) == {"v": 20, "t": "int"}
        assert json.loads(rows[2][1]) == {"v": "x", "t": "str"}
        assert json.loads(rows[3][1]) == {"v": "y", "t": "str"}

    def test_payloads_stored(self, tmp_path: Path) -> None:
        db_path = self._build_db(
            tmp_path,
            np.array([1, 2]),
            payloads=[{"color": "red"}, {"color": "blue"}],
        )

        conn = _create_sqlite3_conn(str(db_path))
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

        conn = _create_sqlite3_conn(str(db_path))
        sqlite_vec.load(conn)
        rows = conn.execute("SELECT rowid FROM vec_payloads").fetchall()
        conn.close()
        # Only synthetic rowid=0 (first entry) should have a payload
        assert len(rows) == 1
        assert rows[0][0] == 0

    def test_mixed_ids_allocate_synthetic_rowids(self, tmp_path: Path) -> None:
        """All IDs (int and string) get synthetic monotonic rowids."""
        shard_dir = tmp_path / "shard"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=4),
        )
        ids = np.array([10, "doc_a", 2, "doc_b"], dtype=object)
        vecs = np.random.randn(4, 4).astype(np.float32)
        adapter.write_vector_batch(ids, vecs)
        adapter.checkpoint()

        conn = _create_sqlite3_conn(str(shard_dir / "shard.db"))
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT internal_id, original_id FROM vec_id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()

        assert len(rows) == 4
        assert json.loads(rows[0][1]) == {"v": 10, "t": "int"}
        assert json.loads(rows[1][1]) == {"v": "doc_a", "t": "str"}
        assert json.loads(rows[2][1]) == {"v": 2, "t": "int"}
        assert json.loads(rows[3][1]) == {"v": "doc_b", "t": "str"}

    def test_mixed_ids_payloads_persist_on_synthetic_rowids(
        self, tmp_path: Path
    ) -> None:
        shard_dir = tmp_path / "shard"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=4),
        )
        ids = np.array([5, "doc_a", 7, "doc_b"], dtype=object)
        vecs = np.random.randn(4, 4).astype(np.float32)
        payloads: list[dict[str, Any] | None] = [
            {"kind": "int"},
            {"kind": "string_a"},
            None,
            {"kind": "string_b"},
        ]
        adapter.write_vector_batch(ids, vecs, payloads)
        adapter.checkpoint()

        conn = _create_sqlite3_conn(str(shard_dir / "shard.db"))
        sqlite_vec.load(conn)
        id_map_rows = conn.execute(
            "SELECT internal_id, original_id FROM vec_id_map ORDER BY internal_id"
        ).fetchall()
        payload_rows = conn.execute(
            "SELECT rowid, payload FROM vec_payloads ORDER BY rowid"
        ).fetchall()
        conn.close()

        # All 4 IDs get synthetic rowids 0-3
        assert len(id_map_rows) == 4
        assert json.loads(id_map_rows[0][1]) == {"v": 5, "t": "int"}
        assert json.loads(id_map_rows[1][1]) == {"v": "doc_a", "t": "str"}
        assert json.loads(id_map_rows[2][1]) == {"v": 7, "t": "int"}
        assert json.loads(id_map_rows[3][1]) == {"v": "doc_b", "t": "str"}
        # Payloads at synthetic rowids: 0, 1, 3 (index 2 has None payload)
        assert [(rowid, json.loads(payload)) for rowid, payload in payload_rows] == [
            (0, {"kind": "int"}),
            (1, {"kind": "string_a"}),
            (3, {"kind": "string_b"}),
        ]

    def test_mixed_ids_across_batches_checkpoint_is_reproducible(
        self, tmp_path: Path
    ) -> None:
        def build_checkpoint(shard_name: str) -> tuple[str, list[tuple[int, str]]]:
            shard_dir = tmp_path / shard_name
            adapter = SqliteVecAdapter(
                db_url=f"s3://bucket/{shard_name}",
                local_dir=shard_dir,
                vector_spec=MagicMock(dim=4),
            )

            batch1_ids = np.array([4, "doc_a", 8], dtype=object)
            batch1_vecs = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                dtype=np.float32,
            )
            batch2_ids = np.array(["doc_b", 6, "doc_c"], dtype=object)
            batch2_vecs = np.array(
                [[0.8, 0.2, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.7, 0.3, 0.0, 0.0]],
                dtype=np.float32,
            )

            adapter.write_vector_batch(batch1_ids, batch1_vecs)
            adapter.write_vector_batch(batch2_ids, batch2_vecs)
            checkpoint = adapter.checkpoint()

            conn = _create_sqlite3_conn(str(shard_dir / "shard.db"))
            sqlite_vec.load(conn)
            id_map_rows = conn.execute(
                "SELECT internal_id, original_id FROM vec_id_map ORDER BY internal_id"
            ).fetchall()
            conn.close()
            return checkpoint, id_map_rows

        checkpoint_a, id_map_a = build_checkpoint("shard_a")
        checkpoint_b, id_map_b = build_checkpoint("shard_b")

        assert checkpoint_a == checkpoint_b
        assert id_map_a == id_map_b
        # All 6 entries with synthetic rowids 0-5
        assert len(id_map_a) == 6
        decoded = [(r[0], json.loads(r[1])) for r in id_map_a]
        assert decoded == [
            (0, {"v": 4, "t": "int"}),
            (1, {"v": "doc_a", "t": "str"}),
            (2, {"v": 8, "t": "int"}),
            (3, {"v": "doc_b", "t": "str"}),
            (4, {"v": 6, "t": "int"}),
            (5, {"v": "doc_c", "t": "str"}),
        ]

    def test_write_vector_batch_uses_executemany_for_all_staged_tables(
        self, tmp_path: Path
    ) -> None:
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=tmp_path / "shard",
            vector_spec=MagicMock(dim=4),
        )

        real_conn = adapter._conn
        assert real_conn is not None
        real_conn.close()

        conn_mock = MagicMock()
        adapter._conn = conn_mock

        ids = np.array([5, "doc_a", 7], dtype=object)
        vecs = np.random.randn(3, 4).astype(np.float32)
        payloads: list[dict[str, Any] | None] = [
            {"kind": "int"},
            {"kind": "string"},
            None,
        ]

        adapter.write_vector_batch(ids, vecs, payloads)

        sql_calls = [call.args[0] for call in conn_mock.executemany.call_args_list]
        assert any("vec_id_map" in sql for sql in sql_calls)
        assert any("vec_index" in sql for sql in sql_calls)
        assert any("vec_payloads" in sql for sql in sql_calls)
        conn_mock.execute.assert_not_called()


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
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        assert reader.get(b"k1") == b"v1"
        assert reader.get(b"missing") is None
        reader.close()

    def test_vector_search(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
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

        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = reader.search(query, top_k=2)

        assert results[0].id == "doc_a"
        assert results[1].id == "doc_b"
        reader.close()

    def test_search_with_mixed_ids_unchanged_correctness(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "build"
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/shard",
            local_dir=shard_dir,
            vector_spec=MagicMock(dim=4),
        )
        ids = np.array([42, "doc_a", 11], dtype=object)
        vecs = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],  # int id
                [1.0, 0.0, 0.0, 0.0],  # string id: nearest to query
                [0.0, 0.0, 1.0, 0.0],  # int id
            ],
            dtype=np.float32,
        )
        payloads = [{"kind": "int"}, {"kind": "string"}, {"kind": "int2"}]
        adapter.write_vector_batch(ids, vecs, payloads)
        adapter.checkpoint()
        db_bytes = (shard_dir / "shard.db").read_bytes()

        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )

        query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        results = reader.search(query, top_k=3)

        assert [r.id for r in results] == ["doc_a", 42, 11]
        assert results[0].payload == {"kind": "string"}
        reader.close()

    def test_search_after_close_raises(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        reader.close()
        with pytest.raises(SqliteVecAdapterError, match="closed"):
            reader.search(np.zeros(4, dtype=np.float32), top_k=1)

    def test_get_after_close_raises(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        reader.close()
        with pytest.raises(SqliteVecAdapterError, match="closed"):
            reader.get(b"k1")

    def test_context_manager(self, tmp_path: Path) -> None:
        db_bytes = self._build_shard(tmp_path)
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
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
        # DB file must also exist for the cache to be considered valid
        (tmp_path / "shard.db").write_bytes(b"fake-db")
        assert _is_cached_snapshot_current(path, "s3://b/k", "ckpt1")

    def test_missing_db_file(self, tmp_path: Path) -> None:
        """Identity file exists but DB file is missing — cache is invalid."""
        path = tmp_path / "identity.json"
        path.write_text(json.dumps({"db_url": "s3://b/k", "checkpoint_id": "ckpt1"}))
        assert not _is_cached_snapshot_current(path, "s3://b/k", "ckpt1")

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
        conn = _create_sqlite3_conn(str(db_path))
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

        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock()
            reader = SqliteVecShardReader(
                db_url="s3://bucket/shard",
                local_dir=read_dir,
                checkpoint_id="abc",
            )
            instance.get.assert_not_called()
        reader.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestSqliteVecFactory:
    def test_factory_creates_adapter(self, tmp_path: Path) -> None:
        factory = SqliteVecFactory(vector_spec=MagicMock(dim=8))
        adapter = factory(db_url="s3://bucket/shard", local_dir=tmp_path / "shard")
        assert isinstance(adapter, SqliteVecAdapter)
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend"):
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
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
            reader = factory(
                db_url="s3://bucket/shard",
                local_dir=tmp_path / "read",
            )
        assert isinstance(reader, SqliteVecShardReader)
        reader.close()


class TestRunVecSearchStrictErrors:
    """``_run_vec_search`` must propagate any error from the ``vec_id_map``
    lookup. The previous "missing-table tolerance" was removed because every
    snapshot we produce writes ``vec_id_map``; a missing table indicates a
    malformed snapshot, not graceful degradation.
    """

    def _run_with_id_map_error(self, exc: BaseException) -> None:
        from shardyfusion.sqlite_vec_adapter import _run_vec_search

        class _FakeConn:
            def execute(self, sql: str, params: Any = ()) -> Any:
                if "vec_index" in sql:
                    return iter([(7, 0.5)])
                if "vec_id_map" in sql:
                    raise exc
                raise AssertionError(f"Unexpected SQL: {sql}")

        _run_vec_search(_FakeConn(), np.array([1.0, 2.0], dtype=np.float32), top_k=1)

    def test_missing_vec_id_map_table_propagates(self) -> None:
        with pytest.raises(sqlite3.OperationalError, match="vec_id_map"):
            self._run_with_id_map_error(
                sqlite3.OperationalError("no such table: vec_id_map")
            )

    def test_unrelated_exception_propagates(self) -> None:
        with pytest.raises(RuntimeError, match="kaboom"):
            self._run_with_id_map_error(RuntimeError("kaboom"))
