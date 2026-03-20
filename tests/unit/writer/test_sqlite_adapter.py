"""Unit tests for SQLite KV and columnar adapters."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from shardyfusion.sqlite_adapter import (
    SqliteAdapter,
    SqliteAdapterError,
    SqliteColumnarAdapter,
    SqliteColumnarFactory,
    SqliteFactory,
    SqliteShardReader,
)
from shardyfusion.sqlite_schema import ColumnDef, SqliteSchema

# Mock put_bytes globally for all adapter tests (unit tests don't have S3).
_MOCK_PUT = patch("shardyfusion.sqlite_adapter.put_bytes")


# ---------------------------------------------------------------------------
# KV adapter
# ---------------------------------------------------------------------------


class TestSqliteAdapter:
    def test_write_batch_and_read_back(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        adapter = SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir)
        adapter.__enter__()

        pairs = [(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3")]
        adapter.write_batch(pairs)

        checkpoint_id = adapter.checkpoint()
        assert checkpoint_id is not None
        assert len(checkpoint_id) == 64  # SHA-256 hex

        # Verify data in SQLite file
        db_path = local_dir / "shard.db"
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT k, v FROM kv ORDER BY k").fetchall()
        conn.close()
        assert len(rows) == 3
        assert rows[0] == (b"key1", b"val1")

    def test_multiple_batches(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        with _MOCK_PUT:
            with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
                adapter.write_batch([(b"a", b"1"), (b"b", b"2")])
                adapter.write_batch([(b"c", b"3"), (b"d", b"4")])
                adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        count = conn.execute("SELECT count(*) FROM kv").fetchone()[0]
        conn.close()
        assert count == 4

    def test_empty_batch_is_noop(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        with _MOCK_PUT:
            with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
                adapter.write_batch([])
                cp = adapter.checkpoint()
                assert cp is not None  # still produces a valid hash

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        adapter = SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir)
        adapter.__enter__()
        adapter.checkpoint()
        with _MOCK_PUT:
            adapter.close()

        with pytest.raises(SqliteAdapterError, match="already closed"):
            adapter.write_batch([(b"x", b"y")])

    def test_upsert_on_duplicate_key(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        with _MOCK_PUT:
            with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
                adapter.write_batch([(b"key1", b"old")])
                adapter.write_batch([(b"key1", b"new")])
                adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        val = conn.execute("SELECT v FROM kv WHERE k = ?", (b"key1",)).fetchone()[0]
        conn.close()
        assert val == b"new"

    def test_flush_is_noop(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        with _MOCK_PUT:
            with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
                adapter.flush()  # should not raise

    def test_factory_creates_adapter(self, tmp_path: Path) -> None:
        factory = SqliteFactory()
        adapter = factory(db_url="s3://test/shard", local_dir=tmp_path / "shard")
        assert isinstance(adapter, SqliteAdapter)
        with _MOCK_PUT:
            adapter.close()

    def test_factory_is_picklable(self) -> None:
        import pickle

        factory = SqliteFactory(page_size=8192, cache_size_pages=-4000)
        restored = pickle.loads(pickle.dumps(factory))
        assert restored.page_size == 8192
        assert restored.cache_size_pages == -4000


# ---------------------------------------------------------------------------
# Columnar adapter
# ---------------------------------------------------------------------------


class TestSqliteColumnarAdapter:
    @pytest.fixture()
    def schema(self) -> SqliteSchema:
        return SqliteSchema(
            table_name="users",
            columns=(
                ColumnDef(name="user_id", type="INTEGER", primary_key=True),
                ColumnDef(name="name", type="TEXT"),
                ColumnDef(name="age", type="INTEGER"),
            ),
            indexes=(("name",),),
        )

    def test_write_rows_and_query(self, tmp_path: Path, schema: SqliteSchema) -> None:
        local_dir = tmp_path / "shard"
        adapter = SqliteColumnarAdapter(
            db_url="s3://test/shard",
            local_dir=local_dir,
            schema=schema,
        )
        adapter.__enter__()
        adapter.write_rows([(1, "Alice", 30), (2, "Bob", 25)])
        adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        rows = conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()
        conn.close()
        assert rows == [(1, "Alice", 30), (2, "Bob", 25)]

    def test_write_dicts(self, tmp_path: Path, schema: SqliteSchema) -> None:
        local_dir = tmp_path / "shard"
        with _MOCK_PUT:
            with SqliteColumnarAdapter(
                db_url="s3://test/shard", local_dir=local_dir, schema=schema
            ) as adapter:
                adapter.write_dicts([
                    {"user_id": 1, "name": "Alice", "age": 30},
                    {"user_id": 2, "name": "Bob", "age": 25},
                ])
                adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        rows = conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()
        conn.close()
        assert rows == [(1, "Alice", 30), (2, "Bob", 25)]

    def test_write_batch_from_json(self, tmp_path: Path, schema: SqliteSchema) -> None:
        local_dir = tmp_path / "shard"
        with _MOCK_PUT:
            with SqliteColumnarAdapter(
                db_url="s3://test/shard", local_dir=local_dir, schema=schema
            ) as adapter:
                row_json = json.dumps({"user_id": 1, "name": "Alice", "age": 30}).encode()
                adapter.write_batch([(b"\x00\x01", row_json)])
                adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        rows = conn.execute("SELECT * FROM users").fetchall()
        conn.close()
        assert rows == [(1, "Alice", 30)]

    def test_indexes_created(self, tmp_path: Path, schema: SqliteSchema) -> None:
        local_dir = tmp_path / "shard"
        with _MOCK_PUT:
            with SqliteColumnarAdapter(
                db_url="s3://test/shard", local_dir=local_dir, schema=schema
            ) as adapter:
                adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        conn.close()
        index_names = [i[0] for i in indexes]
        assert any("idx_users" in n for n in index_names)

    def test_columnar_factory(self, tmp_path: Path, schema: SqliteSchema) -> None:
        factory = SqliteColumnarFactory(schema=schema)
        adapter = factory(db_url="s3://test/shard", local_dir=tmp_path / "shard")
        assert isinstance(adapter, SqliteColumnarAdapter)
        with _MOCK_PUT:
            adapter.close()

    def test_columnar_factory_is_picklable(self) -> None:
        import pickle

        schema = SqliteSchema(
            table_name="t",
            columns=(ColumnDef(name="id", type="INTEGER", primary_key=True),),
        )
        factory = SqliteColumnarFactory(schema=schema)
        restored = pickle.loads(pickle.dumps(factory))
        assert restored.schema.table_name == "t"


# ---------------------------------------------------------------------------
# Local reader (no S3)
# ---------------------------------------------------------------------------


class TestSqliteShardReaderLocal:
    """Tests SqliteShardReader against a pre-built local SQLite DB."""

    @pytest.fixture()
    def shard_dir(self, tmp_path: Path) -> Path:
        """Create a shard.db with some kv data."""
        shard_dir = tmp_path / "shard"
        shard_dir.mkdir()
        db_path = shard_dir / "shard.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
        conn.executemany(
            "INSERT INTO kv (k, v) VALUES (?, ?)",
            [(b"key1", b"val1"), (b"key2", b"val2")],
        )
        conn.commit()
        conn.close()
        return shard_dir

    def test_get_existing_key(self, shard_dir: Path) -> None:
        reader = SqliteShardReader.__new__(SqliteShardReader)
        reader._db_url = "s3://test/shard"
        reader._db_path = shard_dir / "shard.db"
        conn = sqlite3.connect(
            f"file:{reader._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        reader._conn = conn

        assert reader.get(b"key1") == b"val1"
        assert reader.get(b"key2") == b"val2"

    def test_get_missing_key(self, shard_dir: Path) -> None:
        reader = SqliteShardReader.__new__(SqliteShardReader)
        reader._db_url = "s3://test/shard"
        reader._db_path = shard_dir / "shard.db"
        conn = sqlite3.connect(
            f"file:{reader._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        reader._conn = conn

        assert reader.get(b"missing") is None

    def test_query_method(self, shard_dir: Path) -> None:
        reader = SqliteShardReader.__new__(SqliteShardReader)
        reader._db_url = "s3://test/shard"
        reader._db_path = shard_dir / "shard.db"
        conn = sqlite3.connect(
            f"file:{reader._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        reader._conn = conn

        rows = reader.query("SELECT k, v FROM kv ORDER BY k")
        assert len(rows) == 2
        assert dict(rows[0])["k"] == b"key1"

    def test_close_and_reuse_raises(self, shard_dir: Path) -> None:
        reader = SqliteShardReader.__new__(SqliteShardReader)
        reader._db_url = "s3://test/shard"
        reader._db_path = shard_dir / "shard.db"
        conn = sqlite3.connect(
            f"file:{reader._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        reader._conn = conn

        reader.close()
        with pytest.raises(SqliteAdapterError, match="already closed"):
            reader.get(b"key1")
