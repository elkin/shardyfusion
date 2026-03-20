"""Unit tests for ShardedSqlReader fan-out SQL queries."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shardyfusion.errors import ReaderStateError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.sqlite_query import ShardedSqlReader, _row_to_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DB_FILENAME = "shard.db"


def _make_shard_db(path: Path, rows: list[tuple]) -> None:
    """Create a SQLite shard with a 'data' table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE data (user_id INTEGER PRIMARY KEY, name TEXT, age INTEGER) WITHOUT ROWID"
    )
    conn.executemany("INSERT INTO data VALUES (?, ?, ?)", rows)
    # Also create the kv table for k/v protocol compat
    conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
    conn.commit()
    conn.close()


def _build_manifest(num_shards: int, s3_prefix: str) -> ParsedManifest:
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"{s3_prefix}/shards/run_id=test/db={i:05d}/attempt=00",
            attempt=0,
            row_count=10,
            writer_info=WriterInfo(),
        )
        for i in range(num_shards)
    ]
    build = RequiredBuildMeta(
        run_id="test",
        num_dbs=num_shards,
        s3_prefix=s3_prefix,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        key_encoding=KeyEncoding.U64BE,
        key_col="user_id",
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
        created_at=datetime.now(tz=UTC),
    )
    return ParsedManifest(required_build=build, shards=shards, custom={})


def _make_manifest_ref() -> ManifestRef:
    return ManifestRef(
        ref="s3://bucket/manifests/test/manifest",
        run_id="test",
        published_at=datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShardedSqlReaderFanOut:
    """Test SQL fan-out using pre-built local SQLite databases."""

    @pytest.fixture()
    def reader(self, tmp_path: Path) -> ShardedSqlReader:
        """Build a 3-shard reader with pre-built local SQLite databases."""
        num_shards = 3
        s3_prefix = "s3://bucket/prefix"
        manifest = _build_manifest(num_shards, s3_prefix)
        ref = _make_manifest_ref()

        # Pre-create local shard DBs
        shard_data = {
            0: [(1, "Alice", 30), (4, "Diana", 28)],
            1: [(2, "Bob", 25), (5, "Eve", 35)],
            2: [(3, "Charlie", 22), (6, "Frank", 40)],
        }
        local_root = str(tmp_path / "reader")
        for sid, rows in shard_data.items():
            shard_dir = Path(local_root) / f"shard={sid:05d}"
            _make_shard_db(shard_dir / _DB_FILENAME, rows)

        # Create reader by bypassing S3 download (files already exist)
        mock_store = MagicMock()
        mock_store.load_current.return_value = ref
        mock_store.load_manifest.return_value = manifest

        return ShardedSqlReader(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=mock_store,
        )

    def test_query_shard(self, reader: ShardedSqlReader) -> None:
        rows = reader.query_shard(0, "SELECT * FROM data ORDER BY user_id")
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Diana"

    def test_query_shard_with_params(self, reader: ShardedSqlReader) -> None:
        rows = reader.query_shard(1, "SELECT * FROM data WHERE age > ?", (30,))
        assert len(rows) == 1
        assert rows[0]["name"] == "Eve"

    def test_query_empty_shard(self, reader: ShardedSqlReader) -> None:
        # Shard ID beyond num_shards or empty — returns empty list
        rows = reader.query_shard(999, "SELECT * FROM data")
        assert rows == []

    def test_query_all(self, reader: ShardedSqlReader) -> None:
        rows = reader.query_all("SELECT * FROM data ORDER BY user_id")
        assert len(rows) == 6
        names = sorted(r["name"] for r in rows)
        assert names == ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

    def test_query_all_with_limit(self, reader: ShardedSqlReader) -> None:
        rows = reader.query_all("SELECT * FROM data", limit=3)
        assert len(rows) == 3

    def test_query_all_with_where(self, reader: ShardedSqlReader) -> None:
        rows = reader.query_all("SELECT * FROM data WHERE age >= ?", (30,))
        names = sorted(r["name"] for r in rows)
        assert names == ["Alice", "Eve", "Frank"]

    def test_query_shards_subset(self, reader: ShardedSqlReader) -> None:
        rows = reader.query_shards([0, 2], "SELECT name FROM data ORDER BY name")
        names = sorted(r["name"] for r in rows)
        assert names == ["Alice", "Charlie", "Diana", "Frank"]

    def test_query_all_concurrent(self, reader: ShardedSqlReader) -> None:
        rows = reader.query_all_concurrent(
            "SELECT * FROM data WHERE age < ?", (30,), max_workers=2
        )
        names = sorted(r["name"] for r in rows)
        assert names == ["Bob", "Charlie", "Diana"]

    def test_num_shards(self, reader: ShardedSqlReader) -> None:
        assert reader.num_shards == 3

    def test_shard_ids(self, reader: ShardedSqlReader) -> None:
        assert reader.shard_ids == [0, 1, 2]

    def test_context_manager(self, reader: ShardedSqlReader) -> None:
        with reader:
            reader.query_all("SELECT count(*) as cnt FROM data")
        # After close, accessing should raise
        with pytest.raises(ReaderStateError):
            reader.query_all("SELECT * FROM data")

    def test_schema_from_manifest(self, tmp_path: Path) -> None:
        """When manifest has sqlite_schema custom field, reader picks it up."""
        num_shards = 1
        s3_prefix = "s3://bucket/prefix"
        manifest = _build_manifest(num_shards, s3_prefix)
        manifest.custom["sqlite_schema"] = {
            "table_name": "data",
            "columns": [
                {"name": "user_id", "type": "INTEGER", "primary_key": True},
                {"name": "name", "type": "TEXT"},
            ],
            "indexes": [],
        }
        ref = _make_manifest_ref()

        local_root = str(tmp_path / "reader")
        shard_dir = Path(local_root) / "shard=00000"
        _make_shard_db(shard_dir / _DB_FILENAME, [(1, "Alice", 30)])

        mock_store = MagicMock()
        mock_store.load_current.return_value = ref
        mock_store.load_manifest.return_value = manifest

        reader = ShardedSqlReader(
            s3_prefix=s3_prefix,
            local_root=local_root,
            manifest_store=mock_store,
        )
        assert reader.schema is not None
        assert reader.schema.table_name == "data"
        reader.close()


class TestRowToDict:
    def test_converts_sqlite_row(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")
        row = conn.execute("SELECT * FROM t").fetchone()
        conn.close()

        d = _row_to_dict(row)
        assert d == {"a": 1, "b": "hello"}
