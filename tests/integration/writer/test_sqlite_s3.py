"""Integration tests for SQLite adapter with S3 (moto)."""

import json
from pathlib import Path

import pytest

moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")

from moto import mock_aws  # noqa: E402

from shardyfusion.sqlite_adapter import (  # noqa: E402
    SqliteAdapter,
    SqliteColumnarAdapter,
    SqliteReaderFactory,
    SqliteShardReader,
)
from shardyfusion.sqlite_schema import ColumnDef, SqliteSchema  # noqa: E402

_BUCKET = "test-bucket"
_PREFIX = f"s3://{_BUCKET}/test-prefix"


@pytest.fixture()
def s3_env(monkeypatch: pytest.MonkeyPatch):
    """Set up moto S3 with a test bucket."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


class TestSqliteKvRoundTrip:
    """Write via SqliteAdapter → upload to S3 → read via SqliteShardReader."""

    def test_write_upload_and_read(self, tmp_path: Path, s3_env) -> None:
        db_url = f"{_PREFIX}/shards/run_id=test/db=00000/attempt=00"
        write_dir = tmp_path / "write"
        read_dir = tmp_path / "read"

        # Write
        with SqliteAdapter(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch(
                [
                    (b"key1", b"val1"),
                    (b"key2", b"val2"),
                    (b"key3", b"val3"),
                ]
            )
            checkpoint_id = adapter.checkpoint()

        assert checkpoint_id is not None

        # Read
        reader = SqliteShardReader(
            db_url=db_url,
            local_dir=read_dir,
            checkpoint_id=checkpoint_id,
        )
        assert reader.get(b"key1") == b"val1"
        assert reader.get(b"key2") == b"val2"
        assert reader.get(b"key3") == b"val3"
        assert reader.get(b"missing") is None
        reader.close()

    def test_large_batch(self, tmp_path: Path, s3_env) -> None:
        db_url = f"{_PREFIX}/shards/run_id=test/db=00001/attempt=00"
        write_dir = tmp_path / "write"
        read_dir = tmp_path / "read"

        pairs = [(i.to_bytes(8, "big"), f"value_{i}".encode()) for i in range(10_000)]

        with SqliteAdapter(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch(pairs)
            adapter.checkpoint()

        reader = SqliteShardReader(
            db_url=db_url, local_dir=read_dir, checkpoint_id=None
        )
        assert reader.get((0).to_bytes(8, "big")) == b"value_0"
        assert reader.get((9999).to_bytes(8, "big")) == b"value_9999"
        reader.close()


class TestSqliteColumnarRoundTrip:
    """Write via SqliteColumnarAdapter → S3 → read via SQL."""

    @pytest.fixture()
    def schema(self) -> SqliteSchema:
        return SqliteSchema(
            table_name="users",
            columns=(
                ColumnDef(name="user_id", type="INTEGER", primary_key=True),
                ColumnDef(name="name", type="TEXT"),
                ColumnDef(name="email", type="TEXT"),
                ColumnDef(name="age", type="INTEGER"),
            ),
            indexes=(("email",),),
        )

    def test_columnar_write_and_sql_read(
        self, tmp_path: Path, s3_env, schema: SqliteSchema
    ) -> None:
        db_url = f"{_PREFIX}/shards/run_id=test/db=00000/attempt=00"
        write_dir = tmp_path / "write"
        read_dir = tmp_path / "read"

        # Write rows
        with SqliteColumnarAdapter(
            db_url=db_url, local_dir=write_dir, schema=schema
        ) as adapter:
            adapter.write_rows(
                [
                    (1, "Alice", "alice@example.com", 30),
                    (2, "Bob", "bob@example.com", 25),
                    (3, "Charlie", "charlie@example.com", 35),
                ]
            )
            adapter.checkpoint()

        # Read via SQL
        reader = SqliteShardReader(
            db_url=db_url, local_dir=read_dir, checkpoint_id=None
        )
        rows = reader.query(
            "SELECT name, age FROM users WHERE age > ? ORDER BY name", (28,)
        )
        assert len(rows) == 2
        assert dict(rows[0])["name"] == "Alice"
        assert dict(rows[1])["name"] == "Charlie"
        reader.close()

    def test_columnar_with_write_batch_json(
        self, tmp_path: Path, s3_env, schema: SqliteSchema
    ) -> None:
        """Test DbAdapter-compatible path: JSON-encoded values."""
        db_url = f"{_PREFIX}/shards/run_id=test/db=00000/attempt=00"
        write_dir = tmp_path / "write"
        read_dir = tmp_path / "read"

        row = {"user_id": 42, "name": "Test", "email": "t@t.com", "age": 20}
        key_bytes = (42).to_bytes(8, "big")
        value_bytes = json.dumps(row).encode()

        with SqliteColumnarAdapter(
            db_url=db_url, local_dir=write_dir, schema=schema
        ) as adapter:
            adapter.write_batch([(key_bytes, value_bytes)])
            adapter.checkpoint()

        reader = SqliteShardReader(
            db_url=db_url, local_dir=read_dir, checkpoint_id=None
        )
        rows = reader.query("SELECT * FROM users WHERE user_id = 42")
        assert len(rows) == 1
        assert dict(rows[0])["name"] == "Test"
        reader.close()


class TestSqliteReaderFactory:
    def test_factory_creates_working_reader(self, tmp_path: Path, s3_env) -> None:
        db_url = f"{_PREFIX}/shards/run_id=test/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteAdapter(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.checkpoint()

        factory = SqliteReaderFactory()
        reader = factory(
            db_url=db_url,
            local_dir=tmp_path / "read",
            checkpoint_id=None,
        )
        assert reader.get(b"k") == b"v"
        reader.close()
