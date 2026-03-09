"""Unit tests for database-backed ManifestStore implementations.

Uses SQLite in-memory databases — no external dependencies required.
"""

import sqlite3
from collections.abc import Iterator

import pytest

from shardyfusion.db_manifest_store import SqliteManifestStore
from shardyfusion.errors import (
    ConfigValidationError,
    ManifestParseError,
    ManifestStoreError,
)
from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

# Each test gets a unique DB name to avoid cross-test interference.
_db_counter = 0


@pytest.fixture
def sqlite_store() -> Iterator[SqliteManifestStore]:
    """Create a SqliteManifestStore backed by a unique in-memory database.

    An anchor connection is held open for the test duration so the
    in-memory database persists across the store's open/close cycles.
    """
    global _db_counter
    _db_counter += 1
    db_uri = f"file:test_db_{_db_counter}?mode=memory&cache=shared"

    anchor = sqlite3.connect(db_uri, uri=True)

    store = SqliteManifestStore(lambda: sqlite3.connect(db_uri, uri=True))
    yield store

    anchor.close()


def _make_required_build(
    *, run_id: str = "run-1", num_dbs: int = 2
) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id=run_id,
        created_at="2025-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="_key",
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
        key_encoding=KeyEncoding.U64BE,
    )


def _make_shards(num_dbs: int = 2) -> list[RequiredShardMeta]:
    return [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/db={i:05d}",
            attempt=0,
            row_count=100,
        )
        for i in range(num_dbs)
    ]


class TestSqliteManifestStorePublish:
    def test_publish_returns_run_id(self, sqlite_store: SqliteManifestStore) -> None:
        ref = sqlite_store.publish(
            run_id="run-1",
            required_build=_make_required_build(),
            shards=_make_shards(),
            custom={"source": "test"},
        )
        assert ref == "run-1"

    def test_publish_and_load_roundtrip(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        sqlite_store.publish(
            run_id="run-1",
            required_build=_make_required_build(),
            shards=_make_shards(),
            custom={"source": "test"},
        )

        manifest = sqlite_store.load_manifest("run-1")
        assert manifest.required_build.run_id == "run-1"
        assert manifest.required_build.num_dbs == 2
        assert len(manifest.shards) == 2
        assert manifest.shards[0].db_id == 0
        assert manifest.shards[1].db_id == 1
        assert manifest.custom == {"source": "test"}

    def test_publish_upsert_overwrites(self, sqlite_store: SqliteManifestStore) -> None:
        sqlite_store.publish(
            run_id="run-1",
            required_build=_make_required_build(),
            shards=_make_shards(),
            custom={"version": 1},
        )
        sqlite_store.publish(
            run_id="run-1",
            required_build=_make_required_build(),
            shards=_make_shards(),
            custom={"version": 2},
        )

        manifest = sqlite_store.load_manifest("run-1")
        assert manifest.custom == {"version": 2}


class TestSqliteManifestStoreLoadCurrent:
    def test_load_current_returns_none_when_empty(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        assert sqlite_store.load_current() is None

    def test_load_current_returns_published(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        sqlite_store.publish(
            run_id="run-1",
            required_build=_make_required_build(run_id="run-1"),
            shards=_make_shards(),
            custom={},
        )

        current = sqlite_store.load_current()
        assert current is not None
        assert current.run_id == "run-1"
        assert current.manifest_ref == "run-1"
        assert current.manifest_content_type == "application/json"


class TestSqliteManifestStoreLoadManifest:
    def test_load_manifest_not_found_raises(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        with pytest.raises(ManifestParseError, match="not found"):
            sqlite_store.load_manifest("nonexistent")

    def test_load_manifest_preserves_sharding(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        sqlite_store.publish(
            run_id="run-1",
            required_build=_make_required_build(),
            shards=_make_shards(),
            custom={},
        )

        manifest = sqlite_store.load_manifest("run-1")
        assert manifest.required_build.sharding.strategy == ShardingStrategy.HASH
        assert manifest.required_build.key_encoding == KeyEncoding.U64BE


class TestSqliteManifestStoreEnsureTable:
    def test_ensure_table_false_skips_creation(self) -> None:
        """When ensure_table=False, no DDL is executed at construction."""
        call_count = 0

        def counting_factory() -> sqlite3.Connection:
            nonlocal call_count
            call_count += 1
            return sqlite3.connect(":memory:")

        SqliteManifestStore(counting_factory, ensure_table=False)
        assert call_count == 0

    def test_ensure_table_failure_raises_config_error(self) -> None:
        def bad_factory() -> sqlite3.Connection:
            raise RuntimeError("connection failed")

        with pytest.raises(ConfigValidationError, match="connection failed"):
            SqliteManifestStore(bad_factory)


class TestSqliteManifestStoreConnectionFailures:
    def test_publish_connection_failure_raises_store_error(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        sqlite_store._connection_factory = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("connection lost")
        )

        with pytest.raises(ManifestStoreError, match="connection lost"):
            sqlite_store.publish(
                run_id="run-1",
                required_build=_make_required_build(),
                shards=_make_shards(),
                custom={},
            )

    def test_load_current_connection_failure_raises_store_error(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        sqlite_store._connection_factory = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("connection lost")
        )

        with pytest.raises(ManifestStoreError, match="connection lost"):
            sqlite_store.load_current()

    def test_load_manifest_connection_failure_raises_store_error(
        self, sqlite_store: SqliteManifestStore
    ) -> None:
        sqlite_store._connection_factory = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("connection lost")
        )

        with pytest.raises(ManifestStoreError, match="connection lost"):
            sqlite_store.load_manifest("run-1")


class TestSqliteManifestStoreCustomTable:
    def test_custom_table_name(self) -> None:
        global _db_counter
        _db_counter += 1
        db_uri = f"file:test_db_{_db_counter}?mode=memory&cache=shared"

        anchor = sqlite3.connect(db_uri, uri=True)
        try:
            store = SqliteManifestStore(
                lambda: sqlite3.connect(db_uri, uri=True),
                table_name="my_manifests",
            )
            store.publish(
                run_id="run-1",
                required_build=_make_required_build(),
                shards=_make_shards(),
                custom={},
            )

            conn = sqlite3.connect(db_uri, uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT run_id FROM my_manifests")
            rows = cursor.fetchall()
            conn.close()
            assert len(rows) == 1
            assert rows[0][0] == "run-1"
        finally:
            anchor.close()
