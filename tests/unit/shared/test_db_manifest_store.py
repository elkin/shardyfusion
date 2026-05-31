"""Tests for PostgresManifestStore with normalized schema.

Uses a SQLite-backed fake DB-API 2 connection that translates ``%s`` parameter
markers to ``?`` so we get real SQL execution without requiring PostgreSQL.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pytest

from shardyfusion.db_manifest_store import PostgresManifestStore
from shardyfusion.errors import ManifestParseError
from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

# ---------------------------------------------------------------------------
# SQLite-backed fake DB-API 2 connection
# ---------------------------------------------------------------------------

_PG_MARKER = re.compile(r"%s")


class _FakeCursor:
    """Thin wrapper that translates ``%s`` to ``?`` before executing."""

    def __init__(self, real: sqlite3.Cursor) -> None:
        self._real = real

    @staticmethod
    def _translate(sql: str) -> str:
        return _PG_MARKER.sub("?", sql)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._real.execute(self._translate(sql), params)

    def executemany(self, sql: str, seq: list[tuple[Any, ...]]) -> None:
        self._real.executemany(self._translate(sql), seq)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._real.fetchone()

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._real.fetchall()


class _FakeConnection:
    """Wraps a real sqlite3 connection with a ``_FakeCursor`` that handles ``%s``."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self._con = con

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._con.cursor())

    def commit(self) -> None:
        self._con.commit()

    def rollback(self) -> None:
        self._con.rollback()

    def close(self) -> None:
        pass  # keep the shared connection open


def _sqlite_ddl_fixup(store: PostgresManifestStore) -> None:
    """SQLite doesn't support TIMESTAMPTZ, JSONB, or REFERENCES.

    Re-create the tables with SQLite-compatible DDL using the real
    connection (bypassing the fake cursor).
    """
    conn = store._connection_factory()
    real_con: sqlite3.Connection = conn._con  # type: ignore[attr-defined]
    real_con.execute(f"DROP TABLE IF EXISTS {store._shards_table}")
    real_con.execute(f"DROP TABLE IF EXISTS {store._builds_table}")
    real_con.execute(f"DROP TABLE IF EXISTS {store._pointer_table_name}")

    real_con.execute(
        f"CREATE TABLE {store._builds_table} ("
        f"  run_id           TEXT PRIMARY KEY,"
        f"  created_at       TEXT NOT NULL DEFAULT (datetime('now')),"
        f"  num_dbs          INTEGER NOT NULL,"
        f"  s3_prefix        TEXT NOT NULL,"
        f"  key_col          TEXT NOT NULL DEFAULT '_key',"
        f"  db_path_template TEXT NOT NULL DEFAULT 'db={{db_id:05d}}',"
        f"  shard_prefix     TEXT NOT NULL DEFAULT 'shards',"
        f"  key_encoding     TEXT NOT NULL DEFAULT 'u64be',"
        f"  sharding         TEXT NOT NULL,"
        f"  custom           TEXT NOT NULL DEFAULT '{{}}'"
        f")"
    )
    real_con.execute(
        f"CREATE TABLE {store._shards_table} ("
        f"  run_id        TEXT NOT NULL,"
        f"  db_id         INTEGER NOT NULL,"
        f"  db_url        TEXT,"
        f"  attempt       INTEGER NOT NULL DEFAULT 0,"
        f"  row_count     INTEGER NOT NULL DEFAULT 0,"
        f"  db_bytes      INTEGER NOT NULL DEFAULT 0,"
        f"  checkpoint_id TEXT,"
        f"  min_key       TEXT,"
        f"  max_key       TEXT,"
        f"  writer_info   TEXT,"
        f"  PRIMARY KEY (run_id, db_id)"
        f")"
    )
    real_con.execute(
        f"CREATE TABLE {store._pointer_table_name} ("
        f"  updated_at    TEXT NOT NULL DEFAULT (datetime('now')),"
        f"  manifest_ref  TEXT NOT NULL"
        f")"
    )
    real_con.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{store._pointer_table_name}_updated_at "
        f"ON {store._pointer_table_name} (updated_at DESC)"
    )
    real_con.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> PostgresManifestStore:
    """Create a PostgresManifestStore backed by an in-memory SQLite DB."""
    con = sqlite3.connect(":memory:")
    fake = _FakeConnection(con)
    s = PostgresManifestStore(
        lambda: fake,
        table_name="sf_manifests",
        pointer_table_name="sf_pointer",
        ensure_table=False,
    )
    _sqlite_ddl_fixup(s)
    return s


def _build_meta(
    *,
    run_id: str = "run-1",
    num_dbs: int = 4,
    s3_prefix: str = "s3://bucket/prefix",
    strategy: ShardingStrategy = ShardingStrategy.HASH,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id=run_id,
        created_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC),
        num_dbs=num_dbs,
        s3_prefix=s3_prefix,
        key_col="_key",
        sharding=ManifestShardingSpec(
            strategy=strategy,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
        key_encoding=key_encoding,
    )


def _shard(
    db_id: int,
    *,
    db_url: str | None = None,
    row_count: int = 100,
    min_key: int | str | None = None,
    max_key: int | str | None = None,
    writer_info: WriterInfo | None = None,
) -> RequiredShardMeta:
    return RequiredShardMeta(
        db_id=db_id,
        db_url=db_url or f"s3://bucket/shards/db={db_id:05d}/data.sst",
        attempt=0,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=f"chk-{db_id}",
        writer_info=writer_info or WriterInfo(),
        db_bytes=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPublishAndLoadRoundTrip:
    def test_basic_round_trip(self, store: PostgresManifestStore) -> None:
        build = _build_meta()
        shards = [_shard(0), _shard(1), _shard(2), _shard(3)]
        custom = {"team": "data", "version": 42}

        ref = store.publish(
            run_id="run-1",
            required_build=build,
            shards=shards,
            custom=custom,
        )
        assert ref == "run-1"

        parsed = store.load_manifest(ref)
        assert parsed.required_build.run_id == "run-1"
        assert parsed.required_build.num_dbs == 4
        assert parsed.required_build.s3_prefix == "s3://bucket/prefix"
        assert parsed.required_build.key_encoding == KeyEncoding.U64BE
        assert parsed.required_build.sharding.strategy == ShardingStrategy.HASH
        assert len(parsed.shards) == 4
        assert parsed.custom == custom

    def test_shard_fields_preserved(self, store: PostgresManifestStore) -> None:
        build = _build_meta(num_dbs=2)
        shards = [
            _shard(0, row_count=500, min_key=10, max_key=999),
            _shard(1, row_count=300, min_key="alpha", max_key="zeta"),
        ]

        store.publish(run_id="run-1", required_build=build, shards=shards, custom={})
        parsed = store.load_manifest("run-1")

        s0, s1 = parsed.shards[0], parsed.shards[1]
        assert s0.db_id == 0
        assert s0.row_count == 500
        assert s0.min_key == 10
        assert s0.max_key == 999
        assert s0.checkpoint_id == "chk-0"

        assert s1.db_id == 1
        assert s1.min_key == "alpha"
        assert s1.max_key == "zeta"

    def test_writer_info_preserved(self, store: PostgresManifestStore) -> None:
        wi = WriterInfo(stage_id=3, task_attempt_id=7, attempt=1, duration_ms=1234)
        build = _build_meta(num_dbs=1)
        shards = [_shard(0, writer_info=wi)]

        store.publish(run_id="run-1", required_build=build, shards=shards, custom={})
        parsed = store.load_manifest("run-1")

        assert parsed.shards[0].writer_info.stage_id == 3
        assert parsed.shards[0].writer_info.task_attempt_id == 7
        assert parsed.shards[0].writer_info.duration_ms == 1234

    def test_empty_shards(self, store: PostgresManifestStore) -> None:
        build = _build_meta(num_dbs=4)
        store.publish(run_id="run-1", required_build=build, shards=[], custom={})
        parsed = store.load_manifest("run-1")
        assert parsed.shards == []

    def test_none_min_max_keys(self, store: PostgresManifestStore) -> None:
        build = _build_meta(num_dbs=1)
        shards = [_shard(0, min_key=None, max_key=None)]
        store.publish(run_id="run-1", required_build=build, shards=shards, custom={})
        parsed = store.load_manifest("run-1")
        assert parsed.shards[0].min_key is None
        assert parsed.shards[0].max_key is None


class TestIdempotentUpsert:
    def test_publish_same_run_id_overwrites(self, store: PostgresManifestStore) -> None:
        build = _build_meta(num_dbs=2)
        store.publish(
            run_id="run-1",
            required_build=build,
            shards=[_shard(0), _shard(1)],
            custom={"v": 1},
        )

        # Publish again with different data
        build2 = _build_meta(num_dbs=3)
        store.publish(
            run_id="run-1",
            required_build=build2,
            shards=[_shard(0), _shard(1), _shard(2)],
            custom={"v": 2},
        )

        parsed = store.load_manifest("run-1")
        assert parsed.required_build.num_dbs == 3
        assert len(parsed.shards) == 3
        assert parsed.custom == {"v": 2}


class TestLoadManifestNotFound:
    def test_raises_manifest_parse_error(self, store: PostgresManifestStore) -> None:
        with pytest.raises(ManifestParseError, match="not found"):
            store.load_manifest("nonexistent")


class TestListManifests:
    def test_list_with_limit(self, store: PostgresManifestStore) -> None:
        for i in range(5):
            build = _build_meta(run_id=f"run-{i}", num_dbs=1)
            # Vary created_at so ORDER BY created_at DESC is deterministic
            build.created_at = datetime(2026, 3, 28, 12, 0, i, tzinfo=UTC)
            store.publish(run_id=f"run-{i}", required_build=build, shards=[], custom={})

        refs = store.list_manifests(limit=3)
        assert len(refs) == 3
        # Most recent first
        assert refs[0].run_id == "run-4"

    def test_list_empty(self, store: PostgresManifestStore) -> None:
        refs = store.list_manifests()
        assert refs == []


class TestLoadCurrent:
    def test_from_pointer_table(self, store: PostgresManifestStore) -> None:
        build = _build_meta()
        store.publish(run_id="run-1", required_build=build, shards=[], custom={})

        current = store.load_current()
        assert current is not None
        assert current.ref == "run-1"

    def test_fallback_to_builds_table(self, store: PostgresManifestStore) -> None:
        # Insert a build row directly, bypassing pointer
        build = _build_meta()
        conn = store._connection_factory()
        real_con = conn._con  # type: ignore[attr-defined]
        bd = build.model_dump(mode="json")
        import json

        real_con.execute(
            f"INSERT INTO {store._builds_table} "
            f"  (run_id, created_at, num_dbs, s3_prefix, key_col,"
            f"   db_path_template, shard_prefix, key_encoding, sharding, custom) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run-1",
                bd["created_at"],
                bd["num_dbs"],
                bd["s3_prefix"],
                bd["key_col"],
                bd["db_path_template"],
                bd["shard_prefix"],
                bd["key_encoding"],
                json.dumps(bd["sharding"]),
                "{}",
            ),
        )
        real_con.commit()
        conn.close()

        current = store.load_current()
        assert current is not None
        assert current.ref == "run-1"

    def test_returns_none_when_empty(self, store: PostgresManifestStore) -> None:
        assert store.load_current() is None


class TestSetCurrent:
    def test_switches_pointer(self, store: PostgresManifestStore) -> None:
        build1 = _build_meta(run_id="run-1", num_dbs=1)
        store.publish(run_id="run-1", required_build=build1, shards=[], custom={})

        assert store.load_current().ref == "run-1"  # type: ignore[union-attr]

        # set_current always appends — the latest row wins
        store.set_current("run-1")
        current = store.load_current()
        assert current is not None
        assert current.ref == "run-1"


class TestCelShardingRoundTrip:
    def test_cel_sharding_spec(self, store: PostgresManifestStore) -> None:
        sharding = ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="shard_hash(key) % 3u",
            cel_columns={"key": "int"},
            hash_algorithm="xxh3_64",
        )
        build = _build_meta(num_dbs=3, strategy=ShardingStrategy.HASH)
        # Override with pre-built CEL sharding (can't pass through _build_meta)
        build.sharding = sharding

        store.publish(
            run_id="run-1",
            required_build=build,
            shards=[_shard(0), _shard(1), _shard(2)],
            custom={},
        )
        parsed = store.load_manifest("run-1")
        assert parsed.required_build.sharding.strategy == ShardingStrategy.CEL
        assert parsed.required_build.sharding.cel_expr == "shard_hash(key) % 3u"
        assert parsed.required_build.sharding.cel_columns == {"key": "int"}


class TestCustomFieldsRoundTrip:
    def test_nested_custom_dict(self, store: PostgresManifestStore) -> None:
        custom = {
            "owner": "data-team",
            "config": {"nested": True, "items": [1, 2, 3]},
        }
        build = _build_meta(num_dbs=1)
        store.publish(run_id="run-1", required_build=build, shards=[], custom=custom)
        parsed = store.load_manifest("run-1")
        assert parsed.custom == custom
