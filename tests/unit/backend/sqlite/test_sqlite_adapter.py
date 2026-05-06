"""Unit tests for the SQLite backend adapter."""

import sqlite3
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion.sqlite_adapter import (
    SqliteAdapter,
    SqliteAdapterError,
    SqliteFactory,
    SqliteShardReader,
)


@pytest.fixture()
def mock_backend() -> MagicMock:
    """Mock ObstoreBackend for adapter tests that don't have S3."""
    with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as m:
        instance = m.return_value
        instance.put = MagicMock()
        instance.get = MagicMock()
        yield instance


def _sqlite_bytes(tmp_path: Path, rows: list[tuple[bytes, bytes]]) -> bytes:
    db_path = tmp_path / f"temp-{uuid.uuid4().hex}.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID")
    conn.executemany("INSERT INTO kv (k, v) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return db_path.read_bytes()


# ---------------------------------------------------------------------------
# KV adapter
# ---------------------------------------------------------------------------


class TestSqliteAdapter:
    def test_write_batch_and_read_back(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
            pairs = [(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3")]
            adapter.write_batch(pairs)

            checkpoint_id = adapter.checkpoint()
            assert checkpoint_id is not None
            assert len(checkpoint_id) == 64  # SHA-256 hex

        db_path = local_dir / "shard.db"
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT k, v FROM kv ORDER BY k").fetchall()
        conn.close()
        assert len(rows) == 3
        assert rows[0] == (b"key1", b"val1")

    def test_multiple_batches(self, tmp_path: Path, mock_backend: MagicMock) -> None:
        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
            adapter.write_batch([(b"a", b"1"), (b"b", b"2")])
            adapter.write_batch([(b"c", b"3"), (b"d", b"4")])
            adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        count = conn.execute("SELECT count(*) FROM kv").fetchone()[0]
        conn.close()
        assert count == 4

    def test_empty_batch_is_noop(self, tmp_path: Path, mock_backend: MagicMock) -> None:
        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
            adapter.write_batch([])
            cp = adapter.checkpoint()
            assert cp is not None  # still produces a valid hash

    def test_write_after_close_raises(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        local_dir = tmp_path / "shard"
        adapter = SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir)
        adapter.__enter__()
        adapter.checkpoint()
        adapter.close()

        with pytest.raises(SqliteAdapterError, match="already closed"):
            adapter.write_batch([(b"x", b"y")])

    def test_upsert_on_duplicate_key(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
            adapter.write_batch([(b"key1", b"old")])
            adapter.write_batch([(b"key1", b"new")])
            adapter.checkpoint()

        conn = sqlite3.connect(str(local_dir / "shard.db"))
        val = conn.execute("SELECT v FROM kv WHERE k = ?", (b"key1",)).fetchone()[0]
        conn.close()
        assert val == b"new"

    def test_flush_is_noop(self, tmp_path: Path, mock_backend: MagicMock) -> None:
        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
            adapter.flush()  # should not raise

    def test_close_without_checkpoint_uploads(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        """Calling close() directly (no checkpoint) should still upload."""
        local_dir = tmp_path / "shard"
        with SqliteAdapter(
            db_url="s3://test/shard",
            local_dir=local_dir,
            emit_btree_metadata=False,
        ) as adapter:
            adapter.write_batch([(b"key1", b"val1")])

        mock_backend.put.assert_called_once()
        # Verify the uploaded DB is readable
        conn = sqlite3.connect(str(local_dir / "shard.db"))
        val = conn.execute("SELECT v FROM kv WHERE k = ?", (b"key1",)).fetchone()[0]
        conn.close()
        assert val == b"val1"

    def test_close_retryable_on_upload_failure(self, tmp_path: Path) -> None:
        """If S3 upload fails, close() does not mark as closed — caller can retry."""
        local_dir = tmp_path / "shard"
        # Disable sidecar so this test focuses on the main shard.db retry
        # contract (sidecar emission is best-effort and tested separately).
        adapter = SqliteAdapter(
            db_url="s3://test/shard",
            local_dir=local_dir,
            emit_btree_metadata=False,
        )
        adapter.write_batch([(b"k", b"v")])
        adapter.checkpoint()

        with patch(
            "shardyfusion.sqlite_adapter.ObstoreBackend",
        ) as MockBackend:
            instance = MockBackend.return_value
            instance.put = MagicMock(side_effect=OSError("S3 down"))
            with pytest.raises(OSError, match="S3 down"):
                adapter.close()

        # Adapter is NOT marked closed — retry should work
        with patch(
            "shardyfusion.sqlite_adapter.ObstoreBackend",
        ) as MockBackend:
            instance = MockBackend.return_value
            instance.put = MagicMock()
            adapter.close()
        instance.put.assert_called_once()

    def test_factory_creates_adapter(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        factory = SqliteFactory()
        adapter = factory(db_url="s3://test/shard", local_dir=tmp_path / "shard")
        assert isinstance(adapter, SqliteAdapter)
        adapter.close()

    def test_factory_is_picklable(self) -> None:
        import pickle  # required: testing multiprocess worker serialization

        factory = SqliteFactory(page_size=8192, cache_size_pages=-4000)
        restored = pickle.loads(pickle.dumps(factory))
        assert restored.page_size == 8192
        assert restored.cache_size_pages == -4000


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
        conn.execute(
            "CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID"
        )
        conn.executemany(
            "INSERT INTO kv (k, v) VALUES (?, ?)",
            [(b"key1", b"val1"), (b"key2", b"val2")],
        )
        conn.commit()
        conn.close()
        return shard_dir

    def _make_reader(self, shard_dir: Path) -> SqliteShardReader:
        db_bytes = (shard_dir / "shard.db").read_bytes()
        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_bytes)
            return SqliteShardReader(
                db_url="s3://test/shard",
                local_dir=shard_dir,
                checkpoint_id=None,
            )

    def test_get_existing_key(self, shard_dir: Path) -> None:
        reader = self._make_reader(shard_dir)
        assert reader.get(b"key1") == b"val1"
        assert reader.get(b"key2") == b"val2"
        reader.close()

    def test_get_missing_key(self, shard_dir: Path) -> None:
        reader = self._make_reader(shard_dir)
        assert reader.get(b"missing") is None
        reader.close()

    def test_close_and_reuse_raises(self, shard_dir: Path) -> None:
        reader = self._make_reader(shard_dir)
        reader.close()
        with pytest.raises(SqliteAdapterError, match="already closed"):
            reader.get(b"key1")


class TestSqliteShardReaderDownloadCache:
    def test_redownloads_when_snapshot_identity_changes(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "reader-cache" / "shard=00000"
        db_v1 = _sqlite_bytes(tmp_path, [(b"key", b"old")])
        db_v2 = _sqlite_bytes(tmp_path, [(b"key", b"new")])

        with patch(
            "shardyfusion.sqlite_adapter.ObstoreBackend",
        ) as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(side_effect=[db_v1, db_v2])

            reader_v1 = SqliteShardReader(
                db_url="s3://bucket/run-1/shard=00000/attempt=00",
                local_dir=local_dir,
                checkpoint_id="ckpt-1",
            )
            assert reader_v1.get(b"key") == b"old"
            reader_v1.close()

            reader_v2 = SqliteShardReader(
                db_url="s3://bucket/run-2/shard=00000/attempt=00",
                local_dir=local_dir,
                checkpoint_id="ckpt-2",
            )
            assert reader_v2.get(b"key") == b"new"
            reader_v2.close()

        assert instance.get.call_count == 2

    def test_reuses_download_when_snapshot_identity_matches(
        self, tmp_path: Path
    ) -> None:
        local_dir = tmp_path / "reader-cache" / "shard=00000"
        db_v1 = _sqlite_bytes(tmp_path, [(b"key", b"stable")])

        with patch(
            "shardyfusion.sqlite_adapter.ObstoreBackend",
        ) as MockBackend:
            instance = MockBackend.return_value
            instance.get = MagicMock(return_value=db_v1)

            reader_v1 = SqliteShardReader(
                db_url="s3://bucket/run-1/shard=00000/attempt=00",
                local_dir=local_dir,
                checkpoint_id="ckpt-1",
            )
            assert reader_v1.get(b"key") == b"stable"
            reader_v1.close()

            reader_v2 = SqliteShardReader(
                db_url="s3://bucket/run-1/shard=00000/attempt=00",
                local_dir=local_dir,
                checkpoint_id="ckpt-1",
            )
            assert reader_v2.get(b"key") == b"stable"
            reader_v2.close()

        assert instance.get.call_count == 1


# ---------------------------------------------------------------------------
# B-tree metadata sidecar (writer-side)
# ---------------------------------------------------------------------------


class TestBtreemetaSidecarOnSqliteAdapter:
    """Sidecar emission contract on SqliteAdapter.close().

    Sidecar artifact is best-effort: any failure is logged at TRANSIENT
    severity but never blocks the main shard.db PUT.
    """

    def _put_keys(self, mock_backend: MagicMock) -> list[str]:
        """Return the list of keys passed to backend.put() in call order."""
        return [c.args[0] for c in mock_backend.put.call_args_list]

    def test_flag_off_uploads_only_db(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        local_dir = tmp_path / "shard"
        with SqliteAdapter(
            db_url="s3://test/shard",
            local_dir=local_dir,
            emit_btree_metadata=False,
        ) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.checkpoint()

        keys = self._put_keys(mock_backend)
        assert len(keys) == 1
        assert keys[0].endswith("/shard.db")

    def test_default_uploads_sidecar_then_db(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        # No explicit emit_btree_metadata kwarg — pin the default-on contract
        # so a future regression cannot silently flip the default.
        pytest.importorskip("apsw")
        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.checkpoint()

        keys = self._put_keys(mock_backend)
        assert len(keys) == 2, f"expected sidecar + db; got {keys!r}"
        assert keys[0].endswith("/shard.btreemeta")
        assert keys[1].endswith("/shard.db")
        # Content types
        sidecar_call, db_call = mock_backend.put.call_args_list
        assert sidecar_call.kwargs.get("content_type") == "application/octet-stream"
        assert db_call.kwargs.get("content_type") == "application/x-sqlite3"

    def test_flag_on_explicit_uploads_sidecar(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        pytest.importorskip("apsw")
        local_dir = tmp_path / "shard"
        with SqliteAdapter(
            db_url="s3://test/shard",
            local_dir=local_dir,
            emit_btree_metadata=True,
        ) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.checkpoint()

        keys = self._put_keys(mock_backend)
        assert any(k.endswith("/shard.btreemeta") for k in keys)
        assert any(k.endswith("/shard.db") for k in keys)

    def test_btreemeta_unavailable_silent_skip(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        from shardyfusion.sqlite_adapter import BtreeMetaUnavailableError

        local_dir = tmp_path / "shard"
        with patch(
            "shardyfusion.sqlite_adapter.extract_btree_metadata",
            side_effect=BtreeMetaUnavailableError("apsw_not_installed"),
        ):
            with SqliteAdapter(
                db_url="s3://test/shard", local_dir=local_dir
            ) as adapter:
                adapter.write_batch([(b"k", b"v")])
                adapter.checkpoint()

        # Sidecar skipped silently; main db still uploaded.
        keys = self._put_keys(mock_backend)
        assert keys == [k for k in keys if k.endswith("/shard.db")]
        assert len(keys) == 1

    def test_extraction_runtimeerror_silent_skip(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        local_dir = tmp_path / "shard"
        with patch(
            "shardyfusion.sqlite_adapter.extract_btree_metadata",
            side_effect=RuntimeError("boom"),
        ):
            with SqliteAdapter(
                db_url="s3://test/shard", local_dir=local_dir
            ) as adapter:
                adapter.write_batch([(b"k", b"v")])
                adapter.checkpoint()

        keys = self._put_keys(mock_backend)
        assert len(keys) == 1
        assert keys[0].endswith("/shard.db")

    def test_sidecar_put_failure_silent_skip(self, tmp_path: Path) -> None:
        # Make only the sidecar PUT fail (key ends with /shard.btreemeta);
        # the main /shard.db PUT must still succeed.
        pytest.importorskip("apsw")
        local_dir = tmp_path / "shard"

        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as MockBackend:
            instance = MockBackend.return_value

            calls: list[str] = []

            def put_side_effect(key: str, data: bytes, **kwargs: object) -> None:
                calls.append(key)
                if key.endswith("/shard.btreemeta"):
                    raise OSError("sidecar S3 failure")
                # main db PUT succeeds

            instance.put = MagicMock(side_effect=put_side_effect)

            with SqliteAdapter(
                db_url="s3://test/shard", local_dir=local_dir
            ) as adapter:
                adapter.write_batch([(b"k", b"v")])
                adapter.checkpoint()

            assert any(c.endswith("/shard.btreemeta") for c in calls)
            assert any(c.endswith("/shard.db") for c in calls)

    def test_sidecar_uploaded_first(
        self, tmp_path: Path, mock_backend: MagicMock
    ) -> None:
        pytest.importorskip("apsw")
        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://test/shard", local_dir=local_dir) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.checkpoint()

        keys = self._put_keys(mock_backend)
        # Sidecar must precede db (defensive ordering for direct-S3 consumers).
        sidecar_idx = next(
            i for i, k in enumerate(keys) if k.endswith("/shard.btreemeta")
        )
        db_idx = next(i for i, k in enumerate(keys) if k.endswith("/shard.db"))
        assert sidecar_idx < db_idx
