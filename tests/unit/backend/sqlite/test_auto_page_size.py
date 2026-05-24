"""Unit tests for the ``page_size="auto"`` adapter mode (post-write VACUUM)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion.sqlite_adapter import (
    SqliteAdapter,
    _maybe_repage_to_auto,
)
from shardyfusion.sqlite_page_size import recommend_page_size


def _read_page_size_from_header(db_path: Path) -> int:
    raw = db_path.read_bytes()
    encoded = int.from_bytes(raw[16:18], "big")
    return 65536 if encoded == 1 else encoded


class TestAutoModePicksFromValues:
    def test_small_values_stay_at_default(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteAdapter(
                db_url="s3://bucket/x",
                local_dir=local_dir,
                page_size="auto",
                emit_btree_metadata=False,
            )
            adapter.write_batch([(i.to_bytes(8, "big"), b"v" * 50) for i in range(100)])
            adapter.seal()

        assert _read_page_size_from_header(local_dir / "shard.db") == 4096

    @pytest.mark.parametrize(
        "value_bytes,expected_page_size",
        [
            (1500, 8192),
            (4000, 16384),
            (8000, 32768),
            (12000, 65536),
        ],
    )
    def test_auto_picker_matches_recommend(
        self,
        tmp_path: Path,
        value_bytes: int,
        expected_page_size: int,
    ) -> None:
        # Sanity-check that the picker and the helper agree on the boundary.
        assert recommend_page_size(p95_value_bytes=value_bytes) == expected_page_size

        local_dir = tmp_path / "shard"
        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteAdapter(
                db_url="s3://bucket/x",
                local_dir=local_dir,
                page_size="auto",
                emit_btree_metadata=False,
            )
            adapter.write_batch(
                [(i.to_bytes(8, "big"), b"v" * value_bytes) for i in range(50)]
            )
            adapter.seal()

        assert _read_page_size_from_header(local_dir / "shard.db") == expected_page_size


class TestRepageHelperEdgeCases:
    def test_repage_noop_on_empty_kv(self, tmp_path: Path) -> None:
        # Empty kv → picker has nothing to sample → no rewrite.
        local_dir = tmp_path / "shard"
        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteAdapter(
                db_url="s3://bucket/x",
                local_dir=local_dir,
                page_size="auto",
                emit_btree_metadata=False,
            )
            adapter.seal()
        assert _read_page_size_from_header(local_dir / "shard.db") == 4096

    def test_repage_direct_call_is_idempotent(self, tmp_path: Path) -> None:
        """A second invocation against an already-repaged file is a no-op."""
        import sqlite3

        db_path = tmp_path / "tiny.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA page_size = 16384")
        conn.execute(
            "CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID"
        )
        conn.execute("INSERT INTO kv VALUES (?, ?)", (b"k", b"x" * 4000))
        conn.close()

        before = _read_page_size_from_header(db_path)
        _maybe_repage_to_auto(db_path, db_url="s3://test/x")
        after = _read_page_size_from_header(db_path)
        # Picker recommends 16384 for 4000-byte values; same as current.
        assert before == after == 16384


class TestSqliteVecAdapterAutoMode:
    """Auto-mode VACUUM must keep sqlite-vec virtual tables resolvable.

    SQLite VACUUM walks every schema object during the rewrite; the
    ``vec_index`` virtual table is unresolvable without the sqlite-vec
    extension loaded on the same connection.  The unified adapter wires
    ``_open_sqlite_vec_connection`` into :func:`_maybe_repage_to_auto`
    for exactly this reason — this test exercises the wire-up end to
    end.
    """

    def test_auto_mode_rewrites_unified_shard(self, tmp_path: Path) -> None:
        pytest.importorskip("sqlite_vec")
        import numpy as np

        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecAdapter

        local_dir = tmp_path / "vec_shard"
        spec = VectorSpec(dim=8, metric="cosine", index_type="flat")
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/vec",
                local_dir=local_dir,
                vector_spec=spec,
                page_size="auto",
                emit_btree_metadata=False,
            )
            # ~8 KB KV values force a repage above 4 KB inline threshold.
            adapter.write_batch(
                [(i.to_bytes(8, "big"), b"v" * 8000) for i in range(20)]
            )
            ids = np.arange(20, dtype=np.int64)
            vecs = np.random.rand(20, 8).astype(np.float32)
            adapter.write_vector_batch(ids, vecs)
            adapter.seal()

        # VACUUM with the sqlite-vec extension loaded must succeed, and
        # the post-rewrite file must report the recommended page_size in
        # its SQLite header.
        assert _read_page_size_from_header(local_dir / "shard.db") == 32768

    def test_auto_mode_kv_only_workload(self, tmp_path: Path) -> None:
        """Vec adapter with empty vec_index still routes the repage through
        the sqlite-vec connection opener (the schema still has vec0)."""
        pytest.importorskip("sqlite_vec")

        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecAdapter

        local_dir = tmp_path / "vec_shard_kv_only"
        spec = VectorSpec(dim=8, metric="cosine", index_type="flat")
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/vec2",
                local_dir=local_dir,
                vector_spec=spec,
                page_size="auto",
                emit_btree_metadata=False,
            )
            adapter.write_batch([(i.to_bytes(8, "big"), b"v" * 50) for i in range(50)])
            adapter.seal()

        # Tiny values fit inline at 4 KB; picker stays put.  VACUUM
        # still runs through the vec-aware opener because the vec0
        # schema object is present.
        assert _read_page_size_from_header(local_dir / "shard.db") == 4096
